"""VLA executor — RTCExecutor and camera helpers."""

from __future__ import annotations

import json
import math
import os
import re
import signal
import subprocess
import time
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any

# The Pi0.5 checkpoints here rely on local Hugging Face cache during startup.
# Set offline flags before importing LeRobot/Transformers so Hub constants are
# initialized in offline mode for this process.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
from loguru import logger

try:
    from lerobot.utils.import_utils import register_third_party_plugins
    register_third_party_plugins()
except ImportError:
    from lerobot.utils.import_utils import register_third_party_devices
    register_third_party_devices()

from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import RTCAttentionSchedule
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.rtc.action_queue import ActionQueue
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.rtc.latency_tracker import LatencyTracker
from lerobot.processor.factory import (
    make_default_robot_action_processor,
    make_default_robot_observation_processor,
)
from lerobot.robots.utils import make_robot_from_config
from lerobot_robot_piper.config_piper import PiperConfig
from vla_pi05_piper.patches import apply_all_patches, install_warmup, reset_state


# ── Standby pose helpers ──────────────────────────────────────────────────────

_STANDBY_POSE_CANDIDATES = [
    Path(__file__).with_name("standby_pose.json"),
    Path(__file__).parent.parent.parent / "configs" / "standby_pose.json",
]


def load_standby_pose(path: Path | str | None = None) -> dict[str, float]:
    """Load standby pose JSON. Searches default candidates if path is None."""
    if path is not None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Standby pose not found: {p}")
        with open(p) as f:
            return json.load(f)
    for p in _STANDBY_POSE_CANDIDATES:
        if p.exists():
            with open(p) as f:
                return json.load(f)
    raise FileNotFoundError(
        f"Standby pose not found at {_STANDBY_POSE_CANDIDATES[0]} or {_STANDBY_POSE_CANDIDATES[1]}"
    )


def _force_hf_offline_mode() -> None:
    """Force local-only Hugging Face loading for policy startup in this process."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        import huggingface_hub.constants as hf_constants
        hf_constants.HF_HUB_OFFLINE = True
    except Exception:
        pass
    try:
        from transformers.utils import hub as tf_hub
        tf_hub.HF_HUB_OFFLINE = True
    except Exception:
        pass


_force_hf_offline_mode()


def _kill_busy_realsense_holders(error_text: str) -> bool:
    match = re.search(r"RealSenseCamera\(([^)]+)\)", error_text)
    if match is None:
        return False
    serial = match.group(1).strip()
    try:
        import pyrealsense2 as rs
        port = next(
            str(device.get_info(rs.camera_info.physical_port))
            for device in rs.context().query_devices()
            if str(device.get_info(rs.camera_info.serial_number)).strip() == serial
        )
        root = next(parent for parent in [Path(port), *Path(port).parents] if re.fullmatch(r"\d+-\d+(\.\d+)*", parent.name))
        nodes = sorted({f"/dev/{path.name}" for path in root.glob("**/video4linux/video*")})
        proc = subprocess.run(["lsof", "-t", "--", *nodes], capture_output=True, text=True, timeout=3, check=False) if nodes else None
    except Exception:
        return False
    pids = sorted({int(line) for line in (proc.stdout if proc else "").splitlines() if line.strip().isdigit() and int(line) != os.getpid()})
    if not pids:
        return False
    for pid in pids:
        os.kill(pid, signal.SIGKILL)
    time.sleep(1.0)
    logger.warning("Killed RealSense holders for {}: pids={} nodes={}", serial, pids, nodes)
    return True
# ── Robot wrapper ─────────────────────────────────────────────────────────────


class _RobotWrapper:
    """Thread-safe wrapper around a LeRobot Robot instance."""

    def __init__(self, robot: Any):
        self.robot = robot
        self._lock = Lock()

    def get_observation(self) -> dict:
        with self._lock:
            return self.robot.get_observation()

    def send_action(self, action: Any) -> None:
        with self._lock:
            self.robot.send_action(action)

    @property
    def observation_features(self) -> dict:
        return self.robot.observation_features

    @property
    def action_features(self) -> list[str]:
        return self.robot.action_features


# ── RTCExecutor ───────────────────────────────────────────────────────────────


class RTCExecutor:
    """RTC inference + Piper arm control.

    Lifecycle:
        executor = RTCExecutor(policy_path)
        executor.init()                         # load policy, connect robot
        executor.start("Pick up the red cup.")  # start inference threads
        executor.stop()                         # stop, hold position
        executor.start("Place cup on coaster.") # new task
        executor.cleanup()                      # disconnect robot
    """

    def __init__(
        self,
        policy_path: str,
        device: str = "cuda",
        fps: int = 30,
        can_interface: str = "can0",
        cameras: dict[str, dict] | None = None,
        rtc_execution_horizon: int = 25,
    ):
        self.policy_path = policy_path
        self.device = device
        self.fps = fps
        self.can_interface = can_interface
        self.rtc_execution_horizon = rtc_execution_horizon
        self._queue_threshold = int(os.environ.get("ACTION_QUEUE_THRESHOLD", "35"))

        self.cameras_cfg = cameras or {
            "wrist": {"serial_number_or_name": "109622071821", "width": 640, "height": 480, "fps": 30},
            "main":  {"serial_number_or_name": "918512071262", "width": 640, "height": 480, "fps": 30},
        }

        self._policy = None
        self._robot = None
        self._robot_wrapper: _RobotWrapper | None = None
        self._preprocessor = None
        self._postprocessor = None
        self._obs_processor = None
        self._action_processor = None
        self._dataset_features = None
        self._rtc_config: RTCConfig | None = None
        self._standby_pose: dict[str, float] | None = None

        self._latest_frame: dict[str, np.ndarray] = {}
        self._frame_lock = Lock()

        self._task: str = ""
        self._shutdown_event = Event()
        self._get_actions_thread: Thread | None = None
        self._actor_thread: Thread | None = None
        self._running = False
        self._last_error: str | None = None
        self._lock = Lock()

    # ── Public API ────────────────────────────────────────────────────────────

    def init(self) -> None:
        """Load policy and connect robot. Call once before start()."""
        try:
            self._init_robot()
        except Exception as e:
            if not _kill_busy_realsense_holders(str(e)):
                raise RuntimeError(
                    f"robot connection failed (can_interface={self.can_interface}): {e}"
                ) from e
            logger.warning("Retrying robot init after releasing busy RealSense holders")
            try:
                self._disconnect_without_vendor_home()
            except Exception:
                self._robot = None
                self._robot_wrapper = None
            try:
                self._init_robot()
            except Exception as retry_error:
                raise RuntimeError(
                    f"robot connection failed (can_interface={self.can_interface}): {retry_error}"
                ) from retry_error

        self._load_standby_pose()
        try:
            self._init_policy()
        except Exception as e:
            raise RuntimeError(
                f"policy loading failed (policy_path={self.policy_path}): {e}"
            ) from e

        if not self.go_to_standby():
            raise RuntimeError("failed to reach standby position during initialization")
        logger.info("RTCExecutor initialized")

    def start(self, instruction: str) -> None:
        """Start executing a new instruction. Stops any running task first."""
        self.stop()
        if self._get_actions_thread is not None or self._actor_thread is not None:
            message = "Previous worker threads are still alive"
            self._record_fatal_error(message)
            raise RuntimeError(message)
        if self._policy is None or self._robot_wrapper is None:
            raise RuntimeError("Executor is not initialized")

        with self._lock:
            self._last_error = None
        self._policy.reset()

        if self._robot is not None:
            self._robot._last_cmd_oriented_deg = None
            self._robot._last_cmd_t = None

        reset_state()

        with self._lock:
            self._task = instruction
            self._running = True

        self._shutdown_event = Event()
        action_queue = ActionQueue(self._rtc_config)

        self._get_actions_thread = Thread(
            target=self._get_actions_loop, args=(action_queue,), daemon=True, name="GetActions",
        )
        self._actor_thread = Thread(
            target=self._actor_loop, args=(action_queue,), daemon=True, name="Actor",
        )
        self._get_actions_thread.start()
        self._actor_thread.start()
        logger.info("Executor started: '{}'", instruction)

    def stop(self, return_home: bool = False) -> None:
        """Stop current execution."""
        self._shutdown_event.set()
        with self._lock:
            self._running = False

        if self._get_actions_thread and self._get_actions_thread.is_alive():
            self._get_actions_thread.join(timeout=3.0)
        if self._actor_thread and self._actor_thread.is_alive():
            self._actor_thread.join(timeout=3.0)

        if self._get_actions_thread and not self._get_actions_thread.is_alive():
            self._get_actions_thread = None
        if self._actor_thread and not self._actor_thread.is_alive():
            self._actor_thread = None
        if self._get_actions_thread is not None or self._actor_thread is not None:
            logger.warning("Worker threads did not stop cleanly")

        with self._frame_lock:
            self._latest_frame = {}

        if return_home:
            self.go_to_standby()

    def cleanup(self) -> None:
        """Stop, move to standby then home, and disconnect robot."""
        self.stop()
        if self._robot is None:
            return

        logger.info("CLI shutdown cleanup: moving to standby")
        self.go_to_standby()
        time.sleep(0.5)
        logger.info("CLI shutdown cleanup: moving to home")
        self.go_to_home()
        logger.info("CLI shutdown cleanup: disconnecting without vendor home")
        self._disconnect_without_vendor_home()

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._running

    @property
    def last_error(self) -> str | None:
        with self._lock:
            return self._last_error

    def get_observation(self) -> dict[str, np.ndarray]:
        """Return latest cached camera frames (no hardware read)."""
        with self._frame_lock:
            return dict(self._latest_frame)

    def _move_joint_pose(
        self,
        target: dict[str, float],
        *,
        label: str,
        timeout_s: float,
    ) -> bool:
        """Move to a calibrated joint pose using send_action/get_observation feedback."""
        if self._robot is None:
            return False

        if hasattr(self._robot, "_last_cmd_oriented_deg"):
            self._robot._last_cmd_oriented_deg = None
        if hasattr(self._robot, "_last_cmd_t"):
            self._robot._last_cmd_t = None

        target = {key: float(value) for key, value in target.items()}
        joint_keys = [f"joint_{i}.pos" for i in range(1, 7)]
        deadline = time.time() + timeout_s
        last_obs: dict[str, Any] | None = None
        while time.time() < deadline:
            self._robot.send_action(target)
            time.sleep(1.0 / 30.0)
            last_obs = self._robot.get_observation()
            if all(abs(float(last_obs[key]) - float(target[key])) <= 0.5 for key in joint_keys):
                logger.info("Moved to {} position", label)
                return True

        if last_obs is not None:
            logger.warning(
                "Failed to reach {} position: target={}, actual={}",
                label,
                [round(float(target[key]), 2) for key in joint_keys],
                [round(float(last_obs[key]), 2) for key in joint_keys],
            )
        return False

    def go_to_standby(self) -> bool:
        """Move arm to standby pose in calibrated action space."""
        if self._standby_pose is None:
            if self._robot is None:
                return False
            try:
                self._robot._return_to_home()
            except Exception as e:
                logger.warning("Return to home failed: {}", e)
                return False
            return True
        return self._move_joint_pose(self._standby_pose, label="standby", timeout_s=10.0)

    def go_to_home(self) -> bool:
        """Move arm to calibrated home in the same joint space as standby."""
        return self._move_joint_pose(
            {f"joint_{i}.pos": 0.0 for i in range(1, 7)},
            label="home",
            timeout_s=20.0,
        )

    # ── Init helpers ──────────────────────────────────────────────────────────

    def _load_standby_pose(self) -> None:
        try:
            self._standby_pose = load_standby_pose()
            logger.info("Standby pose loaded")
        except FileNotFoundError as e:
            self._standby_pose = None
            logger.warning("{} — will use _return_to_home()", e)

    def _init_robot(self) -> None:
        cameras = {name: RealSenseCameraConfig(**cfg) for name, cfg in self.cameras_cfg.items()}

        # calibration_fpath = calibration_dir / f"{id}.json", so point to robots/piper/
        _default_cal = Path(__file__).parent.parent.parent / "configs" / ".lerobot_calibration" / "robots" / "piper"
        calibration_dir = Path(os.environ.get("HF_LEROBOT_CALIBRATION", str(_default_cal)))

        robot_config = PiperConfig(
            id="follower",
            can_interface=self.can_interface,
            use_degrees=True,
            ctrl_speed=30,
            max_joint_vel_deg_s=360,
            max_joint_step_deg=15,
            cameras=cameras,
            calibration_dir=calibration_dir,
        )
        logger.info("Connecting Piper robot via {}", self.can_interface)
        self._robot = make_robot_from_config(robot_config)
        self._robot.connect()
        self._robot_wrapper = _RobotWrapper(self._robot)
        self._obs_processor = make_default_robot_observation_processor()
        self._action_processor = make_default_robot_action_processor()
        self._dataset_features = hw_to_dataset_features(
            self._robot_wrapper.observation_features, "observation"
        )
        logger.info("Robot connected via {}", self.can_interface)

    def _init_policy(self) -> None:
        _force_hf_offline_mode()
        apply_all_patches(
            tcs_min_overlap=int(os.environ.get("TCS_MIN_OVERLAP", "8")),
            ema_alpha=float(os.environ.get("EMA_ALPHA", "0.5")),
            rtc_guidance=True,
        )
        config = PreTrainedConfig.from_pretrained(self.policy_path, local_files_only=True)
        policy_class = get_policy_class(config.type)
        self._policy = policy_class.from_pretrained(self.policy_path, config=config, local_files_only=True)

        self._rtc_config = RTCConfig(
            enabled=True,
            execution_horizon=self.rtc_execution_horizon,
            max_guidance_weight=5.0,
            prefix_attention_schedule=RTCAttentionSchedule.EXP,
        )
        self._policy.config.rtc_config = self._rtc_config
        self._policy.init_rtc_processor()
        self._policy = self._policy.to(self.device)
        self._policy.eval()

        self._preprocessor, self._postprocessor = make_pre_post_processors(
            policy_cfg=config,
            pretrained_path=self.policy_path,
            dataset_stats=None,
            preprocessor_overrides={"device_processor": {"device": self.device}},
        )
        logger.info("Policy loaded: {} on {}", self.policy_path, self.device)

        # Trigger torch.compile JIT warmup during init so the first real task
        # doesn't pay the compilation cost. install_warmup wraps predict_action_chunk
        # and runs 2 dummy passes on first call; we call it once here explicitly.
        logger.info("Running torch.compile warmup passes...")
        install_warmup(self._policy)
        # Force warmup now by calling the wrapper with a dummy observation.
        # Build minimal fake obs matching the policy's expected input shape.
        try:
            import torch as _torch
            _dummy_obs = self._robot_wrapper.get_observation()
            _dummy_processed = self._obs_processor(_dummy_obs)
            _dummy_frame = build_dataset_frame(self._dataset_features, _dummy_processed, prefix="observation")
            for _name in _dummy_frame:
                _dummy_frame[_name] = _torch.from_numpy(_dummy_frame[_name])
                if "image" in _name:
                    _dummy_frame[_name] = _dummy_frame[_name].float() / 255.0
                    _dummy_frame[_name] = _dummy_frame[_name].permute(2, 0, 1).contiguous()
                _dummy_frame[_name] = _dummy_frame[_name].unsqueeze(0).to(self.device)
            _dummy_frame["task"] = ["Put the bottle into the white box."]
            _dummy_frame["robot_type"] = getattr(self._robot, "name", "")
            _dummy_preprocessed = self._preprocessor(_dummy_frame)
            self._policy.predict_action_chunk(_dummy_preprocessed, inference_delay=0, prev_chunk_left_over=None)
            logger.info("Warmup complete")
        except Exception as _e:
            logger.warning("Warmup failed (non-fatal): {}", _e)

    # ── Inference thread ──────────────────────────────────────────────────────

    def _get_actions_loop(self, action_queue: ActionQueue) -> None:
        try:
            latency_tracker = LatencyTracker()
            time_per_chunk = 1.0 / self.fps
            while not self._shutdown_event.is_set():
                if action_queue.qsize() <= self._queue_threshold:
                    t0 = time.perf_counter()
                    action_index_before = action_queue.get_action_index()
                    prev_actions = action_queue.get_left_over()
                    inference_delay = math.ceil(latency_tracker.max() / time_per_chunk)

                    obs = self._robot_wrapper.get_observation()
                    self._cache_frames(obs)
                    obs_processed = self._obs_processor(obs)

                    obs_frame = build_dataset_frame(
                        self._dataset_features, obs_processed, prefix="observation"
                    )
                    for name in obs_frame:
                        obs_frame[name] = torch.from_numpy(obs_frame[name])
                        if "image" in name:
                            obs_frame[name] = obs_frame[name].float() / 255.0
                            obs_frame[name] = obs_frame[name].permute(2, 0, 1).contiguous()
                        obs_frame[name] = obs_frame[name].unsqueeze(0).to(self.device)

                    with self._lock:
                        obs_frame["task"] = [self._task]
                    obs_frame["robot_type"] = getattr(self._robot, "name", "")

                    preprocessed = self._preprocessor(obs_frame)
                    actions = self._policy.predict_action_chunk(
                        preprocessed,
                        inference_delay=inference_delay,
                        prev_chunk_left_over=prev_actions,
                    )

                    original_actions = actions.squeeze(0).clone()
                    postprocessed = self._postprocessor(actions).squeeze(0)

                    new_latency = time.perf_counter() - t0
                    latency_tracker.add(new_latency)

                    # Use actual action index diff as real_delay to avoid mismatch warning
                    action_index_after = action_queue.get_action_index()
                    real_delay = action_index_after - action_index_before

                    action_queue.merge(
                        original_actions, postprocessed,
                        real_delay,
                        action_index_before,
                    )
                else:
                    time.sleep(0.1)
        except Exception as e:
            logger.error("[GetActions] Fatal: {}", e)
            self._record_fatal_error(f"GetActions fatal: {e}")

    # ── Actor thread ──────────────────────────────────────────────────────────

    def _actor_loop(self, action_queue: ActionQueue) -> None:
        try:
            interval = 1.0 / self.fps
            while not self._shutdown_event.is_set():
                t = time.perf_counter()
                action = action_queue.get()
                if action is not None:
                    action = action.cpu()
                    action_dict = {
                        key: action[i].item()
                        for i, key in enumerate(self._robot_wrapper.action_features)
                    }
                    processed = self._action_processor((action_dict, None))
                    self._robot_wrapper.send_action(processed)
                time.sleep(max(0, interval - (time.perf_counter() - t) - 0.001))
        except Exception as e:
            logger.error("[Actor] Fatal: {}", e)
            self._record_fatal_error(f"Actor fatal: {e}")

    def _cache_frames(self, obs: dict[str, Any]) -> None:
        cache = {}
        for key, val in obs.items():
            if isinstance(val, torch.Tensor):
                cache[key] = val.cpu().numpy()
            elif isinstance(val, np.ndarray):
                cache[key] = val
        with self._frame_lock:
            self._latest_frame = cache

    def _disconnect_without_vendor_home(self) -> None:
        robot = self._robot
        if robot is None:
            return
        if getattr(robot, "name", None) != "piper":
            robot.disconnect()
            self._robot = None
            self._robot_wrapper = None
            logger.info("Robot disconnected")
            return

        iface = getattr(robot, "_iface", None)
        if iface is not None:
            try:
                iface.disconnect()
            except Exception as e:
                logger.warning("Robot interface disconnect failed: {}", e)
            robot._iface = None

        for attr in (
            "_last_status_deg",
            "_last_obs_motors",
            "_last_cmd_oriented_deg",
            "_last_cmd_t",
            "_last_lowspd_t",
            "_last_lowspd",
            "_freeze_until_t",
            "_tracking_err_t0",
            "_teleop_base_oriented_deg",
        ):
            if hasattr(robot, attr):
                setattr(robot, attr, None)

        recover_times = getattr(robot, "_recover_times", None)
        if recover_times is not None:
            try:
                recover_times.clear()
            except Exception:
                pass

        for cam in getattr(robot, "cameras", {}).values():
            try:
                cam.disconnect()
            except Exception:
                pass

        self._robot = None
        self._robot_wrapper = None
        logger.info("Robot disconnected without vendor home")

    def _record_fatal_error(self, message: str) -> None:
        with self._lock:
            self._running = False
            self._last_error = message
        self._shutdown_event.set()



def capture_live_frames(executor: RTCExecutor) -> dict[str, np.ndarray]:
    """Get current camera frames. Uses cached frames when running, reads hardware when idle."""
    if executor.is_running:
        return executor.get_observation()

    if executor._robot_wrapper is not None:
        try:
            obs = executor._robot_wrapper.get_observation()
            result = {}
            for key, val in obs.items():
                if isinstance(val, torch.Tensor):
                    result[key] = val.cpu().numpy()
                elif isinstance(val, np.ndarray):
                    result[key] = val
            return result
        except Exception as e:
            logger.warning("Failed to get observation from robot wrapper: {}", e)

    return {}
