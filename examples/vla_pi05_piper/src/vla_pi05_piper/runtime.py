"""VLA Runtime - State management and verification loop."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from threading import Lock, Thread
from typing import Any, Awaitable, Callable
import httpx
import numpy as np
from loguru import logger
from PIL import Image

from vla_pi05_piper.executor import RTCExecutor, capture_live_frames

DEFAULT_VERIFY_API_BASE = "https://openrouter.ai/api/v1"
DEFAULT_VERIFY_ARK_API_BASE = "https://ark.cn-beijing.volces.com/api/v3"
DEFAULT_VERIFY_MODEL = "anthropic/claude-3.5-sonnet"
DEFAULT_VERIFY_HTTP_TIMEOUT_S = 30.0
DEFAULT_VERIFY_INTERVAL_S = 2.0
DEFAULT_VERIFY_START_DELAY_S = 3.0
DEFAULT_VERIFY_MAX_TOKENS = 256
DEFAULT_VERIFY_PROVIDER = "openrouter"
VERIFY_PROMPT_PATH = Path(__file__).resolve().parents[4] / "examples" / "missions" / "desk_cleanup" / "verify.md"
VERIFY_MAIN_MAX_EDGE = 384
VERIFY_WRIST_MAX_EDGE = 224
VERIFY_PROVIDERS = {"openrouter", "local", "ark"}
VERIFY_COMPLETED_STREAK_REQUIRED = 1
VERIFY_VERDICT_LINE = re.compile(r"^(completed|continue)\s*[.!?]*$", re.IGNORECASE)


def _verify_debug_base_dir() -> Path:
    """Verify debug output root (<repo>/debug/verify); JPEG/JSON names carry timestamps."""
    return Path(__file__).resolve().parents[4] / "debug" / "verify"


def _start_camera_video_recording(
    output_dir: Path,
    frame_getter: Callable[[], dict[str, np.ndarray]],
    fps: int = 30,
    name_suffix: str | None = None,
) -> None:
    """Spawn a daemon thread that pipes camera frames to ffmpeg (H.264/MP4).

    Runs until the process exits. Each camera gets its own ffmpeg subprocess
    started lazily on the first frame. Only records keys containing 'main' or 'wrist'.
    """
    procs: dict[str, subprocess.Popen] = {}

    def get_or_open(cam: str, h: int, w: int) -> subprocess.Popen | None:
        if cam in procs:
            return procs[cam]
        # yuv420p requires even dimensions
        ew, eh = w + w % 2, h + h % 2
        stem = f"camera_{cam}_{name_suffix}" if name_suffix else f"camera_{cam}"
        out_file = output_dir / f"{stem}.mp4"
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{w}x{h}", "-pix_fmt", "rgb24", "-r", str(fps),
            "-i", "pipe:0",
            "-vcodec", "libx264", "-pix_fmt", "yuv420p",
            "-vf", f"scale={ew}:{eh}",
            "-crf", "23", "-preset", "fast", "-movflags", "+faststart",
            str(out_file),
        ]
        try:
            p = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            procs[cam] = p
            logger.info("VideoRecorder: {} → {}", cam, out_file)
            return p
        except FileNotFoundError:
            logger.error("VideoRecorder: ffmpeg not found — install ffmpeg to enable video recording")
            return None
        except Exception as e:
            logger.error("VideoRecorder: ffmpeg error for {}: {}", cam, e)
            return None

    def loop() -> None:
        interval = 1.0 / fps
        while True:
            t = time.perf_counter()
            try:
                for key, frame in frame_getter().items():
                    kl = key.lower()
                    cam = "main" if "main" in kl else "wrist" if "wrist" in kl else None
                    if cam is None or frame.ndim != 3 or frame.shape[2] != 3:
                        continue
                    if frame.dtype != np.uint8:
                        frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
                    h, w = frame.shape[:2]
                    p = get_or_open(cam, h, w)
                    if p and p.poll() is None:
                        try:
                            p.stdin.write(frame.tobytes())
                        except Exception:
                            procs.pop(cam, None)
            except Exception as e:
                logger.warning("VideoRecorder: {}", e)
            time.sleep(max(0.0, interval - (time.perf_counter() - t)))

    Thread(target=loop, daemon=True, name="VideoRecorder").start()
    logger.info("VideoRecorder started → {}", output_dir)


class VLARuntime:
    """VLA Runtime - manages state and verification loop."""

    def __init__(
        self,
        executor: RTCExecutor,
        verify_config: dict[str, Any],
        notification_callback: Callable[[dict], Awaitable[None]] | None = None,
        verify_prompt_path: Path | None = None,
    ):
        self._executor = executor
        self._verify_config = verify_config
        self._notification_callback = notification_callback
        self._verify_prompt_path = verify_prompt_path
        self._verify_debug = bool(self._verify_config.get("debug", False))
        self._validate_verify_config()
        self._verify_interval_s = self._load_verify_interval_s()
        self._verify_start_delay_s = self._load_verify_start_delay_s()
        self._verify_max_tokens = self._load_verify_max_tokens()
        self._verify_trust_env_proxy = self._load_verify_trust_env_proxy()
        self._verify_debug_dir: Path | None = None
        self._verify_debug_stamp: str | None = None
        self._verify_video_recording_started = False

        http_timeout_s = float(self._verify_config.get("http_timeout_s", DEFAULT_VERIFY_HTTP_TIMEOUT_S))
        self._verify_http_timeout_s = http_timeout_s
        self._http_client = httpx.AsyncClient(
            timeout=http_timeout_s,
            trust_env=self._verify_trust_env_proxy,
            http2=True,
        )

        # A single task snapshot is the source of truth for status and terminal state.
        self._state = "idle"
        self._instruction: str | None = None
        self._completion_mode: str | None = None
        self._task_revision = 0
        self._task_start_time: float | None = None
        self._task_end_time: float | None = None
        self._error: str | None = None

        self._baseline_frames: dict[str, np.ndarray] = {}
        self._baseline_main_b64: str | None = None
        self._baseline_wrist_b64: str | None = None
        self._verify_system_prompt: str | None = None
        self._verify_mode: str = "standalone"

        self._verify_task: asyncio.Task | None = None
        self._verify_step = 0
        self._last_verify_verdict: str | None = None
        self._last_verify_reason: str | None = None
        self._last_verify_latency_ms: int | None = None
        self._last_verify_at: str | None = None
        self._last_verify_error: str | None = None
        self._completed_verify_streak = 0
        self._lock = Lock()

    async def observe(self, cameras: list[str] | None = None) -> dict:
        """Capture current scene from cameras."""
        if cameras is None:
            cameras = ["main"]

        frames = capture_live_frames(self._executor)
        if not frames:
            return {
                "status": "error",
                "error": "No camera frames available. Is the robot connected?",
            }

        filtered_frames: dict[str, np.ndarray] = {}
        for key, frame in frames.items():
            key_lower = key.lower()
            if any(cam.lower() in key_lower for cam in cameras):
                filtered_frames[key] = frame

        if not filtered_frames:
            return {
                "status": "error",
                "error": f"No frames for cameras {cameras}. Available: {list(frames.keys())}",
            }

        status_text = self._build_status_text()
        frame_info = ", ".join(f"{k}:{v.shape}" for k, v in filtered_frames.items())
        status_text += f"\nframes: {frame_info}"

        return {
            "status": "captured",
            "cameras": cameras,
            "frames": filtered_frames,
            "status_text": status_text,
        }

    async def start_task(
        self,
        instruction: str,
        completion_mode: str = "single",
        verify_mode: str = "standalone",
    ) -> dict:
        """Start executing a task.

        verify_mode:
          "standalone" — background verify loop runs automatically (default, for mcp_repl / direct use)
          "manual"     — no background loop; caller drives verify via verify_once()
        """
        if verify_mode not in ("standalone", "manual"):
            return {"error": f"Invalid verify_mode: {verify_mode!r}. Must be 'standalone' or 'manual'"}

        with self._lock:
            if self._state == "running" or self._executor.is_running:
                return {"error": "VLA is already running"}

        task_start_time = time.time()
        try:
            verify_system_prompt = self._load_verify_system_prompt()
        except ValueError as e:
            self._record_start_failure(
                instruction=instruction,
                completion_mode=completion_mode,
                start_time=task_start_time,
                reason=str(e),
                baseline_frames={},
            )
            return {"error": str(e)}

        baseline_frames = self._capture_baseline()
        baseline_main = self._get_camera_frame(baseline_frames, "main")
        baseline_wrist = self._get_camera_frame(baseline_frames, "wrist")
        if baseline_main is None:
            reason = "Missing baseline main frame"
            self._record_start_failure(
                instruction=instruction,
                completion_mode=completion_mode,
                start_time=task_start_time,
                reason=reason,
                baseline_frames=baseline_frames,
            )
            return {"error": reason}
        baseline_main_b64 = self._frame_to_base64(baseline_main, max_edge=VERIFY_MAIN_MAX_EDGE)
        if baseline_main_b64 is None:
            reason = "Failed to encode baseline main frame"
            self._record_start_failure(
                instruction=instruction,
                completion_mode=completion_mode,
                start_time=task_start_time,
                reason=reason,
                baseline_frames=baseline_frames,
            )
            return {"error": reason}
        baseline_wrist_b64 = self._frame_to_base64(baseline_wrist, max_edge=VERIFY_WRIST_MAX_EDGE)

        with self._lock:
            self._task_revision += 1
            self._state = "running"
            self._instruction = instruction
            self._completion_mode = completion_mode
            self._verify_mode = verify_mode
            self._task_start_time = task_start_time
            self._task_end_time = None
            self._error = None
            self._baseline_frames = dict(baseline_frames)
            self._baseline_main_b64 = baseline_main_b64
            self._baseline_wrist_b64 = baseline_wrist_b64
            self._verify_system_prompt = verify_system_prompt
            self._reset_verify_state_locked()

        try:
            self._executor.start(instruction)
        except Exception as e:
            self._record_start_failure(
                instruction=instruction,
                completion_mode=completion_mode,
                start_time=task_start_time,
                reason=str(e),
                baseline_frames=baseline_frames,
            )
            return {"error": str(e)}

        if self._verify_debug:
            self._verify_debug_dir = _verify_debug_base_dir()
            self._verify_debug_dir.mkdir(parents=True, exist_ok=True)
            self._verify_debug_stamp = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-pid{os.getpid()}"
            if not self._verify_video_recording_started:
                self._verify_video_recording_started = True
                _start_camera_video_recording(
                    output_dir=self._verify_debug_dir,
                    frame_getter=lambda: capture_live_frames(self._executor),
                    name_suffix=self._verify_debug_stamp,
                )

        logger.info("Task started: {} (mode={}, verify={})", instruction, completion_mode, verify_mode)
        if verify_mode == "standalone":
            self._start_verify_loop()
        return {
            "status": "started",
            "instruction": instruction,
            "completion_mode": completion_mode,
            "verify_mode": verify_mode,
        }

    def get_status(self) -> dict:
        """Get current task status."""
        with self._lock:
            return {
                "state": self._state,
                "instruction": self._instruction,
                "completion_mode": self._completion_mode,
                "duration_s": self._duration_s_locked(),
                "verify_debug": self._verify_debug,
                "verify_step": self._verify_step,
                "last_verify_verdict": self._last_verify_verdict,
                "last_verify_reason": self._last_verify_reason,
                "last_verify_latency_ms": self._last_verify_latency_ms,
                "last_verify_at": self._last_verify_at,
                "last_verify_error": self._last_verify_error,
                "completed_verify_streak": self._completed_verify_streak,
                "completed_verify_required": VERIFY_COMPLETED_STREAK_REQUIRED,
                "error": self._error,
            }

    def stop_task(self) -> None:
        """Stop current task execution."""
        with self._lock:
            should_mark_stopped = self._state == "running" or self._executor.is_running
        self._cancel_verify_loop()
        self._executor.stop(return_home=True)

        with self._lock:
            if not should_mark_stopped:
                return
            if self._task_start_time is None:
                self._task_start_time = time.time()
            self._task_end_time = time.time()
            self._state = "stopped"
            self._error = None

        logger.info("Task stopped")

    async def verify_once(self) -> dict:
        """Run one verification check (manual mode only).

        Returns:
            {
                "verdict": "completed" | "in_progress" | "failed",
                "reason": str,
                "has_motion": bool,
                "latency_ms": int | None,
                "step": int,
                "image_path": str | None
            }
        """
        with self._lock:
            if self._verify_mode != "manual":
                return {"error": "verify_once only available in manual mode"}
            if self._state != "running":
                return {"error": f"Task not running (state={self._state})"}
            step = self._verify_step + 1

        # Run verification
        try:
            verdict, reason, latency_ms = await self._verify_once(step=step)
            self._record_verify_success(step=step, verdict=verdict, reason=reason, latency_ms=latency_ms)

            # TODO: has_motion detection (compare current frame with previous frame)
            # For now, always return True if executor is running
            has_motion = self._executor.is_running

            # TODO: save image and return path
            image_path = None

            if verdict in {"completed", "failed"}:
                await self._finish_task(verdict, reason)

            return {
                "verdict": verdict,
                "reason": reason,
                "has_motion": has_motion,
                "latency_ms": latency_ms,
                "step": step,
                "image_path": image_path,
            }
        except Exception as e:
            self._record_verify_error(step=step, error=str(e))
            return {
                "verdict": "error",
                "reason": str(e),
                "has_motion": False,
                "latency_ms": None,
                "step": step,
                "image_path": None,
            }

    def cleanup(self) -> None:
        """Cleanup resources."""
        self._cancel_verify_loop()
        self._executor.cleanup()
        asyncio.create_task(self._http_client.aclose())

    def _capture_baseline(self) -> dict[str, np.ndarray]:
        """Capture the single baseline snapshot used for verifier comparisons."""
        return capture_live_frames(self._executor)

    @staticmethod
    def _get_camera_frame(frames: dict[str, np.ndarray], camera_name: str) -> np.ndarray | None:
        """Resolve camera frames from either plain names or observation.images.* keys."""
        direct = frames.get(camera_name)
        if direct is not None:
            return direct

        prefixed = frames.get(f"observation.images.{camera_name}")
        if prefixed is not None:
            return prefixed

        for key, value in frames.items():
            if key.lower().endswith(camera_name.lower()):
                return value

        return None

    def _build_status_text(self) -> str:
        """Build status text for observe result."""
        status = self.get_status()
        instruction = status["instruction"] or "(none)"
        text = (
            "[VLA Status]\n"
            f"state: {status['state']}\n"
            f"instruction: {instruction}\n"
            f"completion_mode: {status['completion_mode'] or '(none)'}\n"
            f"duration_s: {status['duration_s']}"
        )
        if status["error"]:
            text += f"\nerror: {status['error']}"
        return text

    def _start_verify_loop(self) -> None:
        """Start verification loop."""
        self._cancel_verify_loop()
        revision = self._task_revision
        self._verify_task = asyncio.create_task(
            self._verify_loop(revision),
            name=f"vla-verify-{revision}",
        )

    def _cancel_verify_loop(self) -> None:
        """Cancel verification loop."""
        task = self._verify_task
        if task is None:
            return
        try:
            current = asyncio.current_task()
        except RuntimeError:
            current = None
        if task is not current and not task.done():
            task.cancel()
        self._verify_task = None

    async def _verify_loop(self, revision: int) -> None:
        """Run the single-task verifier loop in the background."""
        timeout_s = self._verify_config.get("timeout_s", 180)
        step = 0

        try:
            if self._verify_start_delay_s > 0:
                await asyncio.sleep(self._verify_start_delay_s)
            while True:
                with self._lock:
                    if revision != self._task_revision:
                        return
                    if self._state != "running":
                        return
                    duration = self._duration_s_locked(use_current_time=True)

                if not self._executor.is_running:
                    error = self._executor.last_error or "Executor stopped unexpectedly"
                    await self._finish_task("failed", error)
                    return

                if duration > timeout_s:
                    logger.warning("Task timeout after {}s", duration)
                    await self._finish_task("failed", "Task timeout")
                    return

                step += 1
                try:
                    verdict, reason, latency_ms = await self._verify_once(revision=revision, step=step)
                    self._record_verify_success(
                        step=step,
                        verdict=verdict,
                        reason=reason,
                        latency_ms=latency_ms,
                    )
                    if not self._verify_debug:
                        logger.info("Verify: verdict={}, reason={}", verdict, reason)

                    if verdict == "completed":
                        with self._lock:
                            completed_streak = self._completed_verify_streak
                        if completed_streak >= VERIFY_COMPLETED_STREAK_REQUIRED:
                            await self._finish_task(verdict, reason)
                            return
                        logger.info(
                            "Verify completed streak {}/{}; waiting for confirmation",
                            completed_streak,
                            VERIFY_COMPLETED_STREAK_REQUIRED,
                        )
                        continue

                    if verdict == "failed":
                        await self._finish_task(verdict, reason)
                        return

                except Exception as e:
                    self._record_verify_error(step=step, error=str(e))
                    logger.error("Verification error: {}", e)

                if self._verify_interval_s > 0:
                    await asyncio.sleep(self._verify_interval_s)

        except asyncio.CancelledError:
            logger.info("Verification loop cancelled")
            raise

    async def _verify_once(
        self,
        *,
        revision: int | None = None,
        step: int | None = None,
    ) -> tuple[str, str, int | None]:
        """Run one verification check and return the latest verifier result."""
        round_t0 = time.perf_counter()
        revision, step = self._resolve_verify_trace_ids(revision=revision, step=step)
        self._log_verify_phase(revision=revision, step=step, phase="preprocess_start")

        t_capture = time.perf_counter()
        current_frames = capture_live_frames(self._executor)
        with self._lock:
            baseline_frames = dict(self._baseline_frames)
        baseline_main = self._get_camera_frame(baseline_frames, "main")
        baseline_wrist = self._get_camera_frame(baseline_frames, "wrist")
        current_main = self._get_camera_frame(current_frames, "main")
        current_wrist = self._get_camera_frame(current_frames, "wrist")
        capture_ms = int((time.perf_counter() - t_capture) * 1000)

        if baseline_main is None:
            self._log_missing_frames(baseline_frames, current_frames)
            self._log_verify_phase(
                revision=revision,
                step=step,
                phase="preprocess_done",
                skipped_model=True,
                capture_ms=capture_ms,
                total_ms=int((time.perf_counter() - round_t0) * 1000),
                reason="Missing baseline main frame",
            )
            return ("continue", "Missing baseline main frame", None)
        if current_main is None:
            self._log_missing_frames(baseline_frames, current_frames)
            self._log_verify_phase(
                revision=revision,
                step=step,
                phase="preprocess_done",
                skipped_model=True,
                capture_ms=capture_ms,
                total_ms=int((time.perf_counter() - round_t0) * 1000),
                reason="Missing current main frame",
            )
            return ("continue", "Missing current main frame", None)

        rule_verdict = self._rule_based_check()
        if rule_verdict:
            verdict, reason = rule_verdict
            self._log_verify_phase(
                revision=revision,
                step=step,
                phase="preprocess_done",
                skipped_model=True,
                capture_ms=capture_ms,
                total_ms=int((time.perf_counter() - round_t0) * 1000),
                verdict=verdict,
                reason=reason,
            )
            return (verdict, reason, None)

        return await self._vlm_check(
            baseline_main=baseline_main,
            baseline_wrist=baseline_wrist,
            current_main=current_main,
            current_wrist=current_wrist,
            revision=revision,
            step=step,
            round_t0=round_t0,
            capture_ms=capture_ms,
        )

    def _rule_based_check(self) -> tuple[str, str] | None:
        """Fast rule-based checks."""
        if self._completion_mode == "until_clear":
            with self._lock:
                duration = self._duration_s_locked(use_current_time=True)
            if duration > 180:
                return ("failed", "Timeout waiting for all targets to be cleared")
        return None

    async def _vlm_check(
        self,
        *,
        baseline_main: np.ndarray,
        baseline_wrist: np.ndarray,
        current_main: np.ndarray,
        current_wrist: np.ndarray,
        revision: int | None = None,
        step: int | None = None,
        round_t0: float | None = None,
        capture_ms: int | None = None,
    ) -> tuple[str, str, int | None]:
        """Run the verifier model against the latest scene images."""
        revision, step = self._resolve_verify_trace_ids(revision=revision, step=step)
        if round_t0 is None:
            round_t0 = time.perf_counter()

        t_encode = time.perf_counter()
        with self._lock:
            baseline_main_b64 = self._baseline_main_b64
            baseline_wrist_b64 = self._baseline_wrist_b64
        if baseline_main_b64 is None:
            baseline_main_b64 = self._frame_to_base64(baseline_main, max_edge=VERIFY_MAIN_MAX_EDGE)
        if baseline_wrist_b64 is None:
            baseline_wrist_b64 = self._frame_to_base64(baseline_wrist, max_edge=VERIFY_WRIST_MAX_EDGE)
        current_main_b64 = self._frame_to_base64(current_main, max_edge=VERIFY_MAIN_MAX_EDGE)
        current_wrist_b64 = self._frame_to_base64(current_wrist, max_edge=VERIFY_WRIST_MAX_EDGE)
        encode_ms = int((time.perf_counter() - t_encode) * 1000)

        provider = self._get_verify_provider()
        return await self._openai_compatible_vlm_check(
            provider=provider,
            baseline_main_b64=baseline_main_b64,
            baseline_wrist_b64=baseline_wrist_b64,
            current_main_b64=current_main_b64,
            current_wrist_b64=current_wrist_b64,
            revision=revision,
            step=step,
            round_t0=round_t0,
            capture_ms=capture_ms,
            encode_ms=encode_ms,
        )

    async def _openai_compatible_vlm_check(
        self,
        *,
        provider: str,
        baseline_main_b64: str,
        baseline_wrist_b64: str | None,
        current_main_b64: str,
        current_wrist_b64: str | None,
        revision: int,
        step: int,
        round_t0: float,
        capture_ms: int | None = None,
        encode_ms: int | None = None,
    ) -> tuple[str, str, int | None]:
        """Run the verifier through an OpenAI-compatible chat completions API."""
        t_build_msg = time.perf_counter()
        messages = self._build_verify_messages(
            baseline_main_b64=baseline_main_b64,
            baseline_wrist_b64=baseline_wrist_b64,
            current_main_b64=current_main_b64,
            current_wrist_b64=current_wrist_b64,
        )
        provider_config = self._get_verify_provider_config(provider)
        api_base = self._get_verify_api_base(provider)
        model = str(provider_config.get("model", DEFAULT_VERIFY_MODEL))
        request_payload = {
            "model": model,
            "messages": messages,
            "max_tokens": self._verify_max_tokens,
            "temperature": 0,
        }
        if provider == "ark":
            request_payload["thinking"] = {"type": "disabled"}
        build_msg_ms = int((time.perf_counter() - t_build_msg) * 1000)
        user_content = messages[1]["content"]
        image_count = sum(1 for block in user_content if block.get("type") == "image_url")
        proxy_env = self._summarize_proxy_env()

        if self._verify_debug:
            debug_dir = self._save_verify_debug_artifacts(
                baseline_main_b64=baseline_main_b64,
                baseline_wrist_b64=baseline_wrist_b64,
                current_main_b64=current_main_b64,
                current_wrist_b64=current_wrist_b64,
                request_payload=request_payload,
                provider=provider,
                api_base=api_base,
            )
            logger.info("Saved verify debug artifacts to {}", debug_dir)

        self._log_verify_phase(
            revision=revision,
            step=step,
            phase="request_sent",
            provider=provider,
            model=model,
            api_base=api_base,
            max_tokens=self._verify_max_tokens,
            image_count=image_count,
            trust_env_proxy=self._verify_trust_env_proxy,
            proxy_env=proxy_env,
            capture_ms=capture_ms,
            encode_ms=encode_ms,
            build_msg_ms=build_msg_ms,
            preprocess_ms=int((time.perf_counter() - round_t0) * 1000),
        )
        t0 = time.perf_counter()
        try:
            response = await self._http_client.post(
                f"{api_base}/chat/completions",
                headers=self._build_verify_headers(provider),
                json=request_payload,
            )
            request_ms = int((time.perf_counter() - t0) * 1000)
            provider_request_id = self._extract_provider_request_id(response)

            t_parse = time.perf_counter()
            response.raise_for_status()
            result = response.json()
            content = self._extract_response_text(result)
            verdict, reason = self._parse_verdict(content)
            parse_ms = int((time.perf_counter() - t_parse) * 1000)
            self._save_verify_debug_response(
                revision=revision, step=step,
                raw_content=content, verdict=verdict, reason=reason,
                latency_ms=request_ms, provider=provider, model=model,
                provider_request_id=provider_request_id,
            )
            self._log_verify_phase(
                revision=revision,
                step=step,
                phase="response_received",
                provider=provider,
                model=model,
                api_base=api_base,
                max_tokens=self._verify_max_tokens,
                image_count=image_count,
                trust_env_proxy=self._verify_trust_env_proxy,
                proxy_env=proxy_env,
                request_ms=request_ms,
                parse_ms=parse_ms,
                total_ms=int((time.perf_counter() - round_t0) * 1000),
                provider_request_id=provider_request_id,
                verdict=verdict,
                reason=reason,
                raw_content=content,
            )
            return (verdict, reason, request_ms)

        except Exception as e:
            request_ms = int((time.perf_counter() - t0) * 1000)
            self._save_verify_debug_response(
                revision=revision, step=step,
                raw_content="", verdict="error", reason=str(e),
                latency_ms=request_ms, provider=provider, model=model,
                provider_request_id=None,
            )
            self._log_verify_phase(
                revision=revision,
                step=step,
                phase="response_error",
                provider=provider,
                model=model,
                api_base=api_base,
                max_tokens=self._verify_max_tokens,
                image_count=image_count,
                trust_env_proxy=self._verify_trust_env_proxy,
                proxy_env=proxy_env,
                request_ms=request_ms,
                total_ms=int((time.perf_counter() - round_t0) * 1000),
                provider_request_id=None,
                error=str(e),
            )
            logger.error(
                "Verifier API error provider={} api_base={} model={} error={}",
                provider,
                api_base,
                model,
                e,
            )
            return ("continue", f"API error: {str(e)}", request_ms)

    def _get_verify_provider(self) -> str:
        """Return the configured verifier provider."""
        provider = str(self._verify_config.get("provider", DEFAULT_VERIFY_PROVIDER)).strip().lower()
        if provider not in VERIFY_PROVIDERS:
            raise ValueError(f"verify.provider must be one of {sorted(VERIFY_PROVIDERS)}")
        return provider

    def _get_verify_provider_config(self, provider: str) -> dict[str, Any]:
        """Return the selected verifier provider block."""
        providers = self._verify_config.get("providers")
        if not isinstance(providers, dict):
            raise ValueError("verify.providers is required")

        provider_config = providers.get(provider)
        if not isinstance(provider_config, dict):
            raise ValueError(f"verify.providers.{provider} is required")
        return provider_config

    def _get_verify_api_base(self, provider: str) -> str:
        """Return the configured verifier API base URL."""
        provider_config = self._get_verify_provider_config(provider)
        api_base = provider_config.get("api_base")
        if api_base:
            return str(api_base).rstrip("/")
        if provider == "openrouter":
            return DEFAULT_VERIFY_API_BASE
        if provider == "ark":
            return DEFAULT_VERIFY_ARK_API_BASE
        raise ValueError("verify.providers.local.api_base is required")

    def _validate_verify_config(self) -> None:
        """Validate verifier provider configuration at startup."""
        provider = self._get_verify_provider()
        provider_config = self._get_verify_provider_config(provider)
        if not provider_config.get("model"):
            raise ValueError(f"verify.providers.{provider}.model is required")
        if provider == "local" and not provider_config.get("api_base"):
            raise ValueError("verify.providers.local.api_base is required")
        if provider in {"openrouter", "ark"} and not provider_config.get("api_key"):
            raise ValueError(f"verify.providers.{provider}.api_key is required")

    def _load_verify_interval_s(self) -> float:
        """Return the configured verify polling interval."""
        interval_s = float(self._verify_config.get("interval_s", DEFAULT_VERIFY_INTERVAL_S))
        if interval_s <= 0:
            return DEFAULT_VERIFY_INTERVAL_S
        return interval_s

    def _load_verify_start_delay_s(self) -> float:
        """Seconds to wait after task start before the first verify call (standalone loop only)."""
        delay_s = float(self._verify_config.get("start_delay_s", DEFAULT_VERIFY_START_DELAY_S))
        if delay_s < 0:
            raise ValueError("verify.start_delay_s must be >= 0")
        return delay_s

    def _load_verify_max_tokens(self) -> int:
        """Return the configured verify response token limit."""
        max_tokens = int(self._verify_config.get("max_tokens", DEFAULT_VERIFY_MAX_TOKENS))
        if max_tokens <= 0:
            raise ValueError("verify.max_tokens must be > 0")
        return max_tokens

    def _load_verify_trust_env_proxy(self) -> bool:
        """Return whether verifier HTTP requests should honor proxy env vars."""
        return bool(self._verify_config.get("trust_env_proxy", True))

    @staticmethod
    def _summarize_proxy_env() -> str:
        """Summarize active proxy-related environment variables."""
        proxy_keys = [
            key
            for key in ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY")
            if os.getenv(key)
        ]
        return ",".join(proxy_keys) if proxy_keys else "none"

    @staticmethod
    def _extract_provider_request_id(response: Any) -> str | None:
        """Extract a provider-side request id from common HTTP response headers."""
        headers = getattr(response, "headers", None)
        if headers is None or not hasattr(headers, "get"):
            return None
        for key in ("x-request-id", "request-id", "x-b3-traceid", "trace-id"):
            value = headers.get(key)
            if isinstance(value, str) and value:
                return value
        return None

    @staticmethod
    def _build_debug_content(
        content: list[dict[str, Any]],
        *,
        image_names: list[str],
    ) -> list[dict[str, Any]]:
        """Replace inline images with saved file references in debug payloads."""

        debug_content: list[dict[str, Any]] = []
        image_index = 0
        for block in content:
            block_type = block.get("type")
            if block_type in {"text", "input_text"}:
                debug_content.append({"type": block_type, "text": block["text"]})
                continue
            if block_type in {"image_url", "input_image"}:
                debug_content.append({"type": "image_file", "path": image_names[image_index]})
                image_index += 1
        return debug_content

    def _build_debug_request_payload(
        self,
        request_payload: dict[str, Any],
        *,
        image_names: list[str],
    ) -> dict[str, Any]:
        """Build a file-based debug payload from the outbound verifier request."""
        debug_payload = dict(request_payload)
        if "messages" in request_payload:
            debug_payload["messages"] = [
                {"role": message["role"], "content": message["content"]}
                if message["role"] == "system"
                else {
                    "role": message["role"],
                    "content": self._build_debug_content(message["content"], image_names=image_names),
                }
                for message in request_payload["messages"]
            ]
        if "input" in request_payload:
            debug_payload["input"] = [
                {
                    "role": message["role"],
                    "content": self._build_debug_content(message["content"], image_names=image_names),
                }
                for message in request_payload["input"]
            ]
        return debug_payload

    def _build_verify_messages(
        self,
        *,
        baseline_main_b64: str,
        baseline_wrist_b64: str | None,
        current_main_b64: str,
        current_wrist_b64: str | None,
    ) -> list[dict[str, Any]]:
        """Build a labeled multimodal prompt so image order is explicit."""
        verify_system_prompt = self._verify_system_prompt or self._load_verify_system_prompt()
        user_content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    f"Instruction: {self._instruction}\n"
                    f"Completion mode: {self._completion_mode}"
                ),
            },
            {"type": "text", "text": "[baseline_main] Desk-wide view before the task."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{baseline_main_b64}"}},
        ]
        if baseline_wrist_b64 is not None:
            user_content.extend(
                [
                    {"type": "text", "text": "[baseline_wrist] Close-up wrist view before the task."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{baseline_wrist_b64}"}},
                ]
            )
        user_content.extend(
            [
                {"type": "text", "text": "[current_main] Desk-wide view now."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{current_main_b64}"}},
            ]
        )
        if current_wrist_b64 is not None:
            user_content.extend(
                [
                    {"type": "text", "text": "[current_wrist] Close-up wrist view now."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{current_wrist_b64}"}},
                ]
            )

        return [
            {
                "role": "system",
                "content": verify_system_prompt,
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]

    def _build_verify_headers(self, provider: str) -> dict[str, str]:
        """Build request headers for the configured verifier backend."""
        headers: dict[str, str] = {}

        api_key = self._get_verify_provider_config(provider).get("api_key")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        if provider == "openrouter":
            headers["HTTP-Referer"] = "https://github.com/yourusername/vlapilot"
            headers["X-Title"] = "VLAPilot"

        return headers

    @staticmethod
    def _format_verify_log_value(value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, float):
            return str(round(value, 1))
        if isinstance(value, str):
            return json.dumps(value, ensure_ascii=False)
        return str(value)

    @staticmethod
    def _build_verify_log_field(key: str, value: Any) -> str:
        if key in {"request_s", "total_s"}:
            return f"{key}={value}"
        return f"{key}={VLARuntime._format_verify_log_value(value)}"

    @staticmethod
    def _format_verify_seconds(ms: int) -> str:
        return f"{ms / 1000:.3f}"

    @staticmethod
    def _get_verdict_highlight_color(verdict: str | None) -> str:
        if verdict == "completed":
            return "green"
        if verdict == "continue":
            return "yellow"
        if verdict == "failed":
            return "red"
        return "yellow"

    @staticmethod
    def _highlight_placeholder(color: str) -> str:
        return f"<{color}>{{}}</{color}>"

    def _emit_info_log(self, message: str, *args: Any, colors: bool = False) -> None:
        if colors:
            logger.opt(colors=True).info(message, *args)
            return
        logger.info(message, *args)

    def _resolve_verify_trace_ids(
        self,
        *,
        revision: int | None,
        step: int | None,
    ) -> tuple[int, int]:
        with self._lock:
            resolved_revision = revision if revision is not None else self._task_revision
            resolved_step = step if step is not None else self._verify_step + 1
        return (resolved_revision, resolved_step)

    def _log_verify_phase(
        self,
        *,
        revision: int,
        step: int,
        phase: str,
        provider: str | None = None,
        model: str | None = None,
        api_base: str | None = None,
        max_tokens: int | None = None,
        image_count: int | None = None,
        trust_env_proxy: bool | None = None,
        proxy_env: str | None = None,
        provider_request_id: str | None = None,
        capture_ms: int | None = None,
        encode_ms: int | None = None,
        build_msg_ms: int | None = None,
        preprocess_ms: int | None = None,
        request_ms: int | None = None,
        parse_ms: int | None = None,
        total_ms: int | None = None,
        skipped_model: bool | None = None,
        verdict: str | None = None,
        reason: str | None = None,
        raw_content: str | None = None,
        error: str | None = None,
    ) -> None:
        """Emit a structured debug log for one verify phase."""
        if not self._verify_debug:
            return

        raw_fields: list[tuple[str, Any]] = [
            ("rev", revision),
            ("step", step),
            ("phase", phase),
        ]
        with self._lock:
            duration_s = self._duration_s_locked(use_current_time=True)
        raw_fields.append(("task_duration_s", duration_s))
        for key, value in (
            ("provider", provider),
            ("model", model),
            ("api_base", api_base),
            ("max_tokens", max_tokens),
            ("image_count", image_count),
            ("trust_env_proxy", trust_env_proxy),
            ("proxy_env", proxy_env),
            ("provider_request_id", provider_request_id),
            ("capture_ms", capture_ms),
            ("encode_ms", encode_ms),
            ("build_msg_ms", build_msg_ms),
            ("preprocess_ms", preprocess_ms),
            ("request_ms", request_ms),
            ("parse_ms", parse_ms),
            ("total_ms", total_ms),
            ("skipped_model", skipped_model),
            ("verdict", verdict),
            ("reason", reason),
            ("raw", raw_content.replace("\n", "\\n") if raw_content else None),
            ("error", error),
        ):
            if value is not None:
                raw_fields.append((key, value))

        message_parts = ["Verify debug"]
        message_args: list[str] = []
        colors = False
        highlight_color = self._get_verdict_highlight_color(verdict)
        for key, value in raw_fields:
            field_key = key
            field_value: Any = value
            if key in ("capture_ms", "encode_ms", "build_msg_ms", "preprocess_ms", "request_ms", "parse_ms", "total_ms"):
                field_key = f"{key[:-2]}s"
                field_value = self._format_verify_seconds(value)
            rendered_field = self._build_verify_log_field(field_key, field_value)
            if field_key in ("verdict", "reason"):
                message_parts.append(self._highlight_placeholder(highlight_color))
                colors = True
            else:
                message_parts.append("{}")
            message_args.append(rendered_field)

        self._emit_info_log(" ".join(message_parts), *message_args, colors=colors)

    def _save_verify_debug_response(
        self,
        *,
        revision: int,
        step: int,
        raw_content: str,
        verdict: str,
        reason: str,
        latency_ms: int | None,
        provider: str,
        model: str,
        provider_request_id: str | None,
    ) -> None:
        """Save the verifier response alongside request artifacts."""
        if not self._verify_debug:
            return
        assert self._verify_debug_dir is not None
        assert self._verify_debug_stamp is not None
        prefix = f"rev{revision:04d}-step{step:04d}-{self._verify_debug_stamp}"
        response_payload = {
            "provider": provider,
            "model": model,
            "raw_content": raw_content,
            "verdict": verdict,
            "reason": reason,
            "latency_ms": latency_ms,
            "provider_request_id": provider_request_id,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        }
        (self._verify_debug_dir / f"{prefix}-response.json").write_text(
            json.dumps(response_payload, indent=2, ensure_ascii=False)
        )

    def _save_verify_debug_artifacts(
        self,
        *,
        baseline_main_b64: str,
        baseline_wrist_b64: str | None,
        current_main_b64: str,
        current_wrist_b64: str | None,
        request_payload: dict[str, Any],
        provider: str,
        api_base: str,
    ) -> Path:
        """Save the exact JPEGs and metadata that will be sent to the verifier."""
        assert self._verify_debug_dir is not None
        assert self._verify_debug_stamp is not None
        with self._lock:
            revision = self._task_revision
            step = self._verify_step + 1
            instruction = self._instruction
            completion_mode = self._completion_mode

        prefix = f"rev{revision:04d}-step{step:04d}-{self._verify_debug_stamp}"
        debug_dir = self._verify_debug_dir

        image_names: list[str] = []
        for name, b64 in [
            (f"{prefix}-baseline_main.jpg", baseline_main_b64),
            (f"{prefix}-baseline_wrist.jpg", baseline_wrist_b64),
            (f"{prefix}-current_main.jpg", current_main_b64),
            (f"{prefix}-current_wrist.jpg", current_wrist_b64),
        ]:
            if b64 is None:
                continue
            (debug_dir / name).write_bytes(base64.b64decode(b64))
            image_names.append(name)
        debug_request_payload = self._build_debug_request_payload(request_payload, image_names=image_names)
        payload = {
            "instruction": instruction,
            "completion_mode": completion_mode,
            "provider": provider,
            "api_base": api_base,
            "session_dir": str(debug_dir),
            "verify_file_stamp": self._verify_debug_stamp,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "images": {
                key: name
                for key, name in (
                    ("baseline_main", f"{prefix}-baseline_main.jpg"),
                    ("baseline_wrist", f"{prefix}-baseline_wrist.jpg"),
                    ("current_main", f"{prefix}-current_main.jpg"),
                    ("current_wrist", f"{prefix}-current_wrist.jpg"),
                )
                if (debug_dir / name).exists()
            },
        }
        payload.update(debug_request_payload)
        (debug_dir / f"{prefix}-request.json").write_text(json.dumps(payload, indent=2))
        return debug_dir

    def _load_verify_system_prompt(self) -> str:
        """Load the verifier system prompt from the shared task prompt file."""
        prompt_path = self._verify_prompt_path or VERIFY_PROMPT_PATH
        try:
            prompt = prompt_path.read_text(encoding="utf-8").strip()
        except FileNotFoundError as e:
            raise ValueError(f"Missing verifier prompt file: {prompt_path}") from e
        if not prompt:
            raise ValueError(f"Verifier prompt file is empty: {prompt_path}")
        return prompt

    @staticmethod
    def _extract_response_text(result: dict[str, Any]) -> str:
        """Normalize OpenAI-compatible response content into plain text."""
        content = result["choices"][0]["message"]["content"]
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return json.dumps(content)

        text_parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
                continue
            if not isinstance(block, dict):
                continue
            text = block.get("text")
            if isinstance(text, str):
                text_parts.append(text)

        return "\n".join(part for part in text_parts if part)

    def _frame_to_base64(self, frame: np.ndarray | None, *, max_edge: int) -> str | None:
        """Convert numpy frame to base64 JPEG."""
        if frame is None:
            return None

        try:
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
            image = Image.fromarray(frame)
            image.thumbnail((max_edge, max_edge))
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=85)
            jpeg_bytes = buffer.getvalue()
            return base64.b64encode(jpeg_bytes).decode("utf-8")

        except Exception as e:
            logger.error("Frame conversion error: {}", e)
            return None

    def _parse_verdict(self, content: str) -> tuple[str, str]:
        """Parse verifier output: lines with only `completed` or `continue`, plus a few completed heuristics."""
        text = self._strip_code_fences(content).strip()
        if not text:
            return ("continue", "Empty verifier response")

        for line in text.splitlines():
            s = line.strip()
            if not s:
                continue
            m = VERIFY_VERDICT_LINE.match(s)
            if m:
                word = m.group(1).lower()
                if word == "completed":
                    return ("completed", "Task appears completed")
                return ("continue", "Task not yet complete")

        tl = text.lower()
        if '"verdict"' in tl and '"completed"' in tl:
            return ("completed", "Task appears completed")
        if "verdict: completed" in tl or "verdict=completed" in tl:
            return ("completed", "Task appears completed")
        if "task is completed" in tl or "task appears completed" in tl:
            return ("completed", "Task appears completed")
        return ("continue", "Unable to parse verdict")

    @staticmethod
    def _strip_code_fences(content: str) -> str:
        """Remove markdown ``` fences around verifier text."""
        stripped = content.strip()
        if not (stripped.startswith("```") and stripped.endswith("```")):
            return stripped

        lines = stripped.splitlines()
        if len(lines) < 2:
            return stripped
        return "\n".join(lines[1:-1]).strip()

    def _record_verify_success(
        self,
        *,
        step: int,
        verdict: str,
        reason: str,
        latency_ms: int | None,
    ) -> None:
        with self._lock:
            self._verify_step = step
            self._last_verify_verdict = verdict
            self._last_verify_reason = reason
            self._last_verify_latency_ms = latency_ms
            self._last_verify_at = datetime.now().isoformat(timespec="seconds")
            self._last_verify_error = None
            self._completed_verify_streak = self._completed_verify_streak + 1 if verdict == "completed" else 0

    def _record_verify_error(self, *, step: int, error: str) -> None:
        with self._lock:
            self._verify_step = step
            self._last_verify_verdict = None
            self._last_verify_reason = None
            self._last_verify_latency_ms = None
            self._last_verify_at = datetime.now().isoformat(timespec="seconds")
            self._last_verify_error = error
            self._completed_verify_streak = 0

    async def _finish_task(self, verdict: str, reason: str) -> None:
        """Finalize a task after the verifier reaches a terminal verdict."""
        highlight_color = self._get_verdict_highlight_color(verdict)
        self._emit_info_log(
            (
                f"Task finished: {self._highlight_placeholder(highlight_color)}, "
                f"{self._highlight_placeholder(highlight_color)}"
            ),
            f"verdict={verdict}",
            f"reason={reason}",
            colors=True,
        )

        with self._lock:
            finished_at = time.time()
            event = {
                "type": "vla_terminal",
                "instruction": self._instruction,
                "verdict": verdict,
                "reason": reason,
                "completion_mode": self._completion_mode,
                "duration_s": self._duration_s_locked(end_time=finished_at),
            }
            self._state = verdict
            self._task_end_time = finished_at
            self._error = reason if verdict == "failed" else None

        self._cancel_verify_loop()
        self._executor.stop(return_home=True)

        if self._notification_callback:
            try:
                await self._notification_callback(event)
            except Exception as e:
                logger.error("Notification error: {}", e)

    def _log_missing_frames(
        self,
        baseline_frames: dict[str, np.ndarray],
        current_frames: dict[str, np.ndarray],
    ) -> None:
        if not self._verify_debug:
            return
        self._emit_info_log(
            "Verify debug frames baseline_keys={} current_keys={}",
            sorted(baseline_frames.keys()),
            sorted(current_frames.keys()),
        )

    def _reset_verify_state_locked(self) -> None:
        self._verify_step = 0
        self._last_verify_verdict = None
        self._last_verify_reason = None
        self._last_verify_latency_ms = None
        self._last_verify_at = None
        self._last_verify_error = None
        self._completed_verify_streak = 0

    def _record_start_failure(
        self,
        *,
        instruction: str,
        completion_mode: str,
        start_time: float,
        reason: str,
        baseline_frames: dict[str, np.ndarray],
    ) -> None:
        with self._lock:
            self._task_revision += 1
            self._state = "failed"
            self._instruction = instruction
            self._completion_mode = completion_mode
            self._task_start_time = start_time
            self._task_end_time = time.time()
            self._baseline_frames = dict(baseline_frames)
            self._baseline_main_b64 = None
            self._baseline_wrist_b64 = None
            self._verify_system_prompt = None
            self._error = reason
            self._reset_verify_state_locked()

    def _duration_s_locked(
        self,
        *,
        end_time: float | None = None,
        use_current_time: bool = False,
    ) -> float:
        if self._task_start_time is None:
            return 0
        if end_time is not None:
            final_time = end_time
        elif self._task_end_time is not None:
            final_time = self._task_end_time
        elif use_current_time:
            final_time = time.time()
        else:
            final_time = time.time()
        return round(max(0.0, final_time - self._task_start_time), 1)
