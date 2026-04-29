"""Tests for RTCExecutor startup behavior."""

from __future__ import annotations

import importlib
import signal
from pathlib import Path
import sys
import types

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _load_executor_module():
    pytest.importorskip("lerobot_robot_piper.config_piper")
    sys.modules.pop("vla_pi05_piper.executor", None)
    return importlib.import_module("vla_pi05_piper.executor")


class _FakeLink:
    def __init__(self, target: str):
        self._target = Path(target)
        self.name = self._target.name

    def resolve(self) -> Path:
        return self._target


def test_kill_busy_realsense_holders_kills_target_pids(monkeypatch):
    executor_module = _load_executor_module()
    killed: list[tuple[int, int]] = []

    class _Proc:
        stdout = "111\n222\n"

    class _CameraInfo:
        serial_number = "serial_number"
        physical_port = "physical_port"

    class _Device:
        def get_info(self, key):
            if key == _CameraInfo.serial_number:
                return "918512071262"
            if key == _CameraInfo.physical_port:
                return "/sys/devices/pci0000:00/usb6/6-1/6-1:1.0/video4linux/video6"
            raise KeyError(key)

    class _Context:
        def query_devices(self):
            return [_Device()]

    rs = types.SimpleNamespace(camera_info=_CameraInfo, context=lambda: _Context())

    monkeypatch.setattr(
        executor_module.Path,
        "glob",
        lambda self, pattern: [_FakeLink("/sys/devices/pci0000:00/usb6/6-1/6-1:0.0/video4linux/video2"), _FakeLink("/sys/devices/pci0000:00/usb6/6-1/6-1:1.0/video4linux/video3")],
    )
    monkeypatch.setitem(sys.modules, "pyrealsense2", rs)
    monkeypatch.setattr(executor_module.subprocess, "run", lambda *args, **kwargs: _Proc())
    monkeypatch.setattr(executor_module.os, "getpid", lambda: 222)
    monkeypatch.setattr(executor_module.os, "kill", lambda pid, sig: killed.append((pid, sig)))
    monkeypatch.setattr(executor_module.time, "sleep", lambda _: None)

    released = executor_module._kill_busy_realsense_holders(
        "Failed to open RealSenseCamera(918512071262). Device or resource busy"
    )

    assert released is True
    assert killed == [(111, signal.SIGKILL)]


def test_executor_init_retries_once_after_releasing_busy_camera(monkeypatch):
    executor_module = _load_executor_module()
    executor = executor_module.RTCExecutor(policy_path="/tmp/policy")
    calls: list[str] = []

    def fake_init_robot():
        calls.append("init_robot")
        if calls.count("init_robot") == 1:
            raise RuntimeError("Failed to open RealSenseCamera(918512071262). Device or resource busy")

    monkeypatch.setattr(executor, "_init_robot", fake_init_robot)
    monkeypatch.setattr(executor_module, "_kill_busy_realsense_holders", lambda text: True)
    monkeypatch.setattr(executor, "_disconnect_without_vendor_home", lambda: calls.append("disconnect"))
    monkeypatch.setattr(executor, "_load_standby_pose", lambda: calls.append("load_pose"))
    monkeypatch.setattr(executor, "_init_policy", lambda: calls.append("init_policy"))
    monkeypatch.setattr(executor, "go_to_standby", lambda: calls.append("standby") or True)

    executor.init()

    assert calls == [
        "init_robot",
        "disconnect",
        "init_robot",
        "load_pose",
        "init_policy",
        "standby",
    ]


def test_executor_init_fails_without_busy_camera_release(monkeypatch):
    executor_module = _load_executor_module()
    executor = executor_module.RTCExecutor(policy_path="/tmp/policy")

    monkeypatch.setattr(executor, "_init_robot", lambda: (_ for _ in ()).throw(RuntimeError("camera open failed")))
    monkeypatch.setattr(executor_module, "_kill_busy_realsense_holders", lambda text: False)

    with pytest.raises(RuntimeError, match="robot connection failed"):
        executor.init()


def test_go_to_standby_uses_robot_action_space(monkeypatch):
    executor_module = _load_executor_module()
    executor = executor_module.RTCExecutor(policy_path="/tmp/policy")
    executor._standby_pose = {
        "joint_1.pos": 1.0,
        "joint_2.pos": 2.0,
        "joint_3.pos": 3.0,
        "joint_4.pos": 4.0,
        "joint_5.pos": 5.0,
        "joint_6.pos": 6.0,
        "gripper.pos": 0.3,
    }

    sent: list[dict[str, float]] = []

    class _Robot:
        def __init__(self):
            self._last_cmd_oriented_deg = [99.0] * 6
            self._last_cmd_t = 1.0

        def send_action(self, action):
            sent.append(dict(action))

        def get_observation(self):
            return dict(executor._standby_pose)

    monkeypatch.setattr(executor_module.time, "sleep", lambda _: None)
    executor._robot = _Robot()

    executor.go_to_standby()

    assert sent == [executor._standby_pose]
    assert executor._robot._last_cmd_oriented_deg is None
    assert executor._robot._last_cmd_t is None


def test_go_to_home_uses_same_joint_pose_helper(monkeypatch):
    executor_module = _load_executor_module()
    executor = executor_module.RTCExecutor(policy_path="/tmp/policy")
    calls = []

    monkeypatch.setattr(
        executor,
        "_move_joint_pose",
        lambda target, *, label, timeout_s: calls.append((target, label, timeout_s)) or True,
    )

    assert executor.go_to_home() is True
    assert calls == [(
        {
            "joint_1.pos": 0.0,
            "joint_2.pos": 0.0,
            "joint_3.pos": 0.0,
            "joint_4.pos": 0.0,
            "joint_5.pos": 0.0,
            "joint_6.pos": 0.0,
        },
        "home",
        20.0,
    )]


def test_executor_init_fails_when_standby_is_unreachable(monkeypatch):
    executor_module = _load_executor_module()
    executor = executor_module.RTCExecutor(policy_path="/tmp/policy")

    monkeypatch.setattr(executor, "_init_robot", lambda: None)
    monkeypatch.setattr(executor, "_load_standby_pose", lambda: None)
    monkeypatch.setattr(executor, "_init_policy", lambda: None)
    monkeypatch.setattr(executor, "go_to_standby", lambda: False)

    with pytest.raises(RuntimeError, match="failed to reach standby position"):
        executor.init()
