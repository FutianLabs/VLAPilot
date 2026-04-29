"""Test VLA Runtime."""

import asyncio
import base64
import io
from pathlib import Path
import sys
import time
import types
from typing import Any
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

executor_stub = types.ModuleType("vla_pi05_piper.executor")
executor_stub.RTCExecutor = object
executor_stub.capture_live_frames = lambda executor: executor.get_observation()
sys.modules.setdefault("vla_pi05_piper.executor", executor_stub)

import vla_pi05_piper.runtime as runtime_module
from vla_pi05_piper.runtime import VLARuntime


def make_verify_config(provider: str = "openrouter") -> dict:
    return {
        "provider": provider,
        "providers": {
            "openrouter": {
                "model": "anthropic/claude-3.5-sonnet",
                "api_key": "test-key",
            },
            "local": {
                "model": "Qwen2.5-VL-7B-Instruct",
                "api_base": "http://127.0.0.1:1821/v1",
                "api_key": "",
            },
            "ark": {
                "model": "doubao-seed-2-0-lite-250821",
                "api_key": "ark-test-key",
            },
        },
        "timeout_s": 10,
        "start_delay_s": 0,
    }


def load_verify_prompt_text() -> str:
    return runtime_module.VERIFY_PROMPT_PATH.read_text(encoding="utf-8").strip()


@pytest.fixture
def frame():
    return np.zeros((4, 4, 3), dtype=np.uint8)


@pytest.fixture
def mock_executor(frame):
    """Create mock executor."""
    executor = Mock()
    executor.is_running = False
    executor.last_error = None
    executor.get_observation.return_value = {
        "observation.images.main": frame,
        "observation.images.wrist": frame,
    }

    def start_side_effect(_instruction):
        executor.is_running = True

    def stop_side_effect(return_home=False):
        executor.is_running = False

    executor.start.side_effect = start_side_effect
    executor.stop.side_effect = stop_side_effect
    return executor


@pytest.fixture
def runtime(mock_executor):
    """Create runtime instance."""
    verify_config = make_verify_config("openrouter")
    return VLARuntime(
        executor=mock_executor,
        verify_config=verify_config,
        notification_callback=None,
    )


@pytest.fixture(autouse=True)
def stub_capture_live_frames(monkeypatch):
    monkeypatch.setattr(
        runtime_module,
        "capture_live_frames",
        lambda executor: executor.get_observation(),
    )


def test_observe(runtime):
    """Test observe method."""
    result = asyncio.run(runtime.observe(["main"]))
    assert result["status"] == "captured"
    assert "status_text" in result


def test_start_task(runtime, mock_executor):
    """Test start_task method."""
    result = asyncio.run(runtime.start_task("Test instruction", "single"))

    assert result["status"] == "started"
    assert result["instruction"] == "Test instruction"
    mock_executor.start.assert_called_once_with("Test instruction")
    assert runtime._verify_system_prompt == load_verify_prompt_text()
    assert runtime._baseline_main_b64 is not None

    status = runtime.get_status()
    assert status["state"] == "running"
    assert status["instruction"] == "Test instruction"
    assert status["completion_mode"] == "single"
    assert status["error"] is None


def test_start_task_requires_baseline_main(runtime, mock_executor, frame):
    """Task start should fail fast when the baseline main camera is missing."""
    mock_executor.get_observation.return_value = {
        "observation.images.wrist": frame,
    }

    result = asyncio.run(runtime.start_task("Test instruction", "single"))

    assert result == {"error": "Missing baseline main frame"}
    mock_executor.start.assert_not_called()

    status = runtime.get_status()
    assert status["state"] == "failed"
    assert status["instruction"] == "Test instruction"
    assert status["error"] == "Missing baseline main frame"


def test_start_task_fails_when_verify_prompt_file_is_missing(runtime, monkeypatch, tmp_path, mock_executor):
    missing_path = tmp_path / "does-not-exist-verify.md"
    monkeypatch.setattr(runtime_module, "VERIFY_PROMPT_PATH", missing_path)

    result = asyncio.run(runtime.start_task("Test instruction", "single"))

    assert result == {"error": f"Missing verifier prompt file: {missing_path}"}
    mock_executor.start.assert_not_called()
    status = runtime.get_status()
    assert status["state"] == "failed"
    assert status["error"] == f"Missing verifier prompt file: {missing_path}"


def test_start_task_fails_when_verify_prompt_file_is_empty(runtime, monkeypatch, tmp_path, mock_executor):
    prompt_path = tmp_path / "verify.md"
    prompt_path.write_text("   \n", encoding="utf-8")
    monkeypatch.setattr(runtime_module, "VERIFY_PROMPT_PATH", prompt_path)

    result = asyncio.run(runtime.start_task("Test instruction", "single"))

    assert result == {"error": f"Verifier prompt file is empty: {prompt_path}"}
    mock_executor.start.assert_not_called()
    status = runtime.get_status()
    assert status["state"] == "failed"
    assert status["error"] == f"Verifier prompt file is empty: {prompt_path}"


def test_start_task_reloads_verify_prompt_for_next_task(runtime, monkeypatch):
    prompt_versions = iter(["prompt version one", "prompt version two"])
    monkeypatch.setattr(runtime, "_load_verify_system_prompt", lambda: next(prompt_versions))

    first = asyncio.run(runtime.start_task("Test instruction", "single"))
    assert first["status"] == "started"
    assert runtime._verify_system_prompt == "prompt version one"
    runtime.stop_task()

    second = asyncio.run(runtime.start_task("Test instruction", "single"))
    assert second["status"] == "started"
    assert runtime._verify_system_prompt == "prompt version two"


def test_get_status(runtime):
    """Test get_status method."""
    status = runtime.get_status()
    assert status["state"] == "idle"
    assert status["instruction"] is None
    assert status["verify_debug"] is False
    assert status["verify_step"] == 0
    assert status["last_verify_verdict"] is None
    assert status["last_verify_latency_ms"] is None
    assert status["error"] is None


def test_stop_task_preserves_task_snapshot(runtime, mock_executor):
    """Stopping a task should keep terminal state and task metadata visible."""
    asyncio.run(runtime.start_task("Test instruction", "single"))

    runtime.stop_task()

    mock_executor.stop.assert_called()
    status = runtime.get_status()
    assert status["state"] == "stopped"
    assert status["instruction"] == "Test instruction"
    assert status["completion_mode"] == "single"
    assert status["error"] is None


def test_get_camera_frame_accepts_plain_and_prefixed_keys(runtime):
    """Runtime should resolve both raw camera names and observation.images.* keys."""
    plain = {"main": "plain-main", "wrist": "plain-wrist"}
    prefixed = {"observation.images.main": "prefixed-main"}

    assert runtime._get_camera_frame(plain, "main") == "plain-main"
    assert runtime._get_camera_frame(plain, "wrist") == "plain-wrist"
    assert runtime._get_camera_frame(prefixed, "main") == "prefixed-main"


def test_finish_task_preserves_terminal_snapshot(mock_executor):
    """Terminal fields should remain readable after the task ends."""
    callback = AsyncMock()
    runtime = VLARuntime(
        executor=mock_executor,
        verify_config=make_verify_config("openrouter"),
        notification_callback=callback,
    )
    runtime._state = "running"
    runtime._instruction = "Test instruction"
    runtime._completion_mode = "single"
    runtime._task_start_time = time.time() - 1.5

    asyncio.run(runtime._finish_task("completed", "done"))

    status = runtime.get_status()
    assert status["state"] == "completed"
    assert status["instruction"] == "Test instruction"
    assert status["completion_mode"] == "single"
    assert status["error"] is None
    mock_executor.stop.assert_called_once_with(return_home=True)
    callback.assert_awaited_once()
    event = callback.await_args.args[0]
    assert event["instruction"] == "Test instruction"
    assert event["completion_mode"] == "single"
    assert event["verdict"] == "completed"
    assert event["reason"] == "done"
    assert event["duration_s"] >= 1.0


def test_finish_task_highlights_terminal_verdict(monkeypatch, mock_executor):
    runtime = VLARuntime(
        executor=mock_executor,
        verify_config=make_verify_config("openrouter"),
        notification_callback=None,
    )
    runtime._state = "running"
    runtime._instruction = "Test instruction"
    runtime._completion_mode = "single"
    runtime._task_start_time = time.time() - 0.5

    logs: list[str] = []

    def fake_emit(message, *args, colors=False):
        logs.append(str(message).format(*args))

    monkeypatch.setattr(runtime, "_emit_info_log", fake_emit)

    asyncio.run(runtime._finish_task("failed", "blocked"))

    assert any(
        "Task finished:" in line
        and "<red>verdict=failed</red>" in line
        and "<red>reason=blocked</red>" in line
        for line in logs
    )


def test_verify_loop_records_latest_result_and_executor_failure(
    monkeypatch,
    mock_executor,
):
    """Debug mode should still record verifier status and fail when executor dies."""
    runtime = VLARuntime(
        executor=mock_executor,
        verify_config={**make_verify_config("openrouter"), "debug": True},
        notification_callback=None,
    )
    runtime._task_revision = 1
    runtime._state = "running"
    runtime._instruction = "Test instruction"
    runtime._completion_mode = "single"
    runtime._task_start_time = time.time()
    mock_executor.is_running = True

    async def fake_verify_once(*, revision=None, step=None):
        mock_executor.is_running = False
        mock_executor.last_error = "GetActions fatal: boom"
        return ("continue", "still running", 123)

    monkeypatch.setattr(runtime, "_verify_once", fake_verify_once)

    asyncio.run(runtime._verify_loop(1))

    status = runtime.get_status()
    assert status["verify_debug"] is True
    assert status["verify_step"] == 1
    assert status["last_verify_verdict"] == "continue"
    assert status["last_verify_reason"] == "still running"
    assert status["last_verify_latency_ms"] == 123
    assert status["last_verify_at"] is not None
    assert status["completed_verify_streak"] == 0
    assert status["state"] == "failed"
    assert status["error"] == "GetActions fatal: boom"


def test_verify_loop_finishes_on_first_completed_result(monkeypatch, mock_executor):
    runtime = VLARuntime(
        executor=mock_executor,
        verify_config={**make_verify_config("openrouter")},
        notification_callback=None,
    )
    runtime._task_revision = 1
    runtime._state = "running"
    runtime._instruction = "Test instruction"
    runtime._completion_mode = "single"
    runtime._task_start_time = time.time()
    mock_executor.is_running = True

    responses = iter(
        [
            ("completed", "first confirmation", 111),
            ("completed", "second confirmation", 222),
        ]
    )
    finish_calls: list[tuple[str, str]] = []

    async def fake_verify_once(*, revision=None, step=None):
        return next(responses)

    async def fake_finish(verdict: str, reason: str):
        finish_calls.append((verdict, reason))
        runtime._state = verdict

    monkeypatch.setattr(runtime, "_verify_once", fake_verify_once)
    monkeypatch.setattr(runtime, "_finish_task", fake_finish)

    asyncio.run(runtime._verify_loop(1))

    status = runtime.get_status()
    assert finish_calls == [("completed", "first confirmation")]
    assert status["verify_step"] == 1
    assert status["last_verify_verdict"] == "completed"
    assert status["last_verify_reason"] == "first confirmation"
    assert status["completed_verify_streak"] == 1
    assert status["state"] == "completed"


def test_verify_loop_waits_for_configured_interval(monkeypatch, mock_executor):
    runtime = VLARuntime(
        executor=mock_executor,
        verify_config={**make_verify_config("openrouter"), "interval_s": 0.5},
        notification_callback=None,
    )
    runtime._task_revision = 1
    runtime._state = "running"
    runtime._instruction = "Test instruction"
    runtime._completion_mode = "single"
    runtime._task_start_time = time.time()
    mock_executor.is_running = True

    responses = iter(
        [
            ("continue", "still running", 111),
            ("completed", "done", 222),
        ]
    )
    finish_calls: list[tuple[str, str]] = []
    sleep_calls: list[float] = []

    async def fake_verify_once(*, revision=None, step=None):
        return next(responses)

    async def fake_finish(verdict: str, reason: str):
        finish_calls.append((verdict, reason))
        runtime._state = verdict

    async def fake_sleep(interval: float):
        sleep_calls.append(interval)

    monkeypatch.setattr(runtime, "_verify_once", fake_verify_once)
    monkeypatch.setattr(runtime, "_finish_task", fake_finish)
    monkeypatch.setattr(runtime_module.asyncio, "sleep", fake_sleep)

    asyncio.run(runtime._verify_loop(1))

    assert finish_calls == [("completed", "done")]
    assert sleep_calls == [0.5]


def test_zero_verify_interval_falls_back_to_default(mock_executor):
    runtime = VLARuntime(
        executor=mock_executor,
        verify_config={**make_verify_config("openrouter"), "interval_s": 0},
        notification_callback=None,
    )

    assert runtime._verify_interval_s == runtime_module.DEFAULT_VERIFY_INTERVAL_S


def test_verify_can_disable_env_proxy(mock_executor):
    runtime = VLARuntime(
        executor=mock_executor,
        verify_config={**make_verify_config("openrouter"), "trust_env_proxy": False},
        notification_callback=None,
    )

    assert runtime._verify_trust_env_proxy is False


def test_completed_verify_streak_resets_on_continue_and_error(runtime):
    runtime._record_verify_success(step=1, verdict="completed", reason="first", latency_ms=100)
    assert runtime.get_status()["completed_verify_streak"] == 1

    runtime._record_verify_success(step=2, verdict="continue", reason="not done", latency_ms=120)
    assert runtime.get_status()["completed_verify_streak"] == 0

    runtime._record_verify_success(step=3, verdict="completed", reason="first again", latency_ms=130)
    assert runtime.get_status()["completed_verify_streak"] == 1

    runtime._record_verify_error(step=4, error="network")
    assert runtime.get_status()["completed_verify_streak"] == 0


def test_manual_verify_once_completed_finishes_task(monkeypatch, mock_executor):
    runtime = VLARuntime(
        executor=mock_executor,
        verify_config=make_verify_config("openrouter"),
        notification_callback=None,
    )
    runtime._verify_mode = "manual"
    runtime._state = "running"
    runtime._instruction = "Test instruction"
    runtime._completion_mode = "single"
    runtime._task_start_time = time.time() - 1
    mock_executor.is_running = True

    async def fake_verify_once(*, revision=None, step=None):
        return ("completed", "task done", 123)

    monkeypatch.setattr(runtime, "_verify_once", fake_verify_once)

    result = asyncio.run(runtime.verify_once())
    status = runtime.get_status()

    assert result["verdict"] == "completed"
    assert result["reason"] == "task done"
    assert status["state"] == "completed"
    assert status["last_verify_reason"] == "task done"
    mock_executor.stop.assert_called_once_with(return_home=True)


def test_manual_verify_once_failed_finishes_task(monkeypatch, mock_executor):
    runtime = VLARuntime(
        executor=mock_executor,
        verify_config=make_verify_config("openrouter"),
        notification_callback=None,
    )
    runtime._verify_mode = "manual"
    runtime._state = "running"
    runtime._instruction = "Test instruction"
    runtime._completion_mode = "single"
    runtime._task_start_time = time.time() - 1
    mock_executor.is_running = True

    async def fake_verify_once(*, revision=None, step=None):
        return ("failed", "blocked", 88)

    monkeypatch.setattr(runtime, "_verify_once", fake_verify_once)

    result = asyncio.run(runtime.verify_once())
    status = runtime.get_status()

    assert result["verdict"] == "failed"
    assert result["reason"] == "blocked"
    assert status["state"] == "failed"
    assert status["error"] == "blocked"
    mock_executor.stop.assert_called_once_with(return_home=True)


def test_verify_once_logs_missing_frame_keys(monkeypatch, frame, mock_executor):
    """Debug mode should show which frame keys are available when required frames are missing."""
    runtime = VLARuntime(
        executor=mock_executor,
        verify_config={**make_verify_config("openrouter"), "debug": True},
        notification_callback=None,
    )
    runtime._baseline_frames = {"observation.images.main": frame}
    mock_executor.get_observation.return_value = {"observation.images.wrist": frame}

    logs: list[str] = []

    def fake_emit(message, *args, colors=False):
        logs.append(str(message).format(*args))

    monkeypatch.setattr(runtime, "_emit_info_log", fake_emit)

    verdict, reason, latency_ms = asyncio.run(runtime._verify_once())

    assert verdict == "continue"
    assert reason == "Missing current main frame"
    assert latency_ms is None
    assert any('phase="preprocess_start"' in line for line in logs)
    assert any("baseline_keys=['observation.images.main']" in line for line in logs)
    assert any("current_keys=['observation.images.wrist']" in line for line in logs)
    assert any('phase="preprocess_done"' in line for line in logs)
    assert any("skipped_model=true" in line for line in logs)
    assert any('<yellow>reason="Missing current main frame"</yellow>' in line for line in logs)


def test_verify_once_debug_logs_request_timing(monkeypatch, tmp_path: Path, frame, mock_executor):
    runtime = VLARuntime(
        executor=mock_executor,
        verify_config={**make_verify_config("local"), "debug": True},
        notification_callback=None,
    )
    runtime._verify_debug_dir = tmp_path
    runtime._verify_debug_stamp = "20260329-120000-pid12345"
    runtime._instruction = "Test instruction"
    runtime._completion_mode = "single"
    runtime._baseline_frames = {"observation.images.main": frame, "observation.images.wrist": frame}
    mock_executor.get_observation.return_value = {
        "observation.images.main": frame,
        "observation.images.wrist": frame,
    }

    logs: list[str] = []

    def fake_emit(message, *args, colors=False):
        logs.append(str(message).format(*args))

    response = Mock()
    response.headers = {"x-request-id": "req-local-123"}
    response.raise_for_status = Mock()
    response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": "continue",
                }
            }
        ]
    }

    runtime._http_client.post = AsyncMock(return_value=response)
    monkeypatch.setattr(runtime, "_emit_info_log", fake_emit)

    verdict, reason, latency_ms = asyncio.run(runtime._verify_once(revision=2, step=7))

    assert verdict == "continue"
    assert reason == "Task not yet complete"
    assert latency_ms is not None
    assert any('rev=2 step=7 phase="preprocess_start"' in line for line in logs)
    assert any(
        'phase="request_sent"' in line
        and "provider=\"local\"" in line
        and "api_base=\"http://127.0.0.1:1821/v1\"" in line
        and "max_tokens=256" in line
        and "image_count=4" in line
        and "trust_env_proxy=true" in line
        and "proxy_env=" in line
        for line in logs
    )
    request_sent_logs = [line for line in logs if 'phase="request_sent"' in line]
    assert len(request_sent_logs) > 0, "Should have request_sent phase log"
    assert any(
        'phase="response_received"' in line
        and "request_s=0." in line
        and "total_s=0." in line
        and "provider_request_id=\"req-local-123\"" in line
        and "request_ms=" not in line
        and "total_ms=" not in line
        for line in logs
    )
    assert any('<yellow>verdict="continue"</yellow>' in line for line in logs)
    assert any('<yellow>reason="Task not yet complete"</yellow>' in line for line in logs)


def test_verify_once_debug_logs_response_error(monkeypatch, tmp_path: Path, frame, mock_executor):
    runtime = VLARuntime(
        executor=mock_executor,
        verify_config={**make_verify_config("local"), "debug": True},
        notification_callback=None,
    )
    runtime._verify_debug_dir = tmp_path
    runtime._verify_debug_stamp = "20260329-120000-pid12345"
    runtime._instruction = "Test instruction"
    runtime._completion_mode = "single"
    runtime._baseline_frames = {"observation.images.main": frame, "observation.images.wrist": frame}
    mock_executor.get_observation.return_value = {
        "observation.images.main": frame,
        "observation.images.wrist": frame,
    }

    logs: list[str] = []

    def fake_emit(message, *args, colors=False):
        logs.append(str(message).format(*args))

    runtime._http_client.post = AsyncMock(side_effect=RuntimeError("boom"))
    monkeypatch.setattr(runtime, "_emit_info_log", fake_emit)

    verdict, reason, latency_ms = asyncio.run(runtime._verify_once(revision=3, step=9))

    assert verdict == "continue"
    assert reason == "API error: boom"
    assert latency_ms is not None
    assert any('phase="request_sent"' in line for line in logs)
    assert any('phase="response_error"' in line and 'error="boom"' in line for line in logs)


def test_vlm_check_uses_custom_api_base_without_auth(frame, mock_executor):
    """A local OpenAI-compatible endpoint should not require auth headers by default."""
    runtime = VLARuntime(
        executor=mock_executor,
        verify_config={**make_verify_config("local"), "http_timeout_s": 45},
        notification_callback=None,
    )
    runtime._instruction = "Test instruction"
    runtime._completion_mode = "single"

    response = Mock()
    response.raise_for_status = Mock()
    response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": [
                        {
                            "type": "text",
                            "text": "```\ncompleted\n```",
                        }
                    ]
                }
            }
        ]
    }
    runtime._http_client.post = AsyncMock(return_value=response)

    verdict, reason, latency_ms = asyncio.run(
        runtime._vlm_check(
            baseline_main=frame,
            baseline_wrist=frame,
            current_main=frame,
            current_wrist=frame,
        )
    )

    assert verdict == "completed"
    assert reason == "Task appears completed"
    assert latency_ms is not None
    assert runtime._http_client.timeout.connect == 45.0

    call = runtime._http_client.post.await_args
    assert call.args[0] == "http://127.0.0.1:1821/v1/chat/completions"
    assert call.kwargs["headers"] == {}
    assert call.kwargs["json"]["model"] == "Qwen2.5-VL-7B-Instruct"
    assert call.kwargs["json"]["max_tokens"] == 256
    content = call.kwargs["json"]["messages"][1]["content"]
    assert [item["type"] for item in content] == [
        "text", "text", "image_url", "text", "image_url",
        "text", "image_url", "text", "image_url",
    ]
    assert content[1]["text"] == "[baseline_main] Desk-wide view before the task."
    assert content[3]["text"] == "[baseline_wrist] Close-up wrist view before the task."
    assert content[5]["text"] == "[current_main] Desk-wide view now."
    assert content[7]["text"] == "[current_wrist] Close-up wrist view now."
    assert "Instruction: Test instruction" in content[0]["text"]
    assert "Completion mode: single" in content[0]["text"]


def test_vlm_check_keeps_openrouter_defaults(frame, runtime):
    """OpenRouter provider should use the default OpenRouter base."""
    runtime._instruction = "Test instruction"
    runtime._completion_mode = "single"

    response = Mock()
    response.raise_for_status = Mock()
    response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": "continue",
                }
            }
        ]
    }
    runtime._http_client.post = AsyncMock(return_value=response)

    verdict, reason, _ = asyncio.run(
        runtime._vlm_check(
            baseline_main=frame,
            baseline_wrist=frame,
            current_main=frame,
            current_wrist=frame,
        )
    )

    assert verdict == "continue"
    assert reason == "Task not yet complete"

    call = runtime._http_client.post.await_args
    assert call.args[0] == "https://openrouter.ai/api/v1/chat/completions"
    assert call.kwargs["headers"]["Authorization"] == "Bearer test-key"
    assert call.kwargs["headers"]["HTTP-Referer"] == "https://github.com/yourusername/vlapilot"
    assert call.kwargs["headers"]["X-Title"] == "VLAPilot"


def test_vlm_check_reuses_cached_baseline_encodings(monkeypatch, frame, mock_executor):
    runtime = VLARuntime(
        executor=mock_executor,
        verify_config=make_verify_config("local"),
        notification_callback=None,
    )
    runtime._instruction = "Test instruction"
    runtime._completion_mode = "single"
    runtime._baseline_main_b64 = "cached-main"
    runtime._baseline_wrist_b64 = "cached-wrist"

    calls: list[int] = []
    original_frame_to_base64 = runtime._frame_to_base64

    def fake_frame_to_base64(frame_value, *, max_edge):
        calls.append(max_edge)
        return original_frame_to_base64(frame_value, max_edge=max_edge)

    response = Mock()
    response.raise_for_status = Mock()
    response.json.return_value = {
        "choices": [{"message": {"content": "continue"}}]
    }
    runtime._http_client.post = AsyncMock(return_value=response)
    monkeypatch.setattr(runtime, "_frame_to_base64", fake_frame_to_base64)

    asyncio.run(
        runtime._vlm_check(
            baseline_main=frame,
            baseline_wrist=frame,
            current_main=frame,
            current_wrist=frame,
        )
    )

    assert calls == [runtime_module.VERIFY_MAIN_MAX_EDGE, runtime_module.VERIFY_WRIST_MAX_EDGE]
    content = runtime._http_client.post.await_args.kwargs["json"]["messages"][1]["content"]
    assert content[2]["image_url"]["url"] == "data:image/jpeg;base64,cached-main"
    assert content[4]["image_url"]["url"] == "data:image/jpeg;base64,cached-wrist"


def test_vlm_check_ark_uses_openai_compatible_endpoint(frame, mock_executor):
    """Ark provider should use httpx + OpenAI-compatible /chat/completions endpoint."""
    runtime = VLARuntime(
        executor=mock_executor,
        verify_config=make_verify_config("ark"),
        notification_callback=None,
    )
    runtime._instruction = "Put the paper ball into the yellow trash bin."
    runtime._completion_mode = "single"

    response = Mock()
    response.raise_for_status = Mock()
    response.json.return_value = {
        "choices": [{"message": {"content": "completed"}}]
    }
    runtime._http_client.post = AsyncMock(return_value=response)

    verdict, reason, latency_ms = asyncio.run(
        runtime._vlm_check(
            baseline_main=frame,
            baseline_wrist=frame,
            current_main=frame,
            current_wrist=frame,
        )
    )

    assert verdict == "completed"
    assert reason == "Task appears completed"
    assert latency_ms is not None

    call = runtime._http_client.post.await_args
    payload = call.kwargs["json"]
    messages = payload["messages"]
    assert messages[0]["role"] == "system"
    user_content = messages[1]["content"]
    assert "Instruction: Put the paper ball into the yellow trash bin." in user_content[0]["text"]
    assert "Completion mode: single" in user_content[0]["text"]
    image_types = [block["type"] for block in user_content if block["type"] == "image_url"]
    assert len(image_types) == 4


def test_build_verify_messages_does_not_require_baseline_change_for_completion(runtime, frame):
    runtime._instruction = "Close the laptop lid."
    runtime._completion_mode = "single"

    b64 = runtime._frame_to_base64(frame, max_edge=runtime_module.VERIFY_MAIN_MAX_EDGE)
    messages = runtime._build_verify_messages(
        baseline_main_b64=b64,
        baseline_wrist_b64=runtime._frame_to_base64(frame, max_edge=runtime_module.VERIFY_WRIST_MAX_EDGE),
        current_main_b64=b64,
        current_wrist_b64=runtime._frame_to_base64(frame, max_edge=runtime_module.VERIFY_WRIST_MAX_EDGE),
    )

    # System message carries the rules (from verify.md); user message carries only dynamic fields.
    system_text = messages[0]["content"]
    assert "completed" in system_text
    assert "continue" in system_text
    assert "baseline" in system_text

    user_text = messages[1]["content"][0]["text"]
    assert "Instruction: Close the laptop lid." in user_text
    assert "Completion mode: single" in user_text


def test_build_verify_messages_omits_missing_wrist_images(runtime, frame):
    runtime._instruction = "Close the laptop lid."
    runtime._completion_mode = "single"

    b64 = runtime._frame_to_base64(frame, max_edge=runtime_module.VERIFY_MAIN_MAX_EDGE)
    messages = runtime._build_verify_messages(
        baseline_main_b64=b64,
        baseline_wrist_b64=None,
        current_main_b64=b64,
        current_wrist_b64=None,
    )

    content = messages[1]["content"]
    assert [item["type"] for item in content] == ["text", "text", "image_url", "text", "image_url"]
    assert all("wrist" not in item.get("text", "") for item in content if item["type"] == "text")


def test_build_verify_headers_skips_openrouter_headers_for_local_provider(mock_executor):
    runtime = VLARuntime(
        executor=mock_executor,
        verify_config={
            "provider": "local",
            "providers": {
                "openrouter": {"model": "anthropic/claude-3.5-sonnet", "api_key": "test-key"},
                "local": {
                    "model": "google/gemini-2.5-flash-lite",
                    "api_base": "http://127.0.0.1:8000/v1",
                    "api_key": "local-key",
                },
            },
        },
        notification_callback=None,
    )

    headers = runtime._build_verify_headers("local")

    assert headers == {"Authorization": "Bearer local-key"}


def test_save_verify_debug_artifacts(tmp_path: Path, frame, mock_executor):
    runtime = VLARuntime(
        executor=mock_executor,
        verify_config={**make_verify_config("local"), "debug": True},
        notification_callback=None,
    )
    runtime._verify_debug_dir = tmp_path
    runtime._verify_debug_stamp = "20260329-233205-pid12345"
    runtime._instruction = "Put the paper ball into the yellow trash bin."
    runtime._completion_mode = "until_clear"
    runtime._task_revision = 2
    runtime._verify_step = 36
    runtime._verify_system_prompt = "debug verify prompt"

    main_b64 = runtime._frame_to_base64(frame, max_edge=runtime_module.VERIFY_MAIN_MAX_EDGE)
    wrist_b64 = runtime._frame_to_base64(frame, max_edge=runtime_module.VERIFY_WRIST_MAX_EDGE)
    debug_dir = runtime._save_verify_debug_artifacts(
        baseline_main_b64=main_b64,
        baseline_wrist_b64=wrist_b64,
        current_main_b64=main_b64,
        current_wrist_b64=wrist_b64,
        request_payload={
            "model": "Qwen2.5-VL-7B-Instruct",
            "messages": runtime._build_verify_messages(
                baseline_main_b64=main_b64,
                baseline_wrist_b64=wrist_b64,
                current_main_b64=main_b64,
                current_wrist_b64=wrist_b64,
            ),
            "max_tokens": 256,
            "temperature": 0,
        },
        provider="local",
        api_base="http://127.0.0.1:1821/v1",
    )

    assert debug_dir == runtime._verify_debug_dir
    fp = "rev0002-step0037-20260329-233205-pid12345"
    assert (debug_dir / f"{fp}-baseline_main.jpg").exists()
    assert (debug_dir / f"{fp}-baseline_wrist.jpg").exists()
    assert (debug_dir / f"{fp}-current_main.jpg").exists()
    assert (debug_dir / f"{fp}-current_wrist.jpg").exists()

    request = runtime_module.json.loads((debug_dir / f"{fp}-request.json").read_text())
    assert request["instruction"] == "Put the paper ball into the yellow trash bin."
    assert request["completion_mode"] == "until_clear"
    assert request["provider"] == "local"
    assert request["model"] == "Qwen2.5-VL-7B-Instruct"
    assert request["session_dir"] == str(runtime._verify_debug_dir)
    assert request["verify_file_stamp"] == "20260329-233205-pid12345"
    assert request["messages"][0]["content"] == "debug verify prompt"
    assert request["messages"][1]["content"][0]["text"].startswith("Instruction:")
    assert request["messages"][1]["content"][2] == {"type": "image_file", "path": f"{fp}-baseline_main.jpg"}
    assert request["messages"][1]["content"][4] == {"type": "image_file", "path": f"{fp}-baseline_wrist.jpg"}
    assert request["messages"][1]["content"][6] == {"type": "image_file", "path": f"{fp}-current_main.jpg"}
    assert request["messages"][1]["content"][8] == {"type": "image_file", "path": f"{fp}-current_wrist.jpg"}


def test_save_verify_debug_response_includes_provider_request_id(tmp_path: Path, mock_executor):
    runtime = VLARuntime(
        executor=mock_executor,
        verify_config={**make_verify_config("local"), "debug": True},
        notification_callback=None,
    )
    runtime._verify_debug_dir = tmp_path
    runtime._verify_debug_stamp = "20260329-233205-pid12345"

    runtime._save_verify_debug_response(
        revision=2,
        step=7,
        raw_content="continue",
        verdict="continue",
        reason="Task not yet complete",
        latency_ms=321,
        provider="local",
        model="Qwen2.5-VL-7B-Instruct",
        provider_request_id="req-debug-456",
    )

    response_payload = runtime_module.json.loads(
        (runtime._verify_debug_dir / "rev0002-step0007-20260329-233205-pid12345-response.json").read_text()
    )
    assert response_payload["provider_request_id"] == "req-debug-456"


def test_save_verify_debug_artifacts_for_ark(tmp_path: Path, frame, mock_executor):
    """Ark provider now uses OpenAI-compatible messages format for debug artifacts."""
    runtime = VLARuntime(
        executor=mock_executor,
        verify_config={**make_verify_config("ark"), "debug": True},
        notification_callback=None,
    )
    runtime._verify_debug_dir = tmp_path
    runtime._verify_debug_stamp = "20260330-120000-pid12345"
    runtime._instruction = "Put the paper ball into the yellow trash bin."
    runtime._completion_mode = "single"
    runtime._task_revision = 4
    runtime._verify_step = 8
    runtime._verify_system_prompt = "debug verify prompt"

    main_b64 = runtime._frame_to_base64(frame, max_edge=runtime_module.VERIFY_MAIN_MAX_EDGE)
    wrist_b64 = runtime._frame_to_base64(frame, max_edge=runtime_module.VERIFY_WRIST_MAX_EDGE)

    debug_dir = runtime._save_verify_debug_artifacts(
        baseline_main_b64=main_b64,
        baseline_wrist_b64=wrist_b64,
        current_main_b64=main_b64,
        current_wrist_b64=wrist_b64,
        request_payload={
            "model": "doubao-seed-2-0-lite-250821",
            "messages": runtime._build_verify_messages(
                baseline_main_b64=main_b64,
                baseline_wrist_b64=wrist_b64,
                current_main_b64=main_b64,
                current_wrist_b64=wrist_b64,
            ),
            "max_tokens": 256,
            "temperature": 0,
        },
        provider="ark",
        api_base=runtime_module.DEFAULT_VERIFY_ARK_API_BASE,
    )

    assert debug_dir == runtime._verify_debug_dir

    fp = "rev0004-step0009-20260330-120000-pid12345"
    request = runtime_module.json.loads((debug_dir / f"{fp}-request.json").read_text())
    assert request["provider"] == "ark"
    assert request["model"] == "doubao-seed-2-0-lite-250821"
    assert request["messages"][0]["content"] == "debug verify prompt"
    assert request["messages"][1]["content"][0]["text"].startswith("Instruction:")
    assert request["messages"][1]["content"][2] == {"type": "image_file", "path": f"{fp}-baseline_main.jpg"}
    assert request["messages"][1]["content"][4] == {"type": "image_file", "path": f"{fp}-baseline_wrist.jpg"}
    assert request["messages"][1]["content"][6] == {"type": "image_file", "path": f"{fp}-current_main.jpg"}
    assert request["messages"][1]["content"][8] == {"type": "image_file", "path": f"{fp}-current_wrist.jpg"}


def test_parse_verdict_finds_keyword_line_among_noise(runtime):
    verdict, reason = runtime._parse_verdict(
        "Verifier result:\ncompleted\nThanks."
    )

    assert verdict == "completed"
    assert reason == "Task appears completed"


def test_frame_to_base64_resizes_main_and_wrist_differently(mock_executor):
    runtime = VLARuntime(
        executor=mock_executor,
        verify_config=make_verify_config("local"),
        notification_callback=None,
    )
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    main_b64 = runtime._frame_to_base64(frame, max_edge=runtime_module.VERIFY_MAIN_MAX_EDGE)
    wrist_b64 = runtime._frame_to_base64(frame, max_edge=runtime_module.VERIFY_WRIST_MAX_EDGE)

    main_image = Image.open(io.BytesIO(base64.b64decode(main_b64)))
    wrist_image = Image.open(io.BytesIO(base64.b64decode(wrist_b64)))

    assert max(main_image.size) == runtime_module.VERIFY_MAIN_MAX_EDGE
    assert max(wrist_image.size) == runtime_module.VERIFY_WRIST_MAX_EDGE


def test_local_provider_requires_api_base(mock_executor):
    config = make_verify_config("local")
    config["providers"]["local"].pop("api_base")

    with pytest.raises(ValueError, match="verify.providers.local.api_base is required"):
        VLARuntime(
            executor=mock_executor,
            verify_config=config,
            notification_callback=None,
        )


def test_openrouter_provider_requires_api_key(mock_executor):
    config = make_verify_config("openrouter")
    config["providers"]["openrouter"].pop("api_key")

    with pytest.raises(ValueError, match="verify.providers.openrouter.api_key is required"):
        VLARuntime(
            executor=mock_executor,
            verify_config=config,
            notification_callback=None,
        )


def test_ark_provider_requires_api_key(mock_executor):
    config = make_verify_config("ark")
    config["providers"]["ark"].pop("api_key")

    with pytest.raises(ValueError, match="verify.providers.ark.api_key is required"):
        VLARuntime(
            executor=mock_executor,
            verify_config=config,
            notification_callback=None,
        )


def test_invalid_verify_provider_fails_fast(mock_executor):
    with pytest.raises(ValueError, match="verify.provider must be one of"):
        VLARuntime(
            executor=mock_executor,
            verify_config={
                "provider": "unknown",
                "model": "anthropic/claude-3.5-sonnet",
            },
            notification_callback=None,
        )


def test_selected_provider_block_is_required(mock_executor):
    config = make_verify_config("local")
    config["providers"].pop("local")

    with pytest.raises(ValueError, match="verify.providers.local is required"):
        VLARuntime(
            executor=mock_executor,
            verify_config=config,
            notification_callback=None,
        )


def test_parse_verdict_accepts_markdown_code_fences(runtime):
    verdict, reason = runtime._parse_verdict("```\ncontinue\n```")

    assert verdict == "continue"
    assert reason == "Task not yet complete"


def test_parse_verdict_plain_continue_word(runtime):
    verdict, reason = runtime._parse_verdict("continue")

    assert verdict == "continue"
    assert reason == "Task not yet complete"


def test_parse_verdict_keeps_explicit_completed_signal(runtime):
    verdict, reason = runtime._parse_verdict("Task is completed.")

    assert verdict == "completed"
    assert reason == "Task appears completed"


def test_parse_verdict_does_not_fail_on_free_text_failed(runtime):
    verdict, reason = runtime._parse_verdict(
        "The paper ball has not been placed into the yellow trash bin; there are now two paper balls remaining on the desk."
    )

    assert verdict == "continue"
    assert reason == "Unable to parse verdict"
