"""Tests for the VLA Agent MCP server."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vlapilot.interface import mcp_server
from vlapilot.interface.mcp_server import (
    PlanState,
    _handle_execute,
    _handle_plan,
    _handle_progress,
    _handle_scene,
    _handle_stop,
)
from vlapilot.core.session import SessionManager
from vlapilot.core.plan import create_plan
from vlapilot.core import execution as execution_module


MOCK_OBSERVE_RESULT = {
    "status": "captured",
    "cameras": ["main", "wrist"],
    "images": [
        {"data": "AAAA", "mimeType": "image/jpeg"},
        {"data": "BBBB", "mimeType": "image/jpeg"},
    ],
    "status_text": "The desk has a bottle and a paper ball.",
}


@pytest.fixture(autouse=True)
def _reset_globals(tmp_path: Path):
    mcp_server._arm = None
    mcp_server._planner_llm = None
    mcp_server._session = SessionManager(str(tmp_path / "sessions"))
    mcp_server._planner_prompt = "planner prompt"
    mcp_server._eligible_tasks = [
        "Put the bottle into the white box.",
        "Put the paper ball into the yellow trash bin.",
    ]
    mcp_server._image_dir = tmp_path / "images"
    mcp_server._active_plan = None
    mcp_server._exec_task = None
    yield
    if mcp_server._exec_task is not None and not mcp_server._exec_task.done():
        mcp_server._exec_task.cancel()
        try:
            asyncio.run(mcp_server._exec_task)
        except Exception:
            pass


def _setup_mocks():
    arm = AsyncMock()
    arm.observe = AsyncMock(return_value=MOCK_OBSERVE_RESULT)
    arm.get_status = AsyncMock(return_value={"state": "idle", "instruction": None})
    arm.start_task = AsyncMock(return_value={"status": "started"})
    arm.verify_once = AsyncMock(return_value={"verdict": "completed", "reason": "done"})
    arm.stop_task = AsyncMock(return_value={"status": "stopped"})
    arm.get_verify_interval_s = Mock(return_value=0.5)
    arm.get_verify_timeout_s = Mock(return_value=180)

    llm = AsyncMock()
    llm.chat = AsyncMock(
        return_value={
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "steps": [
                                    {
                                        "step": 1,
                                        "instruction": "Put the bottle into the white box.",
                                        "completion_mode": "single",
                                        "rationale": "Handle the large object first",
                                    }
                                ]
                            }
                        )
                    }
                }
            ]
        }
    )

    mcp_server._arm = arm
    mcp_server._planner_llm = llm
    return arm, llm


@pytest.mark.asyncio
async def test_scene_returns_text_payload_and_saved_images():
    _setup_mocks()

    result = await _handle_scene({"cameras": ["main"]})

    assert len(result) == 1
    payload = json.loads(result[0].text)
    assert payload["status"] == "captured"
    assert payload["state"] == "idle"
    assert payload["image_paths"]


@pytest.mark.asyncio
async def test_plan_uses_multimodal_prompt_and_starts_session():
    _, llm = _setup_mocks()

    result = await _handle_plan({"instruction": "Clean up the desk"})

    payload = json.loads(result[0].text)
    assert payload["steps"][0]["completion_mode"] == "single"
    assert payload["image_paths"]
    assert mcp_server._active_plan is not None
    assert mcp_server._session is not None
    assert mcp_server._session.current_session is not None

    call_messages = llm.chat.await_args.args[0]
    assert call_messages[0]["content"] == "planner prompt"
    assert call_messages[1]["role"] == "user"
    assert any(block.get("type") == "image_url" for block in call_messages[1]["content"])


@pytest.mark.asyncio
async def test_plan_rejects_observe_error():
    arm, _ = _setup_mocks()
    arm.observe = AsyncMock(return_value={"status": "error", "error": "camera offline"})

    result = await _handle_plan({"instruction": "Clean up the desk"})

    payload = json.loads(result[0].text)
    assert payload["error"] == "Failed to observe scene"


@pytest.mark.asyncio
async def test_execute_and_progress_complete_plan():
    _setup_mocks()
    await _handle_plan({"instruction": "Clean up the desk"})
    plan_id = mcp_server._active_plan.plan_id

    original_sleep = execution_module.asyncio.sleep
    sleep_mock = AsyncMock()
    execution_module.asyncio.sleep = sleep_mock
    try:
        result = await _handle_execute({"plan_id": plan_id})
        payload = json.loads(result[0].text)
        assert payload["status"] == "started"
        await mcp_server._exec_task
        sleep_mock.assert_awaited_once_with(0.5)
    finally:
        execution_module.asyncio.sleep = original_sleep

    progress = await _handle_progress({"plan_id": plan_id})
    progress_payload = json.loads(progress[0].text)
    assert progress_payload["status"] == "completed"
    assert progress_payload["completed_steps"][0]["status"] == "completed"
    assert mcp_server._session.current_session is None


@pytest.mark.asyncio
async def test_execute_uses_configured_verify_timeout():
    arm, _ = _setup_mocks()
    arm.verify_once = AsyncMock(return_value={"verdict": "in_progress", "reason": "still working"})
    arm.get_verify_interval_s = Mock(return_value=0.5)
    arm.get_verify_timeout_s = Mock(return_value=1.0)
    await _handle_plan({"instruction": "Clean up the desk"})
    plan_id = mcp_server._active_plan.plan_id

    original_sleep = execution_module.asyncio.sleep
    sleep_mock = AsyncMock()
    execution_module.asyncio.sleep = sleep_mock
    try:
        await _handle_execute({"plan_id": plan_id})
        await mcp_server._exec_task
    finally:
        execution_module.asyncio.sleep = original_sleep

    assert sleep_mock.await_count == 2
    arm.get_verify_timeout_s.assert_called()
    arm.stop_task.assert_awaited_once()
    progress = await _handle_progress({"plan_id": plan_id})
    payload = json.loads(progress[0].text)
    assert payload["status"] == "failed"
    assert payload["completed_steps"][0]["verify_reason"] == "timeout"


@pytest.mark.asyncio
async def test_execute_fails_on_verify_error_verdict():
    arm, _ = _setup_mocks()
    arm.verify_once = AsyncMock(return_value={"verdict": "error", "reason": "VLM offline"})
    await _handle_plan({"instruction": "Clean up the desk"})
    plan_id = mcp_server._active_plan.plan_id

    original_sleep = execution_module.asyncio.sleep
    execution_module.asyncio.sleep = AsyncMock()
    try:
        await _handle_execute({"plan_id": plan_id})
        await mcp_server._exec_task
    finally:
        execution_module.asyncio.sleep = original_sleep

    progress = await _handle_progress({"plan_id": plan_id})
    payload = json.loads(progress[0].text)
    assert payload["status"] == "failed"
    assert payload["completed_steps"][0]["verify_reason"] == "VLM offline"


@pytest.mark.asyncio
async def test_stop_marks_plan_stopped_and_closes_session():
    _setup_mocks()
    mcp_server._session.start_mission("Clean up the desk")
    mcp_server._active_plan = PlanState(
        plan_id="p_test",
        instruction="Clean up the desk",
        steps=[],
        status="running",
    )

    result = await _handle_stop()

    payload = json.loads(result[0].text)
    assert payload["status"] == "stopped"
    assert mcp_server._active_plan.status == "stopped"
    assert mcp_server._session.current_session is None


@pytest.mark.asyncio
async def test_create_plan_rejects_missing_completion_mode():
    llm = AsyncMock()
    llm.chat = AsyncMock(
        return_value={
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "steps": [
                                    {
                                        "step": 1,
                                        "instruction": "Put the bottle into the white box.",
                                        "rationale": "missing mode",
                                    }
                                ]
                            }
                        )
                    }
                }
            ]
        }
    )

    with pytest.raises(ValueError, match="completion_mode"):
        await create_plan(
            llm,
            instruction="Clean up the desk",
            observation_summary="The desk has a bottle.",
            images_b64=[],
            system_prompt="prompt",
            valid_instructions={"Put the bottle into the white box."},
        )
