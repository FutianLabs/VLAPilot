"""VLA Agent MCP Server - thin planner/execution adapter for NanoBot."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from loguru import logger
from mcp.server import Server
from mcp.types import TextContent, Tool

from vlapilot.backend.mcp_client import ArmClient
from vlapilot.config import load_config, resolve_config_path
from vlapilot.core.execution import PlanExecutionStopped, PlanStepResult, execute_plan_steps
from vlapilot.infra.llm import LLMClient
from vlapilot.infra.logging_config import configure_logging
from vlapilot.core.mission import (
    intersect_task_vocab,
    load_mission_assets,
    load_task_instructions,
    load_task_prompt,
)
from vlapilot.core.session import SessionManager
from vlapilot.core.plan import create_plan

DEFAULT_IMAGE_DIR = Path("/tmp/vlapilot-user")
TERMINAL_PLAN_STATUSES = {"completed", "failed", "stopped"}


@dataclass
class PlanState:
    plan_id: str
    instruction: str
    steps: list[dict[str, Any]]
    status: str = "pending"
    current_step: int = 0
    current_step_started_at: float = 0.0
    results: list[PlanStepResult] = field(default_factory=list)
    error: str = ""


app = Server("vlapilot")

_arm: ArmClient | None = None
_planner_llm: LLMClient | None = None
_session: SessionManager | None = None
_planner_prompt: str | None = None
_eligible_tasks: list[str] = []
_image_dir: Path | None = None
_active_plan: PlanState | None = None
_exec_task: asyncio.Task | None = None


@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="vla_scene",
            description="Capture the current scene and return observation text plus saved image paths.",
            inputSchema={
                "type": "object",
                "properties": {
                    "cameras": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["main", "wrist"]},
                        "description": "Cameras to capture (default: ['main', 'wrist'])",
                    }
                },
            },
        ),
        Tool(
            name="vla_plan",
            description="Create a user-approval plan from the current scene and a high-level instruction.",
            inputSchema={
                "type": "object",
                "properties": {
                    "instruction": {
                        "type": "string",
                        "description": "High-level user instruction, for example 'Clean up the desk'",
                    }
                },
                "required": ["instruction"],
            },
        ),
        Tool(
            name="vla_execute",
            description="Start executing the approved plan in the background.",
            inputSchema={
                "type": "object",
                "properties": {
                    "plan_id": {
                        "type": "string",
                        "description": "Plan ID returned by vla_plan",
                    }
                },
                "required": ["plan_id"],
            },
        ),
        Tool(
            name="vla_progress",
            description="Return the current execution progress for the active plan.",
            inputSchema={
                "type": "object",
                "properties": {
                    "plan_id": {
                        "type": "string",
                        "description": "Plan ID returned by vla_plan",
                    }
                },
                "required": ["plan_id"],
            },
        ),
        Tool(
            name="vla_stop",
            description="Stop the current plan and stop the robotic arm.",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict[str, Any] | None) -> list[TextContent]:
    args = arguments or {}
    try:
        if name == "vla_scene":
            return await _handle_scene(args)
        if name == "vla_plan":
            return await _handle_plan(args)
        if name == "vla_execute":
            return await _handle_execute(args)
        if name == "vla_progress":
            return await _handle_progress(args)
        if name == "vla_stop":
            return await _handle_stop()
        return _json_response({"error": f"Unknown tool: {name}"})
    except Exception as exc:
        logger.exception("Tool {} failed: {}", name, exc)
        return _json_response({"error": str(exc)})


async def _handle_scene(arguments: dict[str, Any]) -> list[TextContent]:
    arm = _require_arm()
    cameras = arguments.get("cameras") or ["main", "wrist"]
    result = await arm.observe(cameras)
    if result.get("status") == "error":
        return _json_response(result)

    image_paths = _save_images_to_dir(result.get("images", []), prefix="scene")
    try:
        status = await arm.get_status()
    except Exception:
        status = {}

    payload = {
        "status": result.get("status", "unknown"),
        "observation_summary": result.get("status_text", ""),
        "cameras": result.get("cameras", cameras),
        "state": status.get("state", "unknown"),
        "current_task": status.get("instruction"),
        "image_paths": image_paths,
    }
    return _json_response(payload)


async def _handle_plan(arguments: dict[str, Any]) -> list[TextContent]:
    global _active_plan

    arm = _require_arm()
    llm = _require_llm()
    planner_prompt = _require_planner_prompt()
    instruction = str(arguments.get("instruction", "")).strip()
    if not instruction:
        return _json_response({"error": "instruction is required"})

    _clear_terminal_plan()
    if _active_plan is not None:
        return _json_response(
            {
                "error": "Another plan is still active",
                "plan_id": _active_plan.plan_id,
                "status": _active_plan.status,
            }
        )

    observation = await arm.observe(["main", "wrist"])
    if observation.get("status") == "error":
        return _json_response(
            {
                "error": "Failed to observe scene",
                "detail": observation.get("error", "unknown error"),
            }
        )

    images = observation.get("images", [])
    steps = await create_plan(
        llm,
        instruction=instruction,
        observation_summary=observation.get("status_text", ""),
        images_b64=[image["data"] for image in images if image.get("data")],
        system_prompt=planner_prompt,
        valid_instructions=set(_eligible_tasks),
    )

    _start_session(instruction)
    image_paths = _save_session_images(images, prefix="plan")
    _record_observe_event(observation, image_paths)

    plan_id = f"p_{uuid4().hex[:8]}"
    _active_plan = PlanState(plan_id=plan_id, instruction=instruction, steps=steps)
    _append_event(
        {
            "type": "plan_created",
            "plan_id": plan_id,
            "instruction": instruction,
            "steps": steps,
        }
    )

    payload = {
        "plan_id": plan_id,
        "instruction": instruction,
        "observation_summary": observation.get("status_text", ""),
        "steps": steps,
        "image_paths": image_paths,
    }
    return _json_response(payload)


async def _handle_execute(arguments: dict[str, Any]) -> list[TextContent]:
    global _exec_task

    plan = _require_plan(arguments.get("plan_id", ""))
    if plan.status != "pending":
        return _json_response({"error": f"Plan is not pending: {plan.status}", "plan_id": plan.plan_id})
    if _exec_task is not None and not _exec_task.done():
        return _json_response({"error": "A plan is already executing", "plan_id": plan.plan_id})

    plan.status = "running"
    plan.current_step = 0
    plan.current_step_started_at = time.time()
    plan.results = []
    plan.error = ""

    _exec_task = asyncio.create_task(_run_plan(plan.plan_id))
    return _json_response(
        {
            "status": "started",
            "plan_id": plan.plan_id,
            "total_steps": len(plan.steps),
        }
    )


async def _handle_progress(arguments: dict[str, Any]) -> list[TextContent]:
    plan = _require_plan(arguments.get("plan_id", ""))
    completed_steps = [
        {
            "step": index + 1,
            "instruction": result.instruction,
            "completion_mode": result.completion_mode,
            "status": result.status,
            "verify_reason": result.verify_reason,
            "duration_s": round(result.duration_s, 1),
            "image_paths": result.image_paths,
        }
        for index, result in enumerate(plan.results)
    ]

    current_instruction = ""
    current_completion_mode = ""
    current_elapsed_s = 0.0
    if plan.status == "running" and plan.current_step < len(plan.steps):
        step = plan.steps[plan.current_step]
        current_instruction = step.get("instruction", "")
        current_completion_mode = step.get("completion_mode", "")
        if plan.current_step_started_at:
            current_elapsed_s = round(time.time() - plan.current_step_started_at, 1)

    payload = {
        "plan_id": plan.plan_id,
        "status": plan.status,
        "current_step": min(plan.current_step + 1, len(plan.steps)) if plan.steps else 0,
        "total_steps": len(plan.steps),
        "current_step_instruction": current_instruction,
        "current_step_completion_mode": current_completion_mode,
        "current_step_elapsed_s": current_elapsed_s,
        "completed_steps": completed_steps,
        "error": plan.error or None,
    }
    return _json_response(payload)


async def _handle_stop() -> list[TextContent]:
    global _exec_task

    arm = _require_arm()
    plan = _active_plan

    if _exec_task is not None and not _exec_task.done():
        _exec_task.cancel()
        try:
            await _exec_task
        except (asyncio.CancelledError, Exception):
            pass
        _exec_task = None

    try:
        await arm.stop_task()
    except Exception as exc:
        logger.warning("stop_task failed: {}", exc)

    if plan is not None and plan.status not in TERMINAL_PLAN_STATUSES:
        plan.status = "stopped"
        plan.error = ""
        _append_event(
            {
                "type": "plan_stopped",
                "plan_id": plan.plan_id,
                "completed_steps": len(plan.results),
                "total_steps": len(plan.steps),
            }
        )
    _finish_session(status="stopped")

    return _json_response(
        {
            "status": "stopped",
            "plan_id": plan.plan_id if plan else None,
            "completed_steps": len(plan.results) if plan else 0,
            "total_steps": len(plan.steps) if plan else 0,
        }
    )


async def _run_plan(plan_id: str) -> None:
    plan = _require_plan(plan_id)
    arm = _require_arm()

    try:
        results = await execute_plan_steps(
            arm,
            plan.steps,
            is_running=lambda: plan.status == "running",
            on_step_start=lambda index, step: _on_step_start(plan, index, step),
            on_step_complete=lambda index, result: _on_step_complete(plan, index, result),
            save_step_images=lambda index, images: _save_session_images(
                images,
                prefix=f"exec-step-{index + 1}",
            ),
        )
        failed_index = next(
            (index for index, result in enumerate(results) if result.status != "completed"),
            None,
        )
        if failed_index is not None:
            failed = results[failed_index]
            plan.status = "failed"
            plan.error = f"Step {failed_index + 1} failed: {failed.verify_reason}"
            return

        plan.status = "completed"
        _append_event({"type": "plan_completed", "plan_id": plan.plan_id, "steps": len(plan.steps)})
    except PlanExecutionStopped:
        return
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.exception("Plan execution failed: {}", exc)
        plan.status = "failed"
        plan.error = str(exc)
    finally:
        if plan.status in TERMINAL_PLAN_STATUSES:
            _finish_session(status=plan.status, error=plan.error or None)


def _on_step_start(plan: PlanState, step_index: int, step: dict[str, Any]) -> None:
    plan.current_step = step_index
    plan.current_step_started_at = time.time()
    _append_event(
        {
            "type": "step_start",
            "plan_id": plan.plan_id,
            "step": step_index + 1,
            "instruction": str(step.get("instruction", "")),
            "completion_mode": str(step.get("completion_mode") or "single"),
        }
    )


def _on_step_complete(plan: PlanState, step_index: int, result: PlanStepResult) -> None:
    plan.results.append(result)
    _append_event({"type": "step_complete", "plan_id": plan.plan_id, **result.to_payload(step_index)})


def _clear_terminal_plan() -> None:
    global _active_plan
    if _active_plan is not None and _active_plan.status in TERMINAL_PLAN_STATUSES:
        _active_plan = None


def _start_session(instruction: str) -> None:
    if _session is None:
        return
    if _session.current_session is not None:
        _session.close()
    _session.start_mission(instruction)


def _finish_session(*, status: str, error: str | None = None) -> None:
    if _session is None or _session.current_session is None:
        return
    updates: dict[str, Any] = {"status": status}
    if error:
        updates["error"] = error
    _session.set_metadata(**updates)
    _session.close()


def _record_observe_event(result: dict[str, Any], image_paths: list[str]) -> None:
    if _session is None or _session.current_session is None:
        return
    payload = {
        "type": "observe",
        "status": result.get("status", "unknown"),
        "summary": result.get("status_text", result.get("error", "")),
        "error": result.get("error"),
        "cameras": result.get("cameras", []),
        "image_paths": image_paths,
    }
    _session.set_latest_observe(payload)
    _session.append_event(payload)


def _append_event(event: dict[str, Any]) -> None:
    if _session is None or _session.current_session is None:
        return
    _session.append_event(event)


def _save_session_images(images: list[dict[str, Any]], *, prefix: str) -> list[str]:
    if _session is None or _session.current_session is None:
        return _save_images_to_dir(images, prefix=prefix)
    return _session.save_observe_images(images, prefix=prefix)


def _save_images_to_dir(images: list[dict[str, Any]], *, prefix: str) -> list[str]:
    image_dir = _image_dir or DEFAULT_IMAGE_DIR
    image_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time())
    paths: list[str] = []

    for index, image in enumerate(images):
        data = image.get("data")
        if not isinstance(data, str) or not data:
            continue
        mime_type = str(image.get("mimeType") or "image/jpeg")
        suffix = {
            "image/jpeg": ".jpg",
            "image/png": ".png",
            "image/webp": ".webp",
        }.get(mime_type, ".bin")
        path = image_dir / f"{prefix}-{timestamp}-{index:02d}{suffix}"
        path.write_bytes(base64.b64decode(data))
        paths.append(str(path))
    return paths


def _require_arm() -> ArmClient:
    if _arm is None:
        raise RuntimeError("Arm client is not initialized")
    return _arm


def _require_llm() -> LLMClient:
    if _planner_llm is None:
        raise RuntimeError("Planner LLM is not initialized")
    return _planner_llm


def _require_planner_prompt() -> str:
    if not _planner_prompt:
        raise RuntimeError("Planner prompt is not initialized")
    return _planner_prompt


def _require_plan(plan_id: str) -> PlanState:
    if _active_plan is None:
        raise RuntimeError("No active plan")
    if plan_id != _active_plan.plan_id:
        raise RuntimeError(f"Plan not found: {plan_id}")
    return _active_plan


def _json_response(payload: dict[str, Any]) -> list[TextContent]:
    return [TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2))]


def main() -> None:
    global _arm, _planner_llm, _session, _planner_prompt, _eligible_tasks, _image_dir

    parser = argparse.ArgumentParser(description="VLA Agent MCP Server")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config file (default: ~/.vlapilot/config.json, fallback: ~/.vla/config.json)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    try:
        config_path = resolve_config_path(args.config)
        config = load_config(config_path)
    except Exception as exc:
        logger.error("Failed to load config: {}", exc)
        sys.exit(1)

    if args.debug:
        config.agent.debug = True

    configure_logging(
        level=config.agent.log_level,
        log_file=config.agent.log_file,
        debug=config.agent.debug,
    )

    mission = load_mission_assets(config.agent.mission_dir)
    planner_prompt_path = Path(config.agent.mission_dir) / "mission_mcp.md"
    if not planner_prompt_path.exists():
        logger.error("Missing MCP planner prompt file: {}", planner_prompt_path)
        sys.exit(1)
    if not planner_prompt_path.read_text(encoding="utf-8").strip():
        logger.error("MCP planner prompt file is empty: {}", planner_prompt_path)
        sys.exit(1)

    arm_tasks = load_task_instructions(config.arm.capability_tasks_file)
    _eligible_tasks = intersect_task_vocab(
        [task["instruction"] for task in mission.mission_tasks],
        arm_tasks,
    )
    if not _eligible_tasks:
        logger.error("Mission task vocabulary and arm capability vocabulary have no overlap")
        sys.exit(1)

    _planner_prompt = load_task_prompt(planner_prompt_path, Path(config.agent.mission_dir) / "tasks.yaml")
    planner = config.agent.planner
    _planner_llm = LLMClient(
        provider=planner.provider,
        model=planner.model,
        api_base=planner.api_base,
        api_key=planner.api_key,
        proxy=planner.proxy,
    )
    _arm = ArmClient(config.arm)
    _session = SessionManager(config.agent.session_dir)
    _image_dir = DEFAULT_IMAGE_DIR

    logger.info("VLA Agent MCP Server starting")

    import anyio
    from mcp.server.stdio import stdio_server

    async def run_server() -> None:
        try:
            await _arm.init()
            async with stdio_server() as (read_stream, write_stream):
                await app.run(read_stream, write_stream, app.create_initialization_options())
        finally:
            if _planner_llm is not None:
                await _planner_llm.close()
            if _arm is not None:
                await _arm.cleanup()

    anyio.run(run_server)


if __name__ == "__main__":
    main()
