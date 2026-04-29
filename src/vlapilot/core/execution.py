"""Shared plan execution loop for CLI and MCP entry points."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from loguru import logger

DEFAULT_PLAN_TIMEOUT_S = 180.0


@dataclass
class PlanStepResult:
    """Terminal result for one executable plan step."""

    instruction: str
    completion_mode: str
    status: str = ""
    verify_reason: str = ""
    duration_s: float = 0.0
    image_paths: list[str] = field(default_factory=list)

    def to_payload(self, step_index: int) -> dict[str, Any]:
        """Return the JSON-serializable payload used in progress/events."""
        return {
            "step": step_index + 1,
            "instruction": self.instruction,
            "completion_mode": self.completion_mode,
            "status": self.status,
            "verify_reason": self.verify_reason,
            "duration_s": round(self.duration_s, 1),
            "image_paths": self.image_paths,
        }


class PlanExecutionStopped(Exception):
    """Raised when the caller marks a plan as no longer running."""


async def execute_plan_steps(
    arm: Any,
    steps: list[dict[str, Any]],
    *,
    is_running: Callable[[], bool] | None = None,
    on_step_start: Callable[[int, dict[str, Any]], None] | None = None,
    on_step_complete: Callable[[int, PlanStepResult], None] | None = None,
    save_step_images: Callable[[int, list[dict[str, Any]]], list[str]] | None = None,
) -> list[PlanStepResult]:
    """Execute plan steps sequentially using manual backend verification."""
    results: list[PlanStepResult] = []

    for index, step in enumerate(steps):
        if is_running is not None and not is_running():
            raise PlanExecutionStopped()

        instruction = str(step.get("instruction", ""))
        completion_mode = str(step.get("completion_mode") or "single")
        started_at = time.time()
        if on_step_start is not None:
            on_step_start(index, step)

        result = PlanStepResult(instruction=instruction, completion_mode=completion_mode)
        start_response = await arm.start_task(
            instruction,
            completion_mode=completion_mode,
            verify_mode="manual",
        )
        if start_response.get("error") or start_response.get("status") == "error":
            result.status = "failed"
            result.verify_reason = start_response.get("error", "start failed")
            result.duration_s = time.time() - started_at
            _complete_step(results, result, index, on_step_complete)
            return results

        interval_s = arm.get_verify_interval_s()
        timeout_s = _get_verify_timeout_s(arm)
        max_polls = max(1, int(timeout_s / interval_s))

        for _ in range(max_polls):
            await asyncio.sleep(interval_s)
            if is_running is not None and not is_running():
                raise PlanExecutionStopped()

            verify_result = await arm.verify_once()
            if verify_result.get("error"):
                result.status = "failed"
                result.verify_reason = verify_result["error"]
                result.duration_s = time.time() - started_at
                break

            verdict = verify_result.get("verdict")
            if verdict == "completed":
                result.status = "completed"
                result.verify_reason = verify_result.get("reason", "")
                result.duration_s = time.time() - started_at
                break
            if verdict in {"failed", "error"}:
                result.status = "failed"
                result.verify_reason = verify_result.get("reason", "")
                result.duration_s = time.time() - started_at
                break
        else:
            result.status = "failed"
            result.verify_reason = "timeout"
            result.duration_s = time.time() - started_at
            try:
                await arm.stop_task()
            except Exception:
                logger.warning("Failed to stop arm after timeout", exc_info=True)

        observation = await arm.observe(["main"])
        if observation.get("status") != "error" and save_step_images is not None:
            result.image_paths = save_step_images(index, observation.get("images", []))

        _complete_step(results, result, index, on_step_complete)
        if result.status != "completed":
            return results

    return results


def _complete_step(
    results: list[PlanStepResult],
    result: PlanStepResult,
    index: int,
    on_step_complete: Callable[[int, PlanStepResult], None] | None,
) -> None:
    results.append(result)
    if on_step_complete is not None:
        on_step_complete(index, result)


def _get_verify_timeout_s(arm: Any) -> float:
    getter = getattr(arm, "get_verify_timeout_s", None)
    if getter is None:
        return DEFAULT_PLAN_TIMEOUT_S
    timeout_s = float(getter())
    if timeout_s <= 0:
        return DEFAULT_PLAN_TIMEOUT_S
    return timeout_s
