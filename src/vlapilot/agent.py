"""Programmatic plan-mode runner used by scripts/run_agent.py."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

from vlapilot.backend.mcp_client import ArmClient
from vlapilot.config import load_config
from vlapilot.core.execution import PlanStepResult, execute_plan_steps
from vlapilot.core.mission import (
    intersect_task_vocab,
    load_mission_assets,
    load_task_instructions,
    load_task_prompt,
)
from vlapilot.core.plan import create_plan
from vlapilot.core.session import SessionManager
from vlapilot.infra.llm import LLMClient
from vlapilot.infra.logging_config import configure_logging


class VLAAgent:
    """Run one high-level instruction through the vla_plan execution pipeline."""

    def __init__(self, config_path: str | Path | None = None, debug: bool = False):
        self.config = load_config(config_path)
        if debug:
            self.config.agent.debug = True

        configure_logging(
            level=self.config.agent.log_level,
            log_file=self.config.agent.log_file,
            debug=self.config.agent.debug,
        )

        self.mission = load_mission_assets(self.config.agent.mission_dir)
        arm_tasks = load_task_instructions(self.config.arm.capability_tasks_file)
        self.eligible_tasks = intersect_task_vocab(
            [task["instruction"] for task in self.mission.mission_tasks],
            arm_tasks,
        )
        if not self.eligible_tasks:
            raise ValueError("Mission task vocabulary and arm capability vocabulary have no overlap")

        planner_prompt_path = Path(self.config.agent.mission_dir) / "mission_mcp.md"
        if not planner_prompt_path.exists():
            raise FileNotFoundError(f"Missing MCP planner prompt file: {planner_prompt_path}")
        self.planner_prompt = load_task_prompt(
            planner_prompt_path,
            Path(self.config.agent.mission_dir) / "tasks.yaml",
        )

        planner = self.config.agent.planner
        self.planner_llm = LLMClient(
            provider=planner.provider,
            model=planner.model,
            api_base=planner.api_base,
            api_key=planner.api_key,
            proxy=planner.proxy,
        )
        self.arm = ArmClient(self.config.arm)
        self.session = SessionManager(self.config.agent.session_dir)

    async def run(self, instruction: str) -> dict[str, Any]:
        """Create a plan, execute it step by step, and return a terminal summary."""
        await self.arm.init()
        observation = await self.arm.observe(["main", "wrist"])
        if observation.get("status") == "error":
            return {
                "status": "failed",
                "error": "Failed to observe scene",
                "detail": observation.get("error", "unknown error"),
            }

        images = observation.get("images", [])
        steps = await create_plan(
            self.planner_llm,
            instruction=instruction,
            observation_summary=observation.get("status_text", ""),
            images_b64=[image["data"] for image in images if image.get("data")],
            system_prompt=self.planner_prompt,
            valid_instructions=set(self.eligible_tasks),
        )

        plan_id = f"p_{uuid4().hex[:8]}"
        self.session.start_mission(instruction)
        image_paths = self.session.save_observe_images(images, prefix="plan")
        self._record_observe_event(observation, image_paths)
        self.session.append_event(
            {
                "type": "plan_created",
                "plan_id": plan_id,
                "instruction": instruction,
                "steps": steps,
            }
        )

        try:
            step_results = await execute_plan_steps(
                self.arm,
                steps,
                on_step_start=lambda index, step: self._append_step_start(plan_id, index, step),
                on_step_complete=lambda index, result: self._append_step_complete(plan_id, index, result),
                save_step_images=lambda index, images: self.session.save_observe_images(
                    images,
                    prefix=f"exec-step-{index + 1}",
                ),
            )
            results = [result.to_payload(index) for index, result in enumerate(step_results)]
            failed_index = next(
                (index for index, result in enumerate(step_results) if result.status != "completed"),
                None,
            )
            status = "failed" if failed_index is not None else "completed"
            error = ""
            if failed_index is not None:
                failed = step_results[failed_index]
                error = f"Step {failed_index + 1} failed: {failed.verify_reason}"

            if status == "completed":
                self.session.append_event({"type": "plan_completed", "plan_id": plan_id, "steps": len(steps)})
            else:
                self.session.append_event({"type": "plan_failed", "plan_id": plan_id, "error": error})
            self.session.set_metadata(status=status, result={"plan_id": plan_id, "results": results})
            return {
                "status": status,
                "plan_id": plan_id,
                "instruction": instruction,
                "steps": steps,
                "results": results,
                "error": error or None,
            }
        finally:
            self.session.close()

    def _append_step_start(self, plan_id: str, index: int, step: dict[str, Any]) -> None:
        self.session.append_event(
            {
                "type": "step_start",
                "plan_id": plan_id,
                "step": index + 1,
                "instruction": str(step.get("instruction", "")),
                "completion_mode": str(step.get("completion_mode") or "single"),
            }
        )

    def _append_step_complete(self, plan_id: str, index: int, result: PlanStepResult) -> None:
        self.session.append_event({"type": "step_complete", "plan_id": plan_id, **result.to_payload(index)})

    def _record_observe_event(self, result: dict[str, Any], image_paths: list[str]) -> None:
        payload = {
            "type": "observe",
            "status": result.get("status", "unknown"),
            "summary": result.get("status_text", result.get("error", "")),
            "error": result.get("error"),
            "cameras": result.get("cameras", []),
            "image_paths": image_paths,
        }
        self.session.set_latest_observe(payload)
        self.session.append_event(payload)

    async def cleanup(self) -> None:
        await self.planner_llm.close()
        await self.arm.cleanup()
