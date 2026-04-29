"""Planning helpers for both the legacy runtime and the MCP planner."""

from __future__ import annotations

import json
import re
from typing import Any

from loguru import logger

from vlapilot.infra.llm import LLMClient

VALID_COMPLETION_MODES = {"single", "until_clear"}


async def create_plan(
    llm: LLMClient,
    *,
    instruction: str,
    observation_summary: str,
    images_b64: list[str],
    system_prompt: str,
    valid_instructions: set[str],
) -> list[dict[str, Any]]:
    """Call the planner model and validate the returned task sequence."""
    user_content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"User instruction:\n{instruction}\n\n"
                f"Current observation summary:\n{observation_summary or '(none)'}\n\n"
                "Return strict JSON only."
            ),
        }
    ]
    for image_b64 in images_b64:
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
            }
        )

    response = await llm.chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
    )
    content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
    steps = _parse_steps(content)
    _validate_steps(steps, valid_instructions)
    return steps


def _parse_steps(content: str) -> list[dict[str, Any]]:
    """Extract a steps list from the planner response."""
    content = _strip_code_fences(content)
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        data = None

    if isinstance(data, dict) and isinstance(data.get("steps"), list):
        return data["steps"]
    if isinstance(data, list):
        return data

    match = re.search(r"\{.*\"steps\"\s*:\s*\[.*\]\s*\}", content, re.DOTALL)
    if match:
        data = json.loads(match.group())
        steps = data.get("steps")
        if isinstance(steps, list):
            return steps

    raise ValueError(f"Failed to parse planner output: {content[:200]}")


def _validate_steps(steps: list[dict[str, Any]], valid_instructions: set[str]) -> None:
    """Validate the planner output against the execution contract."""
    for index, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            raise ValueError(f"Planner step {index} is not an object")

        instruction = str(step.get("instruction", "")).strip()
        if not instruction:
            raise ValueError(f"Planner step {index} is missing instruction")
        if instruction not in valid_instructions:
            logger.warning("Planner step instruction not in vocabulary: {!r}", instruction)
            raise ValueError(f"Planner step {index} instruction not in vocabulary: {instruction!r}")

        completion_mode = str(step.get("completion_mode", "")).strip()
        if completion_mode not in VALID_COMPLETION_MODES:
            raise ValueError(
                f"Planner step {index} has invalid completion_mode: {completion_mode!r}"
            )

        rationale = str(step.get("rationale", "")).strip()
        if not rationale:
            raise ValueError(f"Planner step {index} is missing rationale")


def _strip_code_fences(text: str) -> str:
    """Remove surrounding markdown fences if present."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()
