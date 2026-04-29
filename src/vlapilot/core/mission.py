"""Mission loading helpers for the single-arm agent runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


TASK_VOCAB_PLACEHOLDER = "{tasks_yaml_injected_here}"


@dataclass(frozen=True)
class MissionAssets:
    """Resolved mission assets from one mission directory."""

    mission_dir: Path
    mission_prompt: str
    mission_tasks: list[dict[str, Any]]
    verify_prompt_path: Path


def load_tasks(tasks_path: Path | str) -> list[dict[str, Any]]:
    """Load a task vocabulary file."""
    path = Path(tasks_path).expanduser()
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    tasks: list[dict[str, Any]] = []
    for raw_task in data.get("tasks", []):
        if not isinstance(raw_task, dict):
            continue
        instruction = raw_task.get("instruction")
        if not instruction:
            continue
        task = dict(raw_task)
        task["instruction"] = str(instruction)
        tasks.append(task)
    return tasks


def load_task_instructions(tasks_path: Path | str) -> list[str]:
    """Load just the instruction strings from a task vocabulary file."""
    return [task["instruction"] for task in load_tasks(tasks_path)]


def intersect_task_vocab(
    mission_tasks: list[str],
    capability_tasks: list[str],
) -> list[str]:
    """Return mission tasks that the current arm checkpoint can execute."""
    capability_set = set(capability_tasks)
    return [instruction for instruction in mission_tasks if instruction in capability_set]


def _inject_tasks_into_prompt(template: str, tasks_yaml_text: str) -> str:
    """Inject the raw mission task YAML into the mission prompt template."""
    if TASK_VOCAB_PLACEHOLDER in template:
        return template.replace(TASK_VOCAB_PLACEHOLDER, tasks_yaml_text.strip())

    return (
        f"{template.rstrip()}\n\n"
        "## Trained Task Vocabulary\n"
        f"{tasks_yaml_text.strip()}\n"
    )


def load_task_prompt(prompt_path: Path | str, tasks_path: Path | str) -> str:
    """Load a prompt file and inject the raw task YAML into it."""
    prompt = Path(prompt_path).expanduser().read_text(encoding="utf-8").strip()
    tasks_yaml_text = Path(tasks_path).expanduser().read_text(encoding="utf-8")
    return _inject_tasks_into_prompt(prompt, tasks_yaml_text)


def load_mission_assets(mission_dir: Path | str) -> MissionAssets:
    """Load mission prompt, mission task vocabulary, and verify prompt path."""
    root = Path(mission_dir).expanduser()
    mission_path = root / "mission.md"
    tasks_path = root / "tasks.yaml"
    verify_path = root / "verify.md"

    missing = [str(path) for path in (mission_path, tasks_path, verify_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing mission files: {', '.join(missing)}")

    return MissionAssets(
        mission_dir=root,
        mission_prompt=load_task_prompt(mission_path, tasks_path),
        mission_tasks=load_tasks(tasks_path),
        verify_prompt_path=verify_path,
    )
