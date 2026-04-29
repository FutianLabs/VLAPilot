"""Tests for single-arm agent config and mission loading."""

from __future__ import annotations

import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vlapilot.config import load_config
from vlapilot.core.mission import TASK_VOCAB_PLACEHOLDER, load_mission_assets, load_task_prompt


def test_load_mission_assets_injects_task_yaml():
    mission_dir = Path(__file__).resolve().parents[1] / "examples" / "missions" / "desk_cleanup"

    assets = load_mission_assets(mission_dir)

    assert TASK_VOCAB_PLACEHOLDER not in assets.mission_prompt
    assert "Put the pen into the pen holder." in assets.mission_prompt
    assert assets.verify_prompt_path == mission_dir / "verify.md"


def test_load_task_prompt_injects_task_yaml(tmp_path: Path):
    prompt_path = tmp_path / "mission_mcp.md"
    tasks_path = tmp_path / "tasks.yaml"
    prompt_path.write_text("# Planner\n\n{tasks_yaml_injected_here}\n", encoding="utf-8")
    tasks_path.write_text(
        'tasks:\n  - id: 1\n    instruction: "Put the pen into the pen holder."\n',
        encoding="utf-8",
    )

    prompt = load_task_prompt(prompt_path, tasks_path)

    assert TASK_VOCAB_PLACEHOLDER not in prompt
    assert "Put the pen into the pen holder." in prompt


def test_load_config_validates_new_single_arm_schema(tmp_path: Path):
    mission_dir = tmp_path / "mission"
    mission_dir.mkdir()
    (mission_dir / "mission.md").write_text("# Mission\n\n{tasks_yaml_injected_here}\n", encoding="utf-8")
    (mission_dir / "tasks.yaml").write_text(
        'tasks:\n  - id: 1\n    instruction: "Put the pen into the pen holder."\n',
        encoding="utf-8",
    )
    (mission_dir / "verify.md").write_text("verify prompt", encoding="utf-8")

    capability_tasks = tmp_path / "capability.yaml"
    capability_tasks.write_text(
        'tasks:\n  - id: 1\n    instruction: "Put the pen into the pen holder."\n',
        encoding="utf-8",
    )

    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "agent": {
                    "mission_dir": str(mission_dir),
                    "planner": {
                        "provider": "openai",
                        "model": "gpt-test",
                        "api_base": "http://127.0.0.1:8000/v1",
                        "api_key": "test-key",
                    },
                    "session_dir": str(tmp_path / "sessions"),
                },
                "arm": {
                    "mcp_command": "vla-pi05-piper",
                    "capability_tasks_file": str(capability_tasks),
                    "policy_path": "/tmp/policy",
                    "can_interface": "can0",
                    "device": "cpu",
                    "cameras": {},
                    "verify": {"provider": "local", "providers": {"local": {"model": "verify", "api_base": "http://127.0.0.1"}}},
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_config(str(config_path))

    assert config.agent.mission_dir == str(mission_dir)
    assert config.agent.session_dir == str((tmp_path / "sessions").expanduser())
    assert config.arm.mcp_args == ["--config", str(config_path)]
