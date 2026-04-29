"""Configuration management for the single-arm VLA Agent."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_CANDIDATES = (
    Path.home() / ".vlapilot" / "config.json",
    Path.home() / ".vla" / "config.json",
)


@dataclass
class ModelConfig:
    """LLM model configuration."""
    provider: str
    model: str
    api_base: str
    api_key: str = ""
    proxy: str | None | bool = None


@dataclass
class AgentConfig:
    """Planner-side agent configuration."""
    mission_dir: str
    planner: ModelConfig
    session_dir: str
    max_iterations: int = 50
    log_level: str = "INFO"
    log_file: str | None = None
    debug: bool = False
    debug_dir: str | None = None


@dataclass
class ArmConfig:
    """Single arm runtime and MCP client configuration."""
    mcp_command: str
    mcp_args: list[str]
    capability_tasks_file: str
    policy_path: str
    can_interface: str
    device: str
    cameras: dict[str, Any]
    verify: dict[str, Any]


@dataclass
class VLAAgentConfig:
    """Top-level single-arm application configuration."""
    agent: AgentConfig
    arm: ArmConfig


def resolve_config_path(config_path: str | Path | None = None) -> Path:
    """Resolve explicit config path, otherwise prefer ~/.vlapilot then ~/.vla."""
    if config_path is not None:
        path = Path(config_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        return path

    for candidate in DEFAULT_CONFIG_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Config file not found. Checked: "
        + ", ".join(str(path) for path in DEFAULT_CONFIG_CANDIDATES)
    )


def load_config(config_path: str | Path | None = None) -> VLAAgentConfig:
    """Load the single-arm agent config from one JSON file."""
    path = resolve_config_path(config_path)

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    def resolve_path(raw: str | Path) -> Path:
        candidate = Path(raw).expanduser()
        if candidate.is_absolute():
            return candidate
        return (path.parent / candidate).resolve()

    agent_data = data.get("agent")
    arm_data = data.get("arm")
    if not isinstance(agent_data, dict):
        raise ValueError("config.agent is required")
    if not isinstance(arm_data, dict):
        raise ValueError("config.arm is required")

    planner_data = agent_data.get("planner")
    if not isinstance(planner_data, dict):
        raise ValueError("config.agent.planner is required")

    mission_dir = resolve_path(str(agent_data["mission_dir"]))
    if not mission_dir.exists():
        raise FileNotFoundError(f"Mission directory not found: {mission_dir}")
    for filename in ("mission.md", "tasks.yaml", "verify.md"):
        mission_file = mission_dir / filename
        if not mission_file.exists():
            raise FileNotFoundError(f"Mission file not found: {mission_file}")

    capability_tasks_file = resolve_path(str(arm_data["capability_tasks_file"]))
    if not capability_tasks_file.exists():
        raise FileNotFoundError(f"Capability task file not found: {capability_tasks_file}")

    mcp_args = arm_data.get("mcp_args")
    if not isinstance(mcp_args, list):
        mcp_args = ["--config", str(path)]

    agent = AgentConfig(
        mission_dir=str(mission_dir),
        planner=ModelConfig(
            provider=planner_data["provider"],
            model=planner_data["model"],
            api_base=planner_data.get("api_base", ""),
            api_key=planner_data.get("api_key", ""),
            proxy=planner_data.get("proxy"),
        ),
        session_dir=str(resolve_path(agent_data.get("session_dir", Path.home() / ".vlapilot" / "sessions"))),
        max_iterations=agent_data.get("max_iterations", 50),
        log_level=agent_data.get("log_level", "INFO"),
        log_file=str(resolve_path(agent_data["log_file"])) if agent_data.get("log_file") else None,
        debug=agent_data.get("debug", False),
        debug_dir=str(resolve_path(agent_data["debug_dir"])) if agent_data.get("debug_dir") else None,
    )
    arm = ArmConfig(
        mcp_command=arm_data["mcp_command"],
        mcp_args=[str(arg) for arg in mcp_args],
        capability_tasks_file=str(capability_tasks_file),
        policy_path=arm_data["policy_path"],
        can_interface=arm_data["can_interface"],
        device=arm_data.get("device", "cuda"),
        cameras=dict(arm_data.get("cameras", {})),
        verify=dict(arm_data.get("verify", {})),
    )
    return VLAAgentConfig(agent=agent, arm=arm)
