"""VLA MCP Server - single-arm entry point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from loguru import logger
from mcp.server import Server
from mcp.types import (
    GetPromptResult,
    ImageContent,
    Prompt,
    PromptMessage,
    Resource,
    ResourceTemplate,
    TextContent,
    Tool,
)

from vla_pi05_piper.executor import RTCExecutor
from vla_pi05_piper.runtime import VLARuntime
from vlapilot.core.mission import load_task_instructions


_runtime: VLARuntime | None = None
_config: dict[str, Any] | None = None
_config_path: Path | None = None
_REPO_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_MISSION_DIR = _REPO_ROOT / "examples" / "missions" / "desk_cleanup"
_LEGACY_CONFIG_KEYS = {
    "policy_path",
    "can_interface",
    "device",
    "cameras",
    "verify",
    "models",
    "arms",
    "mission_skill_file",
}
_SECRET_CONFIG_KEYS = {
    "api_key",
    "apikey",
    "access_token",
    "refresh_token",
    "authorization",
    "secret",
}

app = Server("vla-pi05-piper")


def configure_logging() -> None:
    """Enable Loguru color markup for verifier debug output."""
    logger.remove()
    logger.add(
        sys.stderr,
        colorize=True,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "{message}"
        ),
    )


def _resolve_path(raw: str | Path | None) -> Path | None:
    if raw is None:
        return None
    path = Path(raw).expanduser()
    if path.is_absolute() or _config_path is None:
        return path
    return (_config_path.parent / path).resolve()


def _agent_config() -> dict[str, Any]:
    if not isinstance(_config, dict):
        return {}
    agent = _config.get("agent")
    return agent if isinstance(agent, dict) else {}


def _arm_config() -> dict[str, Any]:
    if not isinstance(_config, dict):
        return {}
    arm = _config.get("arm")
    return arm if isinstance(arm, dict) else {}


def _mission_dir() -> Path | None:
    return _resolve_path(_agent_config().get("mission_dir"))


def _capability_tasks_path() -> Path:
    configured = _resolve_path(_arm_config().get("capability_tasks_file"))
    if configured is not None:
        return configured
    return _DEFAULT_MISSION_DIR / "tasks.yaml"


def _verify_prompt_path() -> Path:
    mission_dir = _mission_dir()
    if mission_dir is not None:
        return mission_dir / "verify.md"
    return _DEFAULT_MISSION_DIR / "verify.md"


def _validate_config_shape(data: dict[str, Any], config_path: Path) -> None:
    if isinstance(data.get("agent"), dict) and isinstance(data.get("arm"), dict):
        return

    legacy_keys = sorted(key for key in _LEGACY_CONFIG_KEYS if key in data)
    if legacy_keys:
        raise ValueError(
            "Expected top-level 'agent' and 'arm' in "
            f"{config_path}, but detected legacy flat schema keys: {', '.join(legacy_keys)}"
        )
    raise ValueError(
        f"Expected top-level 'agent' and 'arm' in {config_path}"
    )


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available VLA tools."""
    return [
        Tool(
            name="vla_observe",
            description="Capture current scene from robot cameras",
            inputSchema={
                "type": "object",
                "properties": {
                    "cameras": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["main", "wrist"]},
                        "description": "Cameras to capture (default: ['main'])",
                    }
                },
            },
        ),
        Tool(
            name="vla_start",
            description="Start executing a task instruction",
            inputSchema={
                "type": "object",
                "properties": {
                    "instruction": {
                        "type": "string",
                        "description": "Exact task instruction from vocabulary",
                    },
                    "completion_mode": {
                        "type": "string",
                        "enum": ["single", "until_clear"],
                        "description": "Completion mode (default: 'single')",
                    },
                    "verify_mode": {
                        "type": "string",
                        "enum": ["standalone", "manual"],
                        "description": "standalone: auto verify loop; manual: caller drives verify via vla_verify_once",
                    },
                },
                "required": ["instruction"],
            },
        ),
        Tool(
            name="vla_status",
            description="Get current task status",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="vla_stop",
            description="Stop current task execution",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="vla_verify_once",
            description="Run one verification check (manual mode only). Returns verdict, reason, has_motion.",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent | ImageContent]:
    """Execute tool call."""
    global _runtime

    if _runtime is None:
        return [
            TextContent(
                type="text",
                text=json.dumps(
                    {
                        "error": "VLA runtime is not initialized",
                        "hint": "Restart the server and fix startup errors first",
                    }
                ),
            )
        ]

    try:
        if name == "vla_observe":
            result = await _runtime.observe(arguments.get("cameras"))
            if result.get("status") == "error":
                return [TextContent(type="text", text=json.dumps(result))]

            contents: list[TextContent | ImageContent] = [
                TextContent(type="text", text=result["status_text"])
            ]
            for frame in result.get("frames", {}).values():
                import base64
                import io

                import numpy as np
                from PIL import Image

                if frame.dtype != np.uint8:
                    frame = (frame * 255).astype(np.uint8)
                image = Image.fromarray(frame)
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=85)
                contents.append(
                    ImageContent(
                        type="image",
                        data=base64.b64encode(buffer.getvalue()).decode(),
                        mimeType="image/jpeg",
                    )
                )
            return contents

        if name == "vla_start":
            instruction = arguments.get("instruction")
            if not instruction:
                return [TextContent(type="text", text=json.dumps({"error": "instruction is required"}))]

            valid_instructions = set(load_task_instructions(_capability_tasks_path()))
            if instruction not in valid_instructions:
                return [
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {
                                "error": "Instruction must exactly match one from the vocabulary",
                                "instruction": instruction,
                            }
                        ),
                    )
                ]

            result = await _runtime.start_task(
                instruction,
                arguments.get("completion_mode", "single"),
                arguments.get("verify_mode", "standalone"),
            )
            return [TextContent(type="text", text=json.dumps(result))]

        if name == "vla_status":
            return [TextContent(type="text", text=json.dumps(_runtime.get_status()))]

        if name == "vla_stop":
            _runtime.stop_task()
            return [TextContent(type="text", text=json.dumps({"status": "stopped"}))]

        if name == "vla_verify_once":
            return [TextContent(type="text", text=json.dumps(await _runtime.verify_once()))]

        return [TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]

    except Exception as e:
        logger.error(f"Tool execution error: {e}")
        return [TextContent(type="text", text=json.dumps({"error": str(e)}))]


@app.list_resources()
async def list_resources() -> list[Resource]:
    """List available resources."""
    return [
        Resource(
            uri="vla://tasks/vocabulary",
            name="Task Vocabulary",
            description="List of trained task instructions",
            mimeType="application/yaml",
        ),
        Resource(
            uri="vla://config/current",
            name="Current Configuration",
            description="Current VLA configuration",
            mimeType="application/json",
        ),
    ]


@app.list_resource_templates()
async def list_resource_templates() -> list[ResourceTemplate]:
    """Return an empty template list for MCP clients that probe this method."""
    return []


@app.read_resource()
async def read_resource(uri: str) -> str:
    """Read resource content."""
    uri = str(uri)

    if uri == "vla://tasks/vocabulary":
        tasks_path = _capability_tasks_path()
        if tasks_path.exists():
            return tasks_path.read_text(encoding="utf-8")
        return "# No tasks.yaml found"

    if uri == "vla://config/current":
        if _config:
            safe_config = _redact_config_secrets(_config)
            return json.dumps(safe_config, indent=2)
        return "{}"

    return f"Unknown resource: {uri}"


def _redact_config_secrets(value: Any) -> Any:
    """Return a copy of config-like data with secret values redacted."""
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            if str(key).lower() in _SECRET_CONFIG_KEYS:
                redacted[key] = "***"
            else:
                redacted[key] = _redact_config_secrets(item)
        return redacted
    if isinstance(value, list):
        return [_redact_config_secrets(item) for item in value]
    return value


@app.list_prompts()
async def list_prompts() -> list[Prompt]:
    """List available prompts."""
    return [
        Prompt(
            name="vla_workflow",
            description="VLA robotic arm control workflow guidance",
            arguments=[],
        ),
    ]


@app.get_prompt()
async def get_prompt(name: str, arguments: dict) -> GetPromptResult:
    """Get prompt content."""
    from mcp.types import TextContent as PromptTextContent

    if name != "vla_workflow":
        raise ValueError(f"Unknown prompt: {name}")

    content = """# VLA Robotic Arm Control

You control a Piper robotic arm via VLA (Vision-Language-Action) model.

## Workflow

1. Observe with `vla_observe`
2. Match the scene against `vla://tasks/vocabulary`
3. Start exactly one instruction with `vla_start`
4. Poll `vla_status` or `vla_verify_once`
5. Stop with `vla_stop` if needed

## Important Rules

- Instructions must exactly match the task vocabulary
- Do not start a second task while one is already running
- Prefer `completion_mode="single"` unless the instruction should repeat until the scene is clear
"""
    return GetPromptResult(
        description="VLA robotic arm control workflow guidance",
        messages=[
            PromptMessage(
                role="user",
                content=PromptTextContent(type="text", text=content),
            )
        ],
    )


def _init_runtime() -> VLARuntime:
    """Initialize VLA runtime."""
    global _config

    if _config is None:
        raise RuntimeError("Config not loaded")

    logger.info("Initializing VLA runtime...")
    arm_config = _arm_config()
    if not arm_config:
        raise ValueError("config.arm is required")

    cameras = {}
    for name, cam_cfg in arm_config.get("cameras", {}).items():
        serial = cam_cfg.get("serial_number_or_name") or cam_cfg.get("serial_number")
        if not serial:
            raise ValueError(f"config.arm.cameras.{name}.serial_number_or_name is required")
        cameras[name] = {
            "serial_number_or_name": serial,
            "width": cam_cfg.get("width", 640),
            "height": cam_cfg.get("height", 480),
            "fps": cam_cfg.get("fps", 30),
        }

    executor = RTCExecutor(
        policy_path=str(arm_config["policy_path"]),
        can_interface=str(arm_config["can_interface"]),
        cameras=cameras,
        device=str(arm_config.get("device", "cuda")),
    )
    executor.init()
    logger.info("Executor initialized")

    runtime = VLARuntime(
        executor=executor,
        verify_config=dict(arm_config.get("verify", {})),
        notification_callback=None,
        verify_prompt_path=_verify_prompt_path(),
    )
    logger.info("Runtime initialized")
    return runtime


def load_config(config_path: Path) -> dict[str, Any]:
    """Load configuration from JSON file."""
    resolved = config_path.expanduser()
    if not resolved.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(resolved, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a JSON object: {resolved}")
    _validate_config_shape(data, resolved)
    return data


def _default_config_path() -> Path:
    for candidate in (Path.home() / ".vlapilot" / "config.json", Path.home() / ".vla" / "config.json"):
        if candidate.exists():
            return candidate
    return Path.home() / ".vlapilot" / "config.json"


def main() -> None:
    """Main entry point."""
    global _config, _config_path, _runtime

    configure_logging()

    parser = argparse.ArgumentParser(description="VLA MCP Server")
    parser.add_argument(
        "--config",
        type=Path,
        default=_default_config_path(),
        help="Path to config file (default: ~/.vlapilot/config.json, fallback: ~/.vla/config.json)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode: save verifier request/response artifacts to disk",
    )
    args = parser.parse_args()

    _config_path = args.config.expanduser().resolve()

    try:
        _config = load_config(args.config)
        if args.debug:
            arm_cfg = _config.get("arm", _config)
            verify_cfg = arm_cfg.setdefault("verify", {})
            verify_cfg["debug"] = True
        logger.info(f"Config loaded from {args.config}")
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)

    try:
        _runtime = _init_runtime()
    except Exception as e:
        logger.error(f"Failed to initialize VLA runtime: {e}")
        sys.exit(1)

    logger.info("Starting VLA MCP Server...")

    import anyio
    from mcp.server.stdio import stdio_server

    async def run_server() -> None:
        try:
            async with stdio_server() as (read_stream, write_stream):
                await app.run(read_stream, write_stream, app.create_initialization_options())
        finally:
            if _runtime is not None:
                logger.info("Shutting down VLA runtime...")
                _runtime.cleanup()

    anyio.run(run_server)


if __name__ == "__main__":
    main()
