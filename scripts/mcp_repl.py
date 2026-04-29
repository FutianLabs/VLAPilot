#!/usr/bin/env python3
"""Interactive MCP client for testing the VLA server over stdio."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import shlex
import sys
import textwrap
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from mcp import ClientSession, types
from mcp.client.stdio import StdioServerParameters, stdio_client


DEFAULT_PYTHON_BIN = "/home/qing/projects/clawvla/clawvla/.venv/bin/python"
DEFAULT_CONFIG_CANDIDATES = (
    Path.home() / ".vlapilot" / "config.json",
    Path.home() / ".vla" / "config.json",
)
DEFAULT_SAVE_DIR = Path("/tmp/vlapilot-mcp-client")
VALID_COMPLETION_MODES = {"single", "until_clear"}
SERVER_MODULES = {
    "arm": "vla_pi05_piper.server",
    "agent": "vlapilot.interface.mcp_server",
}
LEGACY_CONFIG_KEYS = {
    "policy_path",
    "can_interface",
    "device",
    "cameras",
    "verify",
    "models",
    "arms",
    "mission_skill_file",
}


@dataclass
class ObserveResult:
    text_blocks: list[str]
    image_paths: list[Path]
    payload: dict[str, Any] | None


def validate_config_shape(config_path: Path) -> None:
    """Fail fast with a clear message before spawning the MCP server."""
    resolved = config_path.expanduser()
    if not resolved.exists():
        raise FileNotFoundError(f"Config file not found: {resolved}")

    with open(resolved, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a JSON object: {resolved}")

    if isinstance(data.get("agent"), dict) and isinstance(data.get("arm"), dict):
        return

    legacy_keys = sorted(key for key in LEGACY_CONFIG_KEYS if key in data)
    if legacy_keys:
        raise ValueError(
            "Legacy config schema detected in "
            f"{resolved}. Expected top-level 'agent' and 'arm', "
            f"but found legacy keys: {', '.join(legacy_keys)}."
        )

    raise ValueError(
        f"Invalid config schema in {resolved}. Expected top-level 'agent' and 'arm'."
    )


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Interactive MCP client for vlapilot")
    parser.add_argument(
        "--python-bin",
        default=DEFAULT_PYTHON_BIN,
        help=f"Python used to launch the MCP server (default: {DEFAULT_PYTHON_BIN})",
    )
    parser.add_argument(
        "--server",
        choices=sorted(SERVER_MODULES),
        default="arm",
        help="Which MCP server to connect to (default: arm)",
    )
    parser.add_argument(
        "--server-module",
        default=None,
        help="Override the MCP server module (default: selected by --server)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config_path(),
        help="Config file path (default: ~/.vlapilot/config.json, fallback: ~/.vla/config.json)",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=DEFAULT_SAVE_DIR,
        help=f"Directory used to save observe images (default: {DEFAULT_SAVE_DIR})",
    )
    parser.add_argument(
        "--cwd",
        type=Path,
        default=repo_root,
        help=f"Working directory for the MCP server process (default: {repo_root})",
    )
    return parser.parse_args()


def default_config_path() -> Path:
    for candidate in DEFAULT_CONFIG_CANDIDATES:
        if candidate.exists():
            return candidate
    return DEFAULT_CONFIG_CANDIDATES[0]


def build_server_params(args: argparse.Namespace) -> StdioServerParameters:
    repo_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = merge_pythonpath(repo_root / "src", env.get("PYTHONPATH"))
    module = args.server_module or SERVER_MODULES[args.server]
    return StdioServerParameters(
        command=args.python_bin,
        args=["-m", module, "--config", str(args.config)],
        env=env,
        cwd=args.cwd,
    )


def merge_pythonpath(src_path: Path, current: str | None) -> str:
    parts = [str(src_path)]
    if current:
        parts.append(current)
    return os.pathsep.join(parts)


def parse_cameras(raw: str | None) -> list[str]:
    if raw is None:
        return ["main", "wrist"]
    cameras = [item.strip() for item in raw.split(",") if item.strip()]
    return cameras or ["main", "wrist"]


def parse_instruction_mode(parts: list[str]) -> tuple[str, str]:
    if not parts:
        raise ValueError("instruction is required")

    mode = "single"
    instruction_parts = parts
    if parts[-1] in VALID_COMPLETION_MODES:
        mode = parts[-1]
        instruction_parts = parts[:-1]

    instruction = " ".join(instruction_parts).strip()
    if not instruction:
        raise ValueError("instruction is required")
    return instruction, mode


def parse_tool_json(result: types.CallToolResult) -> dict[str, Any] | None:
    for block in result.content:
        if isinstance(block, types.TextContent):
            text = block.text.strip()
            if text.startswith("{") and text.endswith("}"):
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    return payload
    return None


def collect_text_blocks(result: types.CallToolResult) -> list[str]:
    blocks: list[str] = []
    for block in result.content:
        if isinstance(block, types.TextContent):
            blocks.append(block.text)
    return blocks


def save_image_blocks(
    result: types.CallToolResult,
    save_root: Path,
    prefix: str,
) -> list[Path]:
    save_root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    image_index = 0
    for block in result.content:
        if not isinstance(block, types.ImageContent):
            continue
        image_index += 1
        ext = mime_extension(block.mimeType)
        target = save_root / f"{prefix}_{image_index:02d}{ext}"
        target.write_bytes(base64.b64decode(block.data))
        written.append(target)
    return written


def mime_extension(mime_type: str) -> str:
    return {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
    }.get(mime_type, ".bin")


def format_status(status: dict[str, Any]) -> str:
    fields = {
        "state": status.get("state"),
        "instruction": status.get("instruction"),
        "completion_mode": status.get("completion_mode"),
        "duration_s": status.get("duration_s"),
        "verify_step": status.get("verify_step"),
        "last_verify_verdict": status.get("last_verify_verdict"),
        "last_verify_reason": status.get("last_verify_reason"),
        "last_verify_latency_ms": status.get("last_verify_latency_ms"),
        "error": status.get("error"),
    }
    return json.dumps(fields, ensure_ascii=False, indent=2)


class VlaMcpRepl:
    """Small interactive MCP client for VLA tool testing."""

    def __init__(self, session: ClientSession, args: argparse.Namespace):
        self._session = session
        self._args = args
        self._observe_counter = 0

    async def run(self) -> int:
        await self._print_banner()
        while True:
            try:
                line = await asyncio.to_thread(input, "vla> ")
            except EOFError:
                print()
                return 0
            except KeyboardInterrupt:
                print()
                return 0

            line = line.strip()
            if not line:
                continue

            try:
                parts = shlex.split(line)
            except ValueError as e:
                print(f"Parse error: {e}")
                continue

            command = parts[0].lower()
            argv = parts[1:]

            try:
                should_exit = await self._dispatch(command, argv)
            except KeyboardInterrupt:
                print("\nInterrupted")
                continue
            except Exception as e:
                print(f"Command failed: {e}")
                continue

            if should_exit:
                return 0

    async def _print_banner(self) -> None:
        init_result = await self._session.initialize()
        tools_result = await self._session.list_tools()
        tool_names = ", ".join(tool.name for tool in tools_result.tools)
        print(
            textwrap.dedent(
                f"""\
                Connected to {init_result.serverInfo.name} {init_result.serverInfo.version}
                Available tools: {tool_names}
                Type 'help' for commands.
                """
            ).strip()
        )

    async def _dispatch(self, command: str, argv: list[str]) -> bool:
        if command in {"quit", "exit"}:
            return True
        if command == "help":
            self._print_help()
            return False
        if command == "tools":
            await self._cmd_tools()
            return False
        if command == "observe":
            cameras = parse_cameras(argv[0] if argv else None)
            await self._cmd_observe(cameras)
            return False
        if command == "start":
            instruction, mode = parse_instruction_mode(argv)
            await self._cmd_start(instruction, mode)
            return False
        if command == "status":
            await self._cmd_status()
            return False
        if command == "stop":
            await self._cmd_stop()
            return False
        if command == "scene":
            cameras = parse_cameras(argv[0] if argv else None)
            await self._cmd_generic("vla_scene", {"cameras": cameras})
            return False
        if command == "plan":
            instruction = " ".join(argv).strip()
            if not instruction:
                print("Usage: plan <instruction>")
                return False
            await self._cmd_generic("vla_plan", {"instruction": instruction})
            return False
        if command == "execute":
            plan_id = argv[0] if argv else ""
            if not plan_id:
                print("Usage: execute <plan_id>")
                return False
            await self._cmd_generic("vla_execute", {"plan_id": plan_id})
            return False
        if command == "progress":
            plan_id = argv[0] if argv else ""
            if not plan_id:
                print("Usage: progress <plan_id>")
                return False
            await self._cmd_generic("vla_progress", {"plan_id": plan_id})
            return False
        print(f"Unknown command: {command}")
        self._print_help()
        return False

    def _print_help(self) -> None:
        print(
            textwrap.dedent(
                """\
                Commands (arm server):
                  observe [main|wrist|main,wrist]
                  start <instruction> [single|until_clear]
                  status
                  stop

                Commands (agent server):
                  scene [main|wrist|main,wrist]
                  plan <instruction>
                  execute <plan_id>
                  progress <plan_id>
                  stop

                General:
                  help / tools / quit
                """
            ).strip()
        )

    async def _cmd_tools(self) -> None:
        result = await self._session.list_tools()
        for tool in result.tools:
            description = tool.description or ""
            print(f"- {tool.name}: {description}")

    async def _cmd_observe(self, cameras: list[str]) -> ObserveResult:
        print(f"Calling vla_observe(cameras={cameras})")
        result = await self._session.call_tool("vla_observe", {"cameras": cameras})
        payload = parse_tool_json(result)
        text_blocks = collect_text_blocks(result)
        save_dir = self._next_observe_dir()
        image_paths = save_image_blocks(result, save_dir, "observe")

        for text in text_blocks:
            print(text)
        if payload and payload.get("error"):
            print(f"observe error: {payload['error']}")
        if image_paths:
            print("Saved images:")
            for path in image_paths:
                print(f"- {path}")

        return ObserveResult(text_blocks=text_blocks, image_paths=image_paths, payload=payload)

    async def _cmd_start(self, instruction: str, completion_mode: str) -> dict[str, Any]:
        print(f"Calling vla_start(instruction={instruction!r}, completion_mode={completion_mode})")
        result = await self._session.call_tool(
            "vla_start",
            {
                "instruction": instruction,
                "completion_mode": completion_mode,
            },
        )
        payload = parse_tool_json(result) or {}
        if payload:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            for text in collect_text_blocks(result):
                print(text)
        return payload

    async def _cmd_status(self) -> dict[str, Any]:
        result = await self._session.call_tool("vla_status", {})
        payload = parse_tool_json(result) or {}
        if payload:
            print(format_status(payload))
        else:
            for text in collect_text_blocks(result):
                print(text)
        return payload

    async def _cmd_stop(self) -> dict[str, Any]:
        print("Calling vla_stop()")
        result = await self._session.call_tool("vla_stop", {})
        payload = parse_tool_json(result) or {}
        if payload:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            for text in collect_text_blocks(result):
                print(text)
        return payload

    async def _cmd_generic(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        print(f"Calling {tool_name}({json.dumps(arguments, ensure_ascii=False)})")
        result = await self._session.call_tool(tool_name, arguments)
        payload = parse_tool_json(result) or {}
        if payload:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            for text in collect_text_blocks(result):
                print(text)
        return payload

    def _next_observe_dir(self) -> Path:
        self._observe_counter += 1
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        save_dir = self._args.save_dir / f"{stamp}-{self._observe_counter:02d}"
        save_dir.mkdir(parents=True, exist_ok=True)
        return save_dir


async def async_main(args: argparse.Namespace) -> int:
    validate_config_shape(args.config)
    server = build_server_params(args)
    async with stdio_client(server) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            repl = VlaMcpRepl(session, args)
            return await repl.run()


def main() -> int:
    args = parse_args()
    try:
        return asyncio.run(async_main(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
