"""Single-arm MCP client wrapper."""

from __future__ import annotations

import json
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from vlapilot.config import ArmConfig
from vlapilot.core.mission import load_tasks

DEFAULT_VERIFY_INTERVAL_S = 2.0
DEFAULT_VERIFY_TIMEOUT_S = 180.0


class ArmClient:
    """Manage one backend MCP session for one arm."""

    def __init__(self, config: ArmConfig):
        self.config = config
        self._client: ClientSession | None = None
        self._stdio_cm: Any | None = None
        self._session_cm: Any | None = None
        self._capability_tasks = load_tasks(self.config.capability_tasks_file)

    async def init(self) -> None:
        """Start the arm MCP process and initialize the client session."""
        if self._client is not None:
            return

        server_params = StdioServerParameters(
            command=self.config.mcp_command,
            args=self.config.mcp_args,
        )
        self._stdio_cm = stdio_client(server_params)
        read_stream, write_stream = await self._stdio_cm.__aenter__()

        self._session_cm = ClientSession(read_stream, write_stream)
        await self._session_cm.__aenter__()
        await self._session_cm.initialize()
        self._client = self._session_cm

    def get_capability_tasks(self) -> list[dict[str, Any]]:
        """Return the configured checkpoint task vocabulary."""
        return list(self._capability_tasks)

    async def observe(self, cameras: list[str] | None = None) -> dict[str, Any]:
        """Observe the current scene and return images plus status text."""
        if cameras is None:
            cameras = ["main", "wrist"]

        client = self._require_client()
        result = await client.call_tool("vla_observe", {"cameras": cameras})
        response: dict[str, Any] = {"status": "ok", "cameras": cameras, "images": []}

        for content in result.content:
            if hasattr(content, "text"):
                response["status_text"] = content.text
                text = content.text.strip()
                if text.startswith("{") and text.endswith("}"):
                    try:
                        payload = json.loads(text)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(payload, dict) and payload.get("status") == "error":
                        return payload
            elif hasattr(content, "data") and hasattr(content, "mimeType"):
                response["images"].append(
                    {
                        "data": content.data,
                        "mimeType": content.mimeType,
                    }
                )

        return response

    async def start_task(
        self,
        instruction: str,
        completion_mode: str = "single",
        verify_mode: str = "manual",
    ) -> dict[str, Any]:
        """Start one task on the arm."""
        return await self._call_json_tool(
            "vla_start",
            {
                "instruction": instruction,
                "completion_mode": completion_mode,
                "verify_mode": verify_mode,
            },
        )

    async def verify_once(self) -> dict[str, Any]:
        """Run one manual verification step."""
        return await self._call_json_tool("vla_verify_once", {})

    def get_verify_interval_s(self) -> float:
        """Return the configured manual verify polling interval."""
        raw_interval = self.config.verify.get("interval_s", DEFAULT_VERIFY_INTERVAL_S)
        interval_s = float(raw_interval)
        if interval_s <= 0:
            return DEFAULT_VERIFY_INTERVAL_S
        return interval_s

    def get_verify_timeout_s(self) -> float:
        """Return the configured manual verify timeout."""
        raw_timeout = self.config.verify.get("timeout_s", DEFAULT_VERIFY_TIMEOUT_S)
        timeout_s = float(raw_timeout)
        if timeout_s <= 0:
            return DEFAULT_VERIFY_TIMEOUT_S
        return timeout_s

    async def get_status(self) -> dict[str, Any]:
        """Get current runtime status."""
        return await self._call_json_tool("vla_status", {})

    async def stop_task(self) -> dict[str, Any]:
        """Stop the current arm task."""
        return await self._call_json_tool("vla_stop", {})

    async def cleanup(self) -> None:
        """Close the MCP session."""
        if self._session_cm is not None:
            try:
                await self._session_cm.__aexit__(None, None, None)
            except Exception:
                pass
        if self._stdio_cm is not None:
            try:
                await self._stdio_cm.__aexit__(None, None, None)
            except Exception:
                pass
        self._client = None
        self._session_cm = None
        self._stdio_cm = None

    def _require_client(self) -> ClientSession:
        if self._client is None:
            raise RuntimeError("Arm client is not initialized")
        return self._client

    async def _call_json_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        client = self._require_client()
        result = await client.call_tool(name, arguments)
        for content in result.content:
            if hasattr(content, "text"):
                text = content.text.strip()
                if text.startswith("{") and text.endswith("}"):
                    try:
                        payload = json.loads(text)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(payload, dict):
                        return payload
        return {}
