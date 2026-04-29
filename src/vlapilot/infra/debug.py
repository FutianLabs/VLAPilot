"""Debug utilities for VLA Agent."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


class DebugRecorder:
    """Records debug information during mission execution."""

    def __init__(self, debug_dir: str | None = None, enabled: bool = False):
        self.enabled = enabled
        if not enabled:
            self.debug_dir = None
            return

        if debug_dir:
            self.debug_dir = Path(debug_dir).expanduser()
        else:
            self.debug_dir = Path(__file__).resolve().parents[2] / "debug" / "planner"

        # Create session directory
        session_name = f"session-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.session_dir = self.debug_dir / session_name
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self.iteration = 0

    def record_llm_call(
        self,
        iteration: int,
        messages: list[dict],
        response: dict,
        tools: list[dict] | None = None
    ):
        """Record LLM call details."""
        if not self.enabled:
            return

        filename = f"iter{iteration:03d}-llm-call.json"
        filepath = self.session_dir / filename

        data = {
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "messages": messages,
            "tools": tools,
            "response": response
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def record_tool_execution(
        self,
        iteration: int,
        tool_name: str,
        arguments: dict,
        result: str,
        error: str | None = None
    ):
        """Record tool execution details."""
        if not self.enabled:
            return

        filename = f"iter{iteration:03d}-tool-{tool_name}.json"
        filepath = self.session_dir / filename

        data = {
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "tool_name": tool_name,
            "arguments": arguments,
            "result": result,
            "error": error
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def record_context(self, iteration: int, context: list[dict]):
        """Record full context sent to VLM."""
        if not self.enabled:
            return

        filename = f"iter{iteration:03d}-context.json"
        filepath = self.session_dir / filename

        with open(filepath, "w") as f:
            json.dump(context, f, indent=2, ensure_ascii=False)

    def record_session_snapshot(self, session_data: dict):
        """Record session snapshot."""
        if not self.enabled:
            return

        filename = f"session-snapshot-{datetime.now().strftime('%H%M%S')}.json"
        filepath = self.session_dir / filename

        with open(filepath, "w") as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)

    def record_error(self, error: Exception, context: dict | None = None):
        """Record error details."""
        if not self.enabled:
            return

        filename = f"error-{datetime.now().strftime('%H%M%S')}.json"
        filepath = self.session_dir / filename

        data = {
            "timestamp": datetime.now().isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def get_debug_dir(self) -> Path | None:
        """Get debug directory path."""
        return self.session_dir if self.enabled else None
