"""Canonical mission session history and audit-log management."""

from __future__ import annotations

import base64
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


def find_legal_message_start(messages: list[dict[str, Any]]) -> int:
    """Find the first index whose tool results have matching assistant calls."""
    declared: set[str] = set()
    start = 0
    for i, message in enumerate(messages):
        role = message.get("role")
        if role == "assistant":
            for tool_call in message.get("tool_calls") or []:
                if isinstance(tool_call, dict) and tool_call.get("id"):
                    declared.add(str(tool_call["id"]))
        elif role == "tool":
            tool_call_id = message.get("tool_call_id")
            if tool_call_id and str(tool_call_id) not in declared:
                start = i + 1
                declared.clear()
                for previous in messages[start : i + 1]:
                    if previous.get("role") == "assistant":
                        for tool_call in previous.get("tool_calls") or []:
                            if isinstance(tool_call, dict) and tool_call.get("id"):
                                declared.add(str(tool_call["id"]))
    return start


@dataclass
class Session:
    """Canonical planner-facing message history for one mission."""

    key: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_history(self, max_messages: int = 500) -> list[dict[str, Any]]:
        """Return a legal recent suffix for planner input."""
        if max_messages > 0:
            sliced = self.messages[-max_messages:]
        else:
            sliced = list(self.messages)

        for index, message in enumerate(sliced):
            if message.get("role") == "user":
                sliced = sliced[index:]
                break

        start = find_legal_message_start(sliced)
        if start:
            sliced = sliced[start:]

        history: list[dict[str, Any]] = []
        for message in sliced:
            entry = {"role": message["role"], "content": message.get("content", "")}
            for key in ("tool_calls", "tool_call_id", "name", "reasoning_content"):
                if key in message:
                    entry[key] = message[key]
            history.append(entry)
        return history


class SessionManager:
    """Manages canonical mission session history plus audit artifacts."""

    def __init__(self, session_dir: str):
        self.session_dir = Path(session_dir).expanduser()
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.current_session: Session | None = None
        self.current_session_path: Path | None = None
        self.current_events_path: Path | None = None
        self.current_mission_id: str | None = None
        self.current_artifact_dir: Path | None = None
        self.current_media_dir: Path | None = None
        self._media_counter = 0

    def start_mission(self, instruction: str) -> str:
        """Start a new mission and persist the canonical user message."""
        mission_id = f"m_{int(time.time())}"
        session_path = self.session_dir / f"{mission_id}.jsonl"
        artifact_dir = self.session_dir / mission_id
        media_dir = artifact_dir / "media"
        events_path = artifact_dir / "events.jsonl"
        media_dir.mkdir(parents=True, exist_ok=True)

        self.current_session = Session(
            key=mission_id,
            metadata={
                "mission_id": mission_id,
                "instruction": instruction,
                "status": "running",
            },
        )
        self.current_session_path = session_path
        self.current_events_path = events_path
        self.current_mission_id = mission_id
        self.current_artifact_dir = artifact_dir
        self.current_media_dir = media_dir
        self._media_counter = 0

        self._save_session()
        self.add_message("user", f"Task: {instruction}")
        return mission_id

    def add_message(self, role: str, content: Any, **kwargs: Any) -> dict[str, Any]:
        """Append one canonical message and persist the session."""
        if not self.current_session:
            raise RuntimeError("No active session. Call start_mission() first.")

        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            **kwargs,
        }
        self.current_session.messages.append(message)
        self.current_session.updated_at = datetime.now()
        self._save_session()
        return message

    def add_messages(self, messages: list[dict[str, Any]]) -> None:
        """Append multiple canonical messages and persist once."""
        if not self.current_session:
            raise RuntimeError("No active session. Call start_mission() first.")

        timestamp = datetime.now().isoformat()
        for message in messages:
            entry = dict(message)
            entry.setdefault("timestamp", timestamp)
            self.current_session.messages.append(entry)
        self.current_session.updated_at = datetime.now()
        self._save_session()

    def append_event(self, event: dict[str, Any]) -> dict[str, Any]:
        """Append one audit event to the mission event log."""
        if not self.current_events_path:
            raise RuntimeError("No active session. Call start_mission() first.")

        payload = {**event, "timestamp": datetime.now().isoformat()}
        with open(self.current_events_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return payload

    def set_metadata(self, **updates: Any) -> None:
        """Update session metadata."""
        if not self.current_session:
            raise RuntimeError("No active session. Call start_mission() first.")
        self.current_session.metadata.update(updates)
        self.current_session.updated_at = datetime.now()
        self._save_session()

    def set_latest_observe(self, observe: dict[str, Any] | None) -> None:
        """Persist the latest observation summary for planner input."""
        self.set_metadata(latest_observe=observe)

    def get_latest_observe(self) -> dict[str, Any] | None:
        """Return the latest observation summary from metadata."""
        metadata = self.get_metadata()
        latest = metadata.get("latest_observe")
        return latest if isinstance(latest, dict) else None

    def save_observe_images(self, images: list[dict[str, Any]], prefix: str = "observe") -> list[str]:
        """Persist observe images to the current mission media directory."""
        if not self.current_media_dir:
            raise RuntimeError("No active session media directory. Call start_mission() first.")

        saved_paths: list[str] = []
        for image in images:
            payload = image.get("data")
            if not isinstance(payload, str) or not payload:
                continue

            mime_type = str(image.get("mimeType") or "application/octet-stream")
            label = self._sanitize_label(str(image.get("camera") or "frame"))
            self._media_counter += 1
            filename = f"{prefix}-{self._media_counter:03d}-{label}{self._mime_extension(mime_type)}"
            path = self.current_media_dir / filename
            path.write_bytes(base64.b64decode(payload))
            saved_paths.append(str(path))

        return saved_paths

    def get_metadata(self) -> dict[str, Any]:
        """Get canonical session metadata."""
        if not self.current_session:
            raise RuntimeError("No active session")
        return dict(self.current_session.metadata)

    def get_history(self, max_messages: int = 500) -> list[dict[str, Any]]:
        """Get canonical planner history."""
        if not self.current_session:
            return []
        return self.current_session.get_history(max_messages=max_messages)

    def get_messages(self) -> list[dict[str, Any]]:
        """Get all canonical messages."""
        if not self.current_session:
            return []
        return [dict(message) for message in self.current_session.messages]

    def get_events(self) -> list[dict[str, Any]]:
        """Get all audit events for the current mission."""
        if not self.current_events_path or not self.current_events_path.exists():
            return []

        events: list[dict[str, Any]] = []
        with open(self.current_events_path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
        return events

    def snapshot(self) -> dict[str, Any]:
        """Return a full in-memory snapshot for debug output."""
        if not self.current_session:
            return {}
        return {
            "metadata": self.get_metadata(),
            "messages": self.get_messages(),
            "events": self.get_events(),
        }

    def close(self) -> None:
        """Close the current mission session."""
        self.current_session = None
        self.current_session_path = None
        self.current_events_path = None
        self.current_mission_id = None
        self.current_artifact_dir = None
        self.current_media_dir = None
        self._media_counter = 0

    def _save_session(self) -> None:
        if not self.current_session or not self.current_session_path:
            raise RuntimeError("No active session. Call start_mission() first.")

        with open(self.current_session_path, "w", encoding="utf-8") as handle:
            metadata_line = {
                "_type": "metadata",
                "key": self.current_session.key,
                "created_at": self.current_session.created_at.isoformat(),
                "updated_at": self.current_session.updated_at.isoformat(),
                "metadata": self.current_session.metadata,
            }
            handle.write(json.dumps(metadata_line, ensure_ascii=False) + "\n")
            for message in self.current_session.messages:
                handle.write(json.dumps(message, ensure_ascii=False) + "\n")

    @staticmethod
    def _sanitize_label(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9_-]+", "-", value).strip("-") or "frame"

    @staticmethod
    def _mime_extension(mime_type: str) -> str:
        return {
            "image/jpeg": ".jpg",
            "image/png": ".png",
            "image/webp": ".webp",
        }.get(mime_type, ".bin")
