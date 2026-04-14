"""Logging: structured event logging to JSONL."""

import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class Event:
    """A logged event."""
    timestamp: str
    session_id: str
    step: int
    event_type: str
    payload: dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)


class EventLogger:
    """Append-only event logger to JSONL file."""

    def __init__(self, session_dir: Path, session_id: str):
        self.session_dir = session_dir
        self.session_id = session_id
        self.events_file = session_dir / "events.jsonl"
        self._current_step = 0

        # Ensure session directory exists
        self.session_dir.mkdir(parents=True, exist_ok=True)

    def set_step(self, step: int):
        """Set current step number."""
        self._current_step = step

    def log(self, event_type: str, payload: dict[str, Any]) -> Event:
        """Log an event. Best-effort, never raises."""
        event = Event(
            timestamp=datetime.now(timezone.utc).isoformat(),
            session_id=self.session_id,
            step=self._current_step,
            event_type=event_type,
            payload=payload,
        )

        try:
            with open(self.events_file, "a") as f:
                f.write(event.to_json() + "\n")
        except Exception:
            pass  # Best-effort logging

        return event

    def log_step_start(self, step: int):
        self.set_step(step)
        self.log("step_start", {"step": step})

    def log_step_end(self, step: int, status: str):
        self.log("step_end", {"step": step, "status": status})

    def log_llm_request(
        self,
        model: str,
        base_url: str,
        message_count: int,
        tool_count: int,
    ):
        self.log("llm_request", {
            "model": model,
            "base_url": base_url,
            "message_count": message_count,
            "tool_count": tool_count,
        })

    def log_llm_response(
        self,
        has_tool_calls: bool,
        tool_call_count: int,
        content_length: int,
        content_preview: str | None,
        reasoning_length: int,
        usage: dict[str, int],
        finish_reason: str,
    ):
        self.log("llm_response", {
            "has_tool_calls": has_tool_calls,
            "tool_call_count": tool_call_count,
            "content_length": content_length,
            "content_preview": content_preview,
            "reasoning_length": reasoning_length,
            "usage": usage,
            "finish_reason": finish_reason,
        })

    def log_tool_call(
        self,
        tool_name: str,
        tool_id: str,
        arguments: dict,
    ):
        self.log("tool_call", {
            "tool_name": tool_name,
            "tool_id": tool_id,
            "arguments": arguments,
        })

    def log_tool_result(
        self,
        tool_name: str,
        tool_id: str,
        success: bool,
        duration_ms: int,
        truncated: bool,
        result_preview: str,
    ):
        self.log("tool_result", {
            "tool_name": tool_name,
            "tool_id": tool_id,
            "success": success,
            "duration_ms": duration_ms,
            "truncated": truncated,
            "result_preview": result_preview[:500],  # Limit preview size
        })

    def log_permission_decision(
        self,
        tool_name: str,
        allowed: bool,
        reason: str,
        rule: str | None,
    ):
        self.log("permission_decision", {
            "tool_name": tool_name,
            "allowed": allowed,
            "reason": reason,
            "rule": rule,
        })

    def log_doom_loop(self, tool_name: str, arguments: dict, count: int):
        self.log("doom_loop", {
            "tool_name": tool_name,
            "arguments": arguments,
            "repeat_count": count,
        })

    def log_error(self, error_type: str, message: str):
        self.log("error", {"error_type": error_type, "message": message})


def generate_session_id() -> str:
    """Generate a session ID based on timestamp."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")[:20]


def get_sessions_dir(workspace_root: Path) -> Path:
    """Get the sessions directory for a workspace."""
    return workspace_root / ".minicode" / "sessions"


def create_session_logger(workspace_root: Path) -> tuple[EventLogger, str]:
    """Create a new session logger."""
    session_id = generate_session_id()
    sessions_dir = get_sessions_dir(workspace_root)
    session_dir = sessions_dir / session_id
    return EventLogger(session_dir, session_id), session_id
