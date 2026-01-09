"""History: session management, storage, and replay."""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator

from .logging_ import get_sessions_dir, Event


@dataclass
class SessionSummary:
    """Summary of a session."""
    session_id: str
    created_at: str
    step_count: int
    tool_calls: int
    final_status: str
    request_preview: str


@dataclass
class SessionData:
    """Full session data."""
    session_id: str
    events: list[dict]
    messages: list[dict]
    summary: dict


class HistoryManager:
    """Manages session history."""

    def __init__(self, workspace_root: Path):
        self.workspace_root = workspace_root
        self.sessions_dir = get_sessions_dir(workspace_root)

    def list_sessions(self, limit: int = 20) -> list[SessionSummary]:
        """List recent sessions."""
        if not self.sessions_dir.exists():
            return []

        sessions = []
        session_dirs = sorted(
            self.sessions_dir.iterdir(),
            key=lambda p: p.name,
            reverse=True,
        )

        for session_dir in session_dirs[:limit]:
            if not session_dir.is_dir():
                continue

            summary = self._load_summary(session_dir)
            if summary:
                sessions.append(summary)

        return sessions

    def _load_summary(self, session_dir: Path) -> SessionSummary | None:
        """Load session summary from directory."""
        session_id = session_dir.name
        events_file = session_dir / "events.jsonl"
        summary_file = session_dir / "summary.json"

        # Try to load from summary.json first
        if summary_file.exists():
            try:
                data = json.loads(summary_file.read_text())
                return SessionSummary(
                    session_id=session_id,
                    created_at=data.get("created_at", ""),
                    step_count=data.get("step_count", 0),
                    tool_calls=data.get("tool_calls", 0),
                    final_status=data.get("final_status", "unknown"),
                    request_preview=data.get("request_preview", ""),
                )
            except Exception:
                pass

        # Fall back to scanning events
        if events_file.exists():
            try:
                step_count = 0
                tool_calls = 0
                created_at = ""
                request_preview = ""

                for line in events_file.read_text().splitlines():
                    try:
                        event = json.loads(line)
                        if not created_at:
                            created_at = event.get("timestamp", "")
                        if event.get("event_type") == "step_start":
                            step_count = max(step_count, event.get("payload", {}).get("step", 0))
                        if event.get("event_type") == "tool_call":
                            tool_calls += 1
                        if event.get("event_type") == "user_request":
                            request_preview = event.get("payload", {}).get("request", "")[:100]
                    except json.JSONDecodeError:
                        continue

                return SessionSummary(
                    session_id=session_id,
                    created_at=created_at,
                    step_count=step_count,
                    tool_calls=tool_calls,
                    final_status="unknown",
                    request_preview=request_preview,
                )
            except Exception:
                pass

        return None

    def get_session(self, session_id: str) -> SessionData | None:
        """Load full session data."""
        session_dir = self.sessions_dir / session_id
        if not session_dir.exists():
            return None

        events = []
        messages = []
        summary = {}

        # Load events
        events_file = session_dir / "events.jsonl"
        if events_file.exists():
            for line in events_file.read_text().splitlines():
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        # Load messages
        messages_file = session_dir / "messages.json"
        if messages_file.exists():
            try:
                messages = json.loads(messages_file.read_text())
            except Exception:
                pass

        # Load summary
        summary_file = session_dir / "summary.json"
        if summary_file.exists():
            try:
                summary = json.loads(summary_file.read_text())
            except Exception:
                pass

        return SessionData(
            session_id=session_id,
            events=events,
            messages=messages,
            summary=summary,
        )

    def replay_session(self, session_id: str) -> Iterator[dict]:
        """Iterate through session events for replay."""
        session = self.get_session(session_id)
        if session is None:
            return

        for event in session.events:
            yield event


class SessionWriter:
    """Writes session data at the end of a run."""

    def __init__(self, session_dir: Path, session_id: str):
        self.session_dir = session_dir
        self.session_id = session_id

    def save_messages(self, messages: list[dict]):
        """Save final messages state."""
        try:
            messages_file = self.session_dir / "messages.json"
            messages_file.write_text(json.dumps(messages, indent=2, default=str))
        except Exception:
            pass

    def save_summary(
        self,
        request: str,
        step_count: int,
        tool_calls: int,
        final_status: str,
        files_touched: list[str],
    ):
        """Save session summary."""
        try:
            summary = {
                "session_id": self.session_id,
                "created_at": datetime.now().isoformat(),
                "request_preview": request[:200],
                "step_count": step_count,
                "tool_calls": tool_calls,
                "final_status": final_status,
                "files_touched": files_touched,
            }
            summary_file = self.session_dir / "summary.json"
            summary_file.write_text(json.dumps(summary, indent=2))
        except Exception:
            pass
