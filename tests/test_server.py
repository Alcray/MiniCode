"""Tests for the FastAPI REST API server."""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from minicode.agent import AgentResult
from minicode.llm import LLMResponse, MockLLMClient
from minicode.server import create_app


@pytest.fixture
def client(tmp_path):
    """Return a TestClient with the FastAPI app."""
    app = create_app()
    return TestClient(app)


@pytest.fixture
def workspace(tmp_path):
    """Return a temporary workspace directory path."""
    return str(tmp_path)


def _make_mock_result(success=True, status="complete"):
    return AgentResult(
        success=success,
        final_message="Done.",
        step_count=1,
        tool_call_count=0,
        files_touched=[],
        status=status,
        session_id="test-session-id",
    )


# ---------------------------------------------------------------------------
# POST /run
# ---------------------------------------------------------------------------


class TestRunEndpoint:
    def test_run_success(self, client, workspace):
        """POST /run returns a RunResponse on success."""
        mock_result = _make_mock_result()

        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            with patch("minicode.server.Agent") as MockAgent:
                mock_agent_instance = MockAgent.return_value
                mock_agent_instance.run.return_value = mock_result

                with patch("minicode.server.LLMClient"):
                    response = client.post(
                        "/run",
                        json={"request": "say hello", "root": workspace},
                    )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["status"] == "complete"
        assert data["session_id"] == "test-session-id"
        assert "final_message" in data
        assert "step_count" in data
        assert "tool_call_count" in data
        assert "files_touched" in data

    def test_run_no_api_key(self, client, workspace):
        """POST /run returns 500 when no API key is set."""
        env_clean = {k: v for k, v in os.environ.items() if "API_KEY" not in k}
        with patch.dict(os.environ, env_clean, clear=True):
            response = client.post(
                "/run",
                json={"request": "say hello", "root": workspace},
            )

        assert response.status_code == 500
        assert "API key" in response.json()["detail"]

    def test_run_missing_request_field(self, client, workspace):
        """POST /run returns 422 when 'request' field is missing."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            response = client.post("/run", json={"root": workspace})

        assert response.status_code == 422

    def test_run_bad_workspace(self, client, tmp_path):
        """POST /run returns 400 when workspace root does not exist."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            response = client.post(
                "/run",
                json={"request": "say hello", "root": "/nonexistent/path/xyz"},
            )

        assert response.status_code == 400
        assert "Workspace root" in response.json()["detail"]

    def test_run_plan_mode(self, client, workspace):
        """POST /run with plan_mode=true runs in plan mode."""
        mock_result = _make_mock_result()

        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            with patch("minicode.server.Agent") as MockAgent:
                mock_agent_instance = MockAgent.return_value
                mock_agent_instance.run.return_value = mock_result
                mock_agent_instance.permission_config = None
                mock_agent_instance._handle_permission_prompt = None

                with patch("minicode.server.LLMClient"):
                    with patch("minicode.permissions.PlanModePermissions"):
                        response = client.post(
                            "/run",
                            json={
                                "request": "analyse code",
                                "root": workspace,
                                "plan_mode": True,
                            },
                        )

        assert response.status_code == 200

    def test_run_with_model_override(self, client, workspace):
        """POST /run accepts a model override."""
        mock_result = _make_mock_result()

        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            with patch("minicode.server.Agent") as MockAgent:
                mock_agent_instance = MockAgent.return_value
                mock_agent_instance.run.return_value = mock_result

                with patch("minicode.server.LLMClient") as MockLLM:
                    response = client.post(
                        "/run",
                        json={
                            "request": "say hello",
                            "root": workspace,
                            "model": "gpt-4o-mini",
                        },
                    )

        assert response.status_code == 200


# ---------------------------------------------------------------------------
# GET /sessions
# ---------------------------------------------------------------------------


class TestSessionsEndpoint:
    def test_list_sessions_empty(self, client, workspace):
        """GET /sessions returns empty list when no sessions exist."""
        response = client.get("/sessions", params={"root": workspace})

        assert response.status_code == 200
        assert response.json() == []

    def test_list_sessions_with_data(self, client, workspace):
        """GET /sessions returns sessions after a run."""
        from minicode.history import SessionSummary as HistorySummary

        mock_sessions = [
            HistorySummary(
                session_id="abc123",
                created_at="2024-01-01T00:00:00",
                step_count=2,
                tool_calls=3,
                final_status="complete",
                request_preview="say hello",
            )
        ]

        with patch("minicode.server.HistoryManager") as MockHistory:
            MockHistory.return_value.list_sessions.return_value = mock_sessions
            response = client.get("/sessions", params={"root": workspace})

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["session_id"] == "abc123"
        assert data[0]["final_status"] == "complete"


# ---------------------------------------------------------------------------
# GET /sessions/{session_id}
# ---------------------------------------------------------------------------


class TestSessionDetailEndpoint:
    def test_get_session_not_found(self, client, workspace):
        """GET /sessions/{id} returns 404 when session does not exist."""
        response = client.get(
            "/sessions/nonexistent-session",
            params={"root": workspace},
        )

        assert response.status_code == 404

    def test_get_session_found(self, client, workspace):
        """GET /sessions/{id} returns session detail when it exists."""
        from minicode.history import SessionData

        mock_session = SessionData(
            session_id="abc123",
            events=[],
            messages=[],
            summary={"final_status": "complete", "step_count": 2},
        )

        with patch("minicode.server.HistoryManager") as MockHistory:
            MockHistory.return_value.get_session.return_value = mock_session
            response = client.get(
                "/sessions/abc123",
                params={"root": workspace},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "abc123"
        assert data["summary"]["final_status"] == "complete"
