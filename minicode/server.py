"""Server: FastAPI REST API for the MiniCode agent."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .agent import Agent, AgentConfig, AgentResult
from .history import HistoryManager
from .llm import LLMClient, LLMConfig
from .permissions import PermissionConfig


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class RunRequest(BaseModel):
    """Request body for POST /run."""

    request: str = Field(..., description="The coding task to perform")
    root: str = Field(".", description="Workspace root directory")
    max_steps: int = Field(30, description="Maximum agent steps", ge=1, le=200)
    plan_mode: bool = Field(False, description="Plan mode: explore only, no modifications")
    model: str | None = Field(None, description="LLM model override")
    base_url: str | None = Field(None, description="LLM API base URL override")
    system_prompt: str | None = Field(None, description="Custom system prompt")
    allowed_tools: list[str] | None = Field(None, description="Restrict available tools")


class RunResponse(BaseModel):
    """Response body for POST /run."""

    success: bool
    final_message: str
    step_count: int
    tool_call_count: int
    files_touched: list[str]
    status: str
    session_id: str


class SessionSummary(BaseModel):
    """Summary of a single session."""

    session_id: str
    created_at: str
    step_count: int
    tool_calls: int
    final_status: str
    request_preview: str


class SessionDetail(BaseModel):
    """Full detail for a single session."""

    session_id: str
    summary: dict[str, Any] | None


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title="MiniCode API",
        description="REST API for the MiniCode agentic coding runtime",
        version="0.1.0",
    )

    # ------------------------------------------------------------------
    # POST /run
    # ------------------------------------------------------------------

    @app.post("/run", response_model=RunResponse)
    def run_agent(body: RunRequest) -> RunResponse:
        """Run the MiniCode agent with the given request."""

        workspace_root = Path(body.root).resolve()
        if not workspace_root.is_dir():
            raise HTTPException(
                status_code=400,
                detail=f"Workspace root does not exist: {body.root}",
            )

        # Build LLM config from environment, apply optional overrides
        llm_config = LLMConfig.from_env()
        if body.model:
            llm_config.model = body.model
        if body.base_url:
            llm_config.base_url = body.base_url

        if not llm_config.api_key:
            raise HTTPException(
                status_code=500,
                detail=(
                    "No API key configured. "
                    "Set OPENAI_API_KEY or MINICODE_API_KEY environment variable."
                ),
            )

        agent_config = AgentConfig(
            max_steps=body.max_steps,
            plan_mode=body.plan_mode,
            system_prompt=body.system_prompt,
            allowed_tools=body.allowed_tools,
        )

        permission_config = PermissionConfig.load(workspace_root)

        llm_client = LLMClient(llm_config)
        try:
            agent = Agent(
                workspace_root=workspace_root,
                llm_client=llm_client,
                config=agent_config,
                permission_config=permission_config,
                # NullUI is used: permissions default to deny unless pre-configured
            )

            if body.plan_mode:
                from .permissions import PlanModePermissions

                agent.permission_manager = PlanModePermissions(
                    agent.permission_config,
                    prompt_callback=agent._handle_permission_prompt,
                )

            result: AgentResult = agent.run(body.request)
        finally:
            llm_client.close()

        return RunResponse(
            success=result.success,
            final_message=result.final_message,
            step_count=result.step_count,
            tool_call_count=result.tool_call_count,
            files_touched=result.files_touched,
            status=result.status,
            session_id=result.session_id,
        )

    # ------------------------------------------------------------------
    # GET /sessions
    # ------------------------------------------------------------------

    @app.get("/sessions", response_model=list[SessionSummary])
    def list_sessions(root: str = ".") -> list[SessionSummary]:
        """List recent agent sessions for the given workspace."""

        workspace_root = Path(root).resolve()
        history = HistoryManager(workspace_root)
        sessions = history.list_sessions()

        return [
            SessionSummary(
                session_id=s.session_id,
                created_at=s.created_at or "",
                step_count=s.step_count,
                tool_calls=s.tool_calls,
                final_status=s.final_status,
                request_preview=s.request_preview,
            )
            for s in sessions
        ]

    # ------------------------------------------------------------------
    # GET /sessions/{session_id}
    # ------------------------------------------------------------------

    @app.get("/sessions/{session_id}", response_model=SessionDetail)
    def get_session(session_id: str, root: str = ".") -> SessionDetail:
        """Get details for a specific session."""

        workspace_root = Path(root).resolve()
        history = HistoryManager(workspace_root)
        session = history.get_session(session_id)

        if session is None:
            raise HTTPException(
                status_code=404,
                detail=f"Session not found: {session_id}",
            )

        return SessionDetail(
            session_id=session_id,
            summary=session.summary,
        )

    return app


# Module-level app instance (used by uvicorn)
app = create_app()
