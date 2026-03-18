"""Agent: core agentic loop with tool calling."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from .llm import LLMClient, LLMConfig, LLMResponse, ToolCall
from .logging_ import EventLogger, create_session_logger
from .history import SessionWriter
from .permissions import PermissionConfig, PermissionManager, PermissionDecision
from .prompts import SYSTEM_PROMPT, PLAN_MODE_PROMPT
from .tools import CustomTool, ToolExecutor, ToolResult, get_tool_schemas
from .workspace import Workspace


@dataclass
class AgentConfig:
    """Configuration for the agent."""
    max_steps: int = 30
    max_tool_calls_per_step: int = 8
    doom_loop_threshold: int = 3
    plan_mode: bool = False
    system_prompt: str | None = None
    allowed_tools: list[str] | None = None


@dataclass
class StepResult:
    """Result of a single agent step."""
    step: int
    llm_response: LLMResponse
    tool_calls: list[dict]
    tool_results: list[dict]
    status: str  # "continue", "complete", "doom_loop", "max_steps", "error"


@dataclass
class AgentResult:
    """Final result of agent run."""
    success: bool
    final_message: str
    step_count: int
    tool_call_count: int
    files_touched: list[str]
    status: str
    session_id: str


class UICallback(Protocol):
    """Protocol for UI callbacks."""

    def on_step_start(self, step: int, max_steps: int) -> None: ...
    def on_llm_response(self, response: LLMResponse) -> None: ...
    def on_tool_call(self, tool_name: str, args: dict) -> None: ...
    def on_tool_result(self, tool_name: str, result: ToolResult) -> None: ...
    def on_permission_prompt(
        self, tool_name: str, args: dict
    ) -> str: ...  # Returns: "allow", "always", "deny"
    def on_doom_loop(self, tool_name: str, args: dict, count: int) -> bool: ...  # Returns: continue?
    def on_complete(self, result: AgentResult) -> None: ...
    def on_error(self, error: str) -> None: ...


class NullUI:
    """Null UI that does nothing."""

    def on_step_start(self, step: int, max_steps: int) -> None:
        pass

    def on_llm_response(self, response: LLMResponse) -> None:
        pass

    def on_tool_call(self, tool_name: str, args: dict) -> None:
        pass

    def on_tool_result(self, tool_name: str, result: ToolResult) -> None:
        pass

    def on_permission_prompt(self, tool_name: str, args: dict) -> str:
        return "deny"  # Deny by default if no UI

    def on_doom_loop(self, tool_name: str, args: dict, count: int) -> bool:
        return False  # Don't continue

    def on_complete(self, result: AgentResult) -> None:
        pass

    def on_error(self, error: str) -> None:
        pass


class Agent:
    """The core agentic loop."""

    def __init__(
        self,
        workspace_root: str | Path,
        llm_client: LLMClient,
        config: AgentConfig | None = None,
        permission_config: PermissionConfig | None = None,
        ui: UICallback | None = None,
        custom_tools: list[CustomTool] | None = None,
    ):
        self.workspace = Workspace(workspace_root)
        self.llm_client = llm_client
        self.config = config or AgentConfig()
        self.permission_config = permission_config or PermissionConfig.load(self.workspace.root)
        self.ui = ui or NullUI()
        self.custom_tools = custom_tools or []

        self.tool_executor = ToolExecutor(self.workspace, custom_tools=self.custom_tools)

        # Set up permission manager with prompt callback
        self.permission_manager = PermissionManager(
            self.permission_config,
            prompt_callback=self._handle_permission_prompt,
        )

        # Session logging
        self.logger, self.session_id = create_session_logger(self.workspace.root)
        self.session_writer = SessionWriter(self.logger.session_dir, self.session_id)

        # Doom loop tracking: list of (tool_name, args_json) for recent calls
        self._recent_calls: list[tuple[str, str]] = []

        # Track files touched
        self._files_touched: set[str] = set()

    def _handle_permission_prompt(
        self, tool_name: str, cache_key: str, args: dict
    ) -> PermissionDecision:
        """Handle permission prompt via UI."""
        response = self.ui.on_permission_prompt(tool_name, args)

        if response == "allow":
            return PermissionDecision(True, "User allowed once", "user_prompt")
        elif response == "always":
            self.permission_manager.allow_always(cache_key)
            return PermissionDecision(True, "User allowed always", "user_prompt")
        else:
            return PermissionDecision(False, "User denied", "user_prompt")

    def _check_doom_loop(self, tool_name: str, args: dict) -> bool:
        """Check if we're in a doom loop. Returns True if should stop."""
        args_json = json.dumps(args, sort_keys=True)
        call_key = (tool_name, args_json)

        # Count consecutive identical calls
        consecutive = 0
        for recent in reversed(self._recent_calls):
            if recent == call_key:
                consecutive += 1
            else:
                break

        if consecutive >= self.config.doom_loop_threshold - 1:
            # This would be the Nth identical call
            self.logger.log_doom_loop(tool_name, args, consecutive + 1)
            return True

        self._recent_calls.append(call_key)
        return False

    def _build_messages(self, request: str) -> list[dict]:
        """Build initial message list."""
        if self.config.system_prompt:
            system_prompt = self.config.system_prompt
        elif self.config.plan_mode:
            system_prompt = PLAN_MODE_PROMPT
        else:
            system_prompt = SYSTEM_PROMPT

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request},
        ]

    def _get_tools(self) -> list[dict]:
        """Get tool schemas, filtered for plan mode / allowed_tools."""
        tools = get_tool_schemas(custom_tools=self.custom_tools)

        if self.config.plan_mode:
            write_tools = {"write", "patch", "bash"}
            tools = [t for t in tools if t["function"]["name"] not in write_tools]

        if self.config.allowed_tools is not None:
            allowed = set(self.config.allowed_tools)
            tools = [t for t in tools if t["function"]["name"] in allowed]

        return tools

    def run(self, request: str) -> AgentResult:
        """Run the agent loop."""
        messages = self._build_messages(request)
        tools = self._get_tools()

        # Log user request
        self.logger.log("user_request", {"request": request})

        step = 0
        total_tool_calls = 0
        final_message = ""
        status = "unknown"

        try:
            while step < self.config.max_steps:
                step += 1
                self.logger.log_step_start(step)
                self.ui.on_step_start(step, self.config.max_steps)

                # Call LLM
                self.logger.log_llm_request(
                    model=self.llm_client.config.model,
                    base_url=self.llm_client.config.base_url,
                    message_count=len(messages),
                    tool_count=len(tools),
                )

                try:
                    response = self.llm_client.complete(messages, tools)
                except Exception as e:
                    self.logger.log_error("llm_error", str(e))
                    self.ui.on_error(f"LLM error: {e}")
                    status = "error"
                    final_message = f"LLM error: {e}"
                    break

                self.logger.log_llm_response(
                    has_tool_calls=response.has_tool_calls,
                    tool_call_count=len(response.tool_calls),
                    content_length=len(response.content or ""),
                    usage=response.usage,
                    finish_reason=response.finish_reason,
                )

                self.ui.on_llm_response(response)

                # Check if done (no tool calls)
                if not response.has_tool_calls:
                    final_message = response.content or ""
                    status = "complete"
                    self.logger.log_step_end(step, "complete")
                    break

                # Process tool calls
                tool_messages = []
                step_tool_calls = 0

                for tool_call in response.tool_calls:
                    if step_tool_calls >= self.config.max_tool_calls_per_step:
                        self.logger.log_error(
                            "max_tool_calls",
                            f"Exceeded {self.config.max_tool_calls_per_step} tool calls per step",
                        )
                        break

                    tool_name = tool_call.name
                    tool_args = tool_call.arguments
                    tool_id = tool_call.id

                    # Log tool call
                    self.logger.log_tool_call(tool_name, tool_id, tool_args)
                    self.ui.on_tool_call(tool_name, tool_args)

                    # Check doom loop
                    if self._check_doom_loop(tool_name, tool_args):
                        should_continue = self.ui.on_doom_loop(
                            tool_name, tool_args, self.config.doom_loop_threshold
                        )
                        if not should_continue:
                            status = "doom_loop"
                            final_message = (
                                f"Stopped: Detected doom loop - {tool_name} called "
                                f"{self.config.doom_loop_threshold} times with identical arguments. "
                                "This usually indicates the tool isn't producing the expected results. "
                                "Please review and try a different approach."
                            )
                            self.logger.log_step_end(step, "doom_loop")
                            break

                    # Check permissions
                    permission = self.permission_manager.check_tool(tool_name, tool_args)
                    self.logger.log_permission_decision(
                        tool_name, permission.allowed, permission.reason, permission.rule
                    )

                    if not permission.allowed:
                        tool_result = ToolResult(
                            success=False,
                            data={"error": f"Permission denied: {permission.reason}"},
                            duration_ms=0,
                        )
                    else:
                        # Execute tool
                        tool_result = self.tool_executor.execute(tool_name, tool_args)

                        # Track files touched
                        if tool_name in ("write", "edit"):
                            if "path" in tool_args:
                                self._files_touched.add(tool_args["path"])

                    # Log result
                    result_json = tool_result.to_json()
                    self.logger.log_tool_result(
                        tool_name,
                        tool_id,
                        tool_result.success,
                        tool_result.duration_ms,
                        tool_result.truncated,
                        result_json[:500],
                    )

                    self.ui.on_tool_result(tool_name, tool_result)

                    # Add tool result message
                    tool_messages.append({
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "content": result_json,
                    })

                    step_tool_calls += 1
                    total_tool_calls += 1

                # Check if we hit doom loop
                if status == "doom_loop":
                    break

                # Add assistant message with tool calls
                assistant_message: dict[str, Any] = {"role": "assistant"}
                if response.content:
                    assistant_message["content"] = response.content
                if response.tool_calls:
                    assistant_message["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in response.tool_calls
                    ]

                messages.append(assistant_message)
                messages.extend(tool_messages)

                self.logger.log_step_end(step, "continue")

            else:
                # Hit max steps
                status = "max_steps"
                final_message = f"Stopped: Reached maximum steps ({self.config.max_steps})"
                self.logger.log_error("max_steps", final_message)

        except Exception as e:
            status = "error"
            final_message = f"Agent error: {e}"
            self.logger.log_error("agent_error", str(e))
            self.ui.on_error(final_message)

        # Build result
        result = AgentResult(
            success=status == "complete",
            final_message=final_message,
            step_count=step,
            tool_call_count=total_tool_calls,
            files_touched=list(self._files_touched),
            status=status,
            session_id=self.session_id,
        )

        # Save session data
        self.session_writer.save_messages(messages)
        self.session_writer.save_summary(
            request=request,
            step_count=step,
            tool_calls=total_tool_calls,
            final_status=status,
            files_touched=list(self._files_touched),
        )

        self.ui.on_complete(result)
        return result
