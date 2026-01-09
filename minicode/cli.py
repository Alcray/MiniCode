"""CLI: Command-line interface for MiniCode."""

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from .agent import Agent, AgentConfig
from .history import HistoryManager
from .llm import LLMClient, LLMConfig
from .permissions import PermissionConfig, PlanModePermissions
from .tui import RichUI, PlainUI


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="minicode",
        description="MiniCode: Minimal agentic coding runtime",
    )

    # Main argument: the request
    parser.add_argument(
        "request",
        nargs="?",
        help="The coding task to perform",
    )

    # Workspace
    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="Workspace root directory (default: current directory)",
    )

    # LLM configuration
    parser.add_argument(
        "--model",
        type=str,
        help="Model to use (overrides MINICODE_MODEL env var)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        help="API base URL (overrides MINICODE_BASE_URL env var)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        help="Provider label (for logging only)",
    )

    # Agent configuration
    parser.add_argument(
        "--max-steps",
        type=int,
        default=30,
        help="Maximum agent steps (default: 30)",
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help="Plan mode: explore only, no modifications",
    )

    # UI
    parser.add_argument(
        "--no-tui",
        action="store_true",
        help="Disable rich TUI, use plain text output",
    )

    # History commands
    parser.add_argument(
        "--list-sessions",
        action="store_true",
        help="List recent sessions",
    )
    parser.add_argument(
        "--replay",
        type=str,
        metavar="SESSION_ID",
        help="Replay a session",
    )
    parser.add_argument(
        "--show",
        type=str,
        metavar="SESSION_ID",
        help="Show session summary",
    )

    return parser.parse_args()


def cmd_list_sessions(args: argparse.Namespace) -> int:
    """List recent sessions."""
    console = Console()
    history = HistoryManager(Path(args.root).resolve())
    sessions = history.list_sessions()

    if not sessions:
        console.print("[dim]No sessions found.[/dim]")
        return 0

    table = Table(title="Recent Sessions")
    table.add_column("Session ID", style="cyan")
    table.add_column("Created", style="dim")
    table.add_column("Steps", justify="right")
    table.add_column("Tools", justify="right")
    table.add_column("Status")
    table.add_column("Request", max_width=40)

    for s in sessions:
        status_style = "green" if s.final_status == "complete" else "yellow"
        table.add_row(
            s.session_id,
            s.created_at[:19] if s.created_at else "-",
            str(s.step_count),
            str(s.tool_calls),
            f"[{status_style}]{s.final_status}[/{status_style}]",
            s.request_preview[:40] + "..." if len(s.request_preview) > 40 else s.request_preview,
        )

    console.print(table)
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    """Replay a session."""
    console = Console()
    history = HistoryManager(Path(args.root).resolve())

    console.print(f"[bold]Replaying session: {args.replay}[/bold]\n")

    current_step = 0
    for event in history.replay_session(args.replay):
        event_type = event.get("event_type", "")
        payload = event.get("payload", {})

        if event_type == "step_start":
            current_step = payload.get("step", 0)
            console.print(f"\n[blue]━━━ Step {current_step} ━━━[/blue]")

        elif event_type == "tool_call":
            tool_name = payload.get("tool_name", "")
            tool_args = payload.get("arguments", {})
            args_str = json.dumps(tool_args)
            if len(args_str) > 100:
                args_str = args_str[:100] + "..."
            console.print(f"  [yellow]→ {tool_name}[/yellow]({args_str})")

        elif event_type == "tool_result":
            tool_name = payload.get("tool_name", "")
            success = payload.get("success", False)
            duration = payload.get("duration_ms", 0)
            status = "[green]✓[/green]" if success else "[red]✗[/red]"
            console.print(f"  {status} {tool_name} ({duration}ms)")

        elif event_type == "doom_loop":
            console.print(f"  [red]⚠ Doom loop detected![/red]")

        elif event_type == "error":
            console.print(f"  [red]Error: {payload.get('message', '')}[/red]")

    # Show final summary
    session = history.get_session(args.replay)
    if session and session.summary:
        console.print(f"\n[bold]Summary:[/bold]")
        console.print(f"  Status: {session.summary.get('final_status', 'unknown')}")
        console.print(f"  Steps: {session.summary.get('step_count', 0)}")
        console.print(f"  Tool calls: {session.summary.get('tool_calls', 0)}")

    return 0


def cmd_show(args: argparse.Namespace) -> int:
    """Show session summary."""
    console = Console()
    history = HistoryManager(Path(args.root).resolve())
    session = history.get_session(args.show)

    if session is None:
        console.print(f"[red]Session not found: {args.show}[/red]")
        return 1

    console.print(f"[bold]Session: {args.show}[/bold]\n")

    if session.summary:
        for key, value in session.summary.items():
            console.print(f"  {key}: {value}")

    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Run the agent."""
    console = Console()

    if not args.request:
        console.print("[red]Error: No request provided[/red]")
        console.print("Usage: minicode \"your request here\"")
        return 1

    # Load LLM config from env, override with CLI args
    llm_config = LLMConfig.from_env()
    if args.model:
        llm_config.model = args.model
    if args.base_url:
        llm_config.base_url = args.base_url
    if args.provider:
        llm_config.provider = args.provider

    # Validate API key
    if not llm_config.api_key:
        console.print("[red]Error: No API key configured[/red]")
        console.print("Set OPENAI_API_KEY or MINICODE_API_KEY environment variable")
        return 1

    # Agent config
    agent_config = AgentConfig(
        max_steps=args.max_steps,
        plan_mode=args.plan,
    )

    # Permissions
    workspace_root = Path(args.root).resolve()
    permission_config = PermissionConfig.load(workspace_root)

    # Use plan mode permissions if --plan
    if args.plan:
        permission_manager_class = PlanModePermissions
    else:
        permission_manager_class = None  # Use default

    # UI
    if args.no_tui:
        ui = PlainUI()
    else:
        ui = RichUI(console)

    # Create and run agent
    try:
        llm_client = LLMClient(llm_config)

        agent = Agent(
            workspace_root=workspace_root,
            llm_client=llm_client,
            config=agent_config,
            permission_config=permission_config,
            ui=ui,
        )

        # Override permission manager for plan mode
        if args.plan:
            from .permissions import PlanModePermissions
            agent.permission_manager = PlanModePermissions(
                agent.permission_config,
                prompt_callback=agent._handle_permission_prompt,
            )

        # Show startup info
        if not args.no_tui:
            console.print(f"[bold magenta]MiniCode[/bold magenta]")
            console.print(f"[dim]Model: {llm_config.model}[/dim]")
            console.print(f"[dim]Workspace: {workspace_root}[/dim]")
            if args.plan:
                console.print(f"[yellow]Plan mode: write operations disabled[/yellow]")
            console.print()

        result = agent.run(args.request)

        return 0 if result.success else 1

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]")
        return 130
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return 1
    finally:
        if "llm_client" in locals():
            llm_client.close()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Handle history commands
    if args.list_sessions:
        return cmd_list_sessions(args)
    if args.replay:
        return cmd_replay(args)
    if args.show:
        return cmd_show(args)

    # Default: run agent
    return cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
