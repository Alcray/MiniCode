"""TUI: Terminal UI using Rich."""

import json
import sys
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from .agent import UICallback, AgentResult
from .llm import LLMResponse
from .tools import ToolResult


class RichUI(UICallback):
    """Rich-based terminal UI."""

    def __init__(self, console: Console | None = None):
        self.console = console or Console()
        self._current_step = 0
        self._max_steps = 0

    def _header(self) -> Panel:
        """Create header panel."""
        text = Text()
        text.append("MiniCode", style="bold magenta")
        text.append(f"  Step {self._current_step}/{self._max_steps}", style="dim")
        return Panel(text, border_style="dim")

    def on_step_start(self, step: int, max_steps: int) -> None:
        self._current_step = step
        self._max_steps = max_steps
        self.console.print()
        self.console.rule(f"[bold blue]Step {step}/{max_steps}[/bold blue]", style="blue")

    def on_llm_response(self, response: LLMResponse) -> None:
        if response.content:
            self.console.print()
            self.console.print(Panel(
                Markdown(response.content),
                title="[bold green]Assistant[/bold green]",
                border_style="green",
            ))

        if response.tool_calls:
            self.console.print(f"\n[dim]Tool calls: {len(response.tool_calls)}[/dim]")

    def on_tool_call(self, tool_name: str, args: dict) -> None:
        self.console.print()

        # Format arguments nicely
        args_str = json.dumps(args, indent=2)
        if len(args_str) > 500:
            args_str = args_str[:500] + "\n... (truncated)"

        self.console.print(Panel(
            Syntax(args_str, "json", theme="monokai", word_wrap=True),
            title=f"[bold yellow]Tool: {tool_name}[/bold yellow]",
            border_style="yellow",
        ))

    def on_tool_result(self, tool_name: str, result: ToolResult) -> None:
        style = "green" if result.success else "red"
        status = "✓" if result.success else "✗"

        # Format result preview
        result_str = json.dumps(result.data, indent=2)
        if len(result_str) > 1000:
            result_str = result_str[:1000] + "\n... (truncated)"

        meta = f"[dim]{result.duration_ms}ms"
        if result.truncated:
            meta += " [yellow](output truncated)[/yellow]"
        meta += "[/dim]"

        self.console.print(Panel(
            Syntax(result_str, "json", theme="monokai", word_wrap=True),
            title=f"[bold {style}]{status} Result: {tool_name}[/bold {style}]  {meta}",
            border_style=style,
        ))

    def on_permission_prompt(self, tool_name: str, args: dict) -> str:
        self.console.print()
        self.console.print(Panel(
            Text.from_markup(
                f"[bold yellow]Permission Required[/bold yellow]\n\n"
                f"Tool: [cyan]{tool_name}[/cyan]\n"
                f"Args: {json.dumps(args, indent=2)[:200]}"
            ),
            border_style="yellow",
        ))

        while True:
            choice = Prompt.ask(
                "[yellow]Allow?[/yellow]",
                choices=["y", "n", "a"],
                default="n",
            )
            if choice == "y":
                return "allow"
            elif choice == "a":
                return "always"
            elif choice == "n":
                return "deny"

    def on_doom_loop(self, tool_name: str, args: dict, count: int) -> bool:
        self.console.print()
        self.console.print(Panel(
            Text.from_markup(
                f"[bold red]⚠ Doom Loop Detected[/bold red]\n\n"
                f"Tool [cyan]{tool_name}[/cyan] has been called {count} times "
                f"with identical arguments.\n\n"
                f"This usually means the tool isn't working as expected."
            ),
            border_style="red",
        ))

        choice = Prompt.ask(
            "[red]Continue anyway?[/red]",
            choices=["y", "n"],
            default="n",
        )
        return choice == "y"

    def on_complete(self, result: AgentResult) -> None:
        self.console.print()

        # Status color
        if result.success:
            status_style = "green"
            status_text = "✓ Complete"
        elif result.status == "doom_loop":
            status_style = "red"
            status_text = "⚠ Doom Loop"
        elif result.status == "max_steps":
            status_style = "yellow"
            status_text = "⚠ Max Steps"
        else:
            status_style = "red"
            status_text = f"✗ {result.status}"

        # Summary table
        table = Table(show_header=False, box=None)
        table.add_column("Key", style="dim")
        table.add_column("Value")
        table.add_row("Status", f"[{status_style}]{status_text}[/{status_style}]")
        table.add_row("Steps", str(result.step_count))
        table.add_row("Tool Calls", str(result.tool_call_count))
        table.add_row("Session", f"[dim]{result.session_id}[/dim]")

        if result.files_touched:
            table.add_row("Files", ", ".join(result.files_touched[:5]))
            if len(result.files_touched) > 5:
                table.add_row("", f"... and {len(result.files_touched) - 5} more")

        self.console.print(Panel(
            table,
            title="[bold]Session Summary[/bold]",
            border_style=status_style,
        ))

        # Final message
        if result.final_message:
            self.console.print()
            self.console.print(Panel(
                Markdown(result.final_message),
                title="[bold]Final Response[/bold]",
                border_style="blue",
            ))

        # Session log location
        self.console.print()
        self.console.print(
            f"[dim]Session logs: .minicode/sessions/{result.session_id}/[/dim]"
        )

    def on_error(self, error: str) -> None:
        self.console.print()
        self.console.print(Panel(
            Text(error, style="red"),
            title="[bold red]Error[/bold red]",
            border_style="red",
        ))


class PlainUI(UICallback):
    """Plain text UI for --no-tui mode."""

    def __init__(self):
        pass

    def _print(self, *args, **kwargs):
        print(*args, **kwargs, flush=True)

    def on_step_start(self, step: int, max_steps: int) -> None:
        self._print(f"\n=== Step {step}/{max_steps} ===")

    def on_llm_response(self, response: LLMResponse) -> None:
        if response.content:
            self._print(f"\n[Assistant]\n{response.content}")
        if response.tool_calls:
            self._print(f"\n[Tool calls: {len(response.tool_calls)}]")

    def on_tool_call(self, tool_name: str, args: dict) -> None:
        args_str = json.dumps(args)
        if len(args_str) > 200:
            args_str = args_str[:200] + "..."
        self._print(f"  -> {tool_name}({args_str})")

    def on_tool_result(self, tool_name: str, result: ToolResult) -> None:
        status = "OK" if result.success else "FAIL"
        preview = json.dumps(result.data)
        if len(preview) > 200:
            preview = preview[:200] + "..."
        self._print(f"  <- {tool_name}: [{status}] {result.duration_ms}ms")
        self._print(f"     {preview}")

    def on_permission_prompt(self, tool_name: str, args: dict) -> str:
        self._print(f"\n[Permission Required] {tool_name}: {args}")
        self._print("Allow? [y]es / [n]o / [a]lways: ", end="")
        try:
            choice = input().strip().lower()
            if choice in ("y", "yes"):
                return "allow"
            elif choice in ("a", "always"):
                return "always"
            else:
                return "deny"
        except EOFError:
            return "deny"

    def on_doom_loop(self, tool_name: str, args: dict, count: int) -> bool:
        self._print(f"\n[DOOM LOOP] {tool_name} called {count} times with same args")
        self._print("Continue? [y/n]: ", end="")
        try:
            choice = input().strip().lower()
            return choice in ("y", "yes")
        except EOFError:
            return False

    def on_complete(self, result: AgentResult) -> None:
        self._print(f"\n=== Complete ===")
        self._print(f"Status: {result.status}")
        self._print(f"Steps: {result.step_count}")
        self._print(f"Tool Calls: {result.tool_call_count}")
        self._print(f"Session: {result.session_id}")
        if result.files_touched:
            self._print(f"Files: {', '.join(result.files_touched)}")
        if result.final_message:
            self._print(f"\n{result.final_message}")

    def on_error(self, error: str) -> None:
        self._print(f"\n[ERROR] {error}", file=sys.stderr)
