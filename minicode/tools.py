"""Tools: registry and implementations for the agent."""

import json
import subprocess
import time
from dataclasses import dataclass
from typing import Any

from .workspace import Workspace


@dataclass
class ToolResult:
    """Result from a tool execution."""
    success: bool
    data: dict[str, Any]
    duration_ms: int
    truncated: bool = False

    def to_json(self) -> str:
        return json.dumps({
            "success": self.success,
            **self.data,
            "_meta": {
                "duration_ms": self.duration_ms,
                "truncated": self.truncated,
            }
        })


# OpenAI-style tool schemas
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "list",
            "description": "List directory entries at the given path. Respects ignore rules.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path relative to workspace root. Default: '.'",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "glob",
            "description": "Find files matching a glob pattern. Respects ignore rules.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern (e.g., '**/*.py', 'src/*.ts')",
                    }
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search for a pattern across files. Returns matching lines with context.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search string to find",
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory or file to search in. Default: '.'",
                    },
                    "glob_filter": {
                        "type": "string",
                        "description": "Optional glob pattern to filter files (e.g., '*.py')",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read file content. Can specify line range for large files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to workspace root",
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "First line to read (1-indexed). Optional.",
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Last line to read (1-indexed). Optional.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write",
            "description": "Create or overwrite a file with the given content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to workspace root",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit",
            "description": "Edit a file by replacing old_string with new_string. The old_string must match exactly (including whitespace). For creating new files or full rewrites, use the write tool instead.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to workspace root",
                    },
                    "old_string": {
                        "type": "string",
                        "description": "The exact text to find and replace (must be unique in the file)",
                    },
                    "new_string": {
                        "type": "string",
                        "description": "The text to replace it with",
                    },
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command in the workspace directory. Returns exit code, stdout, stderr.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to execute",
                    },
                    "timeout_ms": {
                        "type": "integer",
                        "description": "Timeout in milliseconds. Default: 30000 (30s)",
                    },
                },
                "required": ["command"],
            },
        },
    },
]


def get_tool_schemas() -> list[dict]:
    """Return OpenAI-style tool schemas."""
    return TOOL_SCHEMAS


class ToolExecutor:
    """Executes tools against a workspace."""

    MAX_OUTPUT_BYTES = 50_000  # 50KB output limit per tool

    def __init__(self, workspace: Workspace):
        self.workspace = workspace

    def execute(self, tool_name: str, args: dict) -> ToolResult:
        """Execute a tool and return result."""
        start = time.monotonic()

        try:
            if tool_name == "list":
                result = self._execute_list(args)
            elif tool_name == "glob":
                result = self._execute_glob(args)
            elif tool_name == "grep":
                result = self._execute_grep(args)
            elif tool_name == "read":
                result = self._execute_read(args)
            elif tool_name == "write":
                result = self._execute_write(args)
            elif tool_name == "edit":
                result = self._execute_edit(args)
            elif tool_name == "bash":
                result = self._execute_bash(args)
            else:
                result = {"error": f"Unknown tool: {tool_name}"}
        except Exception as e:
            result = {"error": str(e)}

        duration_ms = int((time.monotonic() - start) * 1000)

        # Check for errors
        success = "error" not in result

        # Check for truncation
        truncated = result.pop("_truncated", False)

        return ToolResult(
            success=success,
            data=result,
            duration_ms=duration_ms,
            truncated=truncated,
        )

    def _execute_list(self, args: dict) -> dict:
        path = args.get("path", ".")
        entries = self.workspace.list_dir(path)
        return {"entries": entries, "count": len(entries)}

    def _execute_glob(self, args: dict) -> dict:
        pattern = args.get("pattern", "*")
        matches = self.workspace.glob(pattern)
        return {"matches": matches, "count": len(matches)}

    def _execute_grep(self, args: dict) -> dict:
        query = args.get("query", "")
        path = args.get("path", ".")
        glob_filter = args.get("glob_filter")
        results = self.workspace.grep(query, path, glob_filter)
        return {"matches": results, "count": len(results)}

    def _execute_read(self, args: dict) -> dict:
        path = args.get("path", "")
        start_line = args.get("start_line")
        end_line = args.get("end_line")
        result = self.workspace.read_file(path, start_line, end_line)
        if result.get("truncated"):
            result["_truncated"] = True
        return result

    def _execute_write(self, args: dict) -> dict:
        path = args.get("path", "")
        content = args.get("content", "")
        return self.workspace.write_file(path, content)

    def _execute_edit(self, args: dict) -> dict:
        path = args.get("path", "")
        old_string = args.get("old_string", "")
        new_string = args.get("new_string", "")

        if not old_string:
            return {"error": "old_string cannot be empty"}

        target = self.workspace.resolve_path(path)
        if target is None:
            return {"error": "Path outside workspace"}

        if not target.exists():
            return {"error": f"File not found: {path}"}

        if not target.is_file():
            return {"error": f"Not a file: {path}"}

        try:
            content = target.read_text()

            # Check if old_string exists
            count = content.count(old_string)
            if count == 0:
                return {"error": f"old_string not found in {path}"}
            if count > 1:
                return {"error": f"old_string found {count} times in {path}. Make it more specific to match exactly once."}

            # Replace
            new_content = content.replace(old_string, new_string, 1)
            target.write_text(new_content)

            return {
                "success": True,
                "path": path,
                "replacements": 1,
            }
        except Exception as e:
            return {"error": str(e)}

    def _execute_bash(self, args: dict) -> dict:
        command = args.get("command", "")
        timeout_ms = args.get("timeout_ms", 30000)
        timeout_s = timeout_ms / 1000

        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=self.workspace.root,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )

            stdout = result.stdout
            stderr = result.stderr
            truncated = False

            # Truncate output if needed
            if len(stdout) > self.MAX_OUTPUT_BYTES:
                stdout = stdout[: self.MAX_OUTPUT_BYTES] + "\n... (truncated)"
                truncated = True
            if len(stderr) > self.MAX_OUTPUT_BYTES:
                stderr = stderr[: self.MAX_OUTPUT_BYTES] + "\n... (truncated)"
                truncated = True

            return {
                "exit_code": result.returncode,
                "stdout": stdout,
                "stderr": stderr,
                "_truncated": truncated,
            }

        except subprocess.TimeoutExpired:
            return {
                "exit_code": -1,
                "stdout": "",
                "stderr": f"Command timed out after {timeout_s}s",
                "error": "timeout",
            }
        except Exception as e:
            return {
                "exit_code": -1,
                "stdout": "",
                "stderr": str(e),
                "error": str(e),
            }
