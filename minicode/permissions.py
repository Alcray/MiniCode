"""Permissions: tool execution gating with allow/ask/deny."""

import fnmatch
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable


class PermissionLevel(Enum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


@dataclass
class PermissionDecision:
    """Result of a permission check."""
    allowed: bool
    reason: str
    rule: str | None = None


# Dangerous bash patterns (short list of obvious destructive commands)
DANGEROUS_BASH_PATTERNS = [
    r"\brm\s+(-[rf]+\s+)*[/~]",  # rm -rf /
    r"\brm\s+-[rf]*\s+\*",       # rm -rf *
    r"\bsudo\b",                  # sudo anything
    r"\bcurl\s+.*\|\s*(ba)?sh",   # curl | sh
    r"\bwget\s+.*\|\s*(ba)?sh",   # wget | sh
    r">\s*/dev/sd",               # write to disk devices
    r"\bmkfs\b",                  # format filesystem
    r"\bdd\s+.*of=/dev",          # dd to device
    r":(){.*};:",                 # fork bomb
]


@dataclass
class PermissionConfig:
    """Permission configuration for a workspace."""

    # Tool defaults: allow/ask/deny
    tool_defaults: dict[str, PermissionLevel] = field(default_factory=lambda: {
        "list": PermissionLevel.ALLOW,
        "glob": PermissionLevel.ALLOW,
        "grep": PermissionLevel.ALLOW,
        "read": PermissionLevel.ALLOW,
        "write": PermissionLevel.ASK,
        "edit": PermissionLevel.ASK,
        "bash": PermissionLevel.ASK,
    })

    # File path patterns to deny reading (except .env.example)
    deny_read_patterns: list[str] = field(default_factory=lambda: [
        ".env",
        ".env.*",
        "!.env.example",  # Exception: allow .env.example
    ])

    # File path patterns to deny writing
    deny_write_patterns: list[str] = field(default_factory=lambda: [])

    # Session-level "always allow" overrides (not persisted by default)
    session_allows: set[str] = field(default_factory=set)

    @classmethod
    def load(cls, workspace_root: Path) -> "PermissionConfig":
        """Load config from .minicode.json if present."""
        config_path = workspace_root / ".minicode.json"
        config = cls()

        if config_path.exists():
            try:
                data = json.loads(config_path.read_text())
                if "tool_defaults" in data:
                    for tool, level in data["tool_defaults"].items():
                        config.tool_defaults[tool] = PermissionLevel(level)
                if "deny_read_patterns" in data:
                    config.deny_read_patterns = data["deny_read_patterns"]
                if "deny_write_patterns" in data:
                    config.deny_write_patterns = data["deny_write_patterns"]
            except Exception:
                pass  # Use defaults on error

        return config

    def save(self, workspace_root: Path) -> None:
        """Save config to .minicode.json."""
        config_path = workspace_root / ".minicode.json"
        data = {
            "tool_defaults": {k: v.value for k, v in self.tool_defaults.items()},
            "deny_read_patterns": self.deny_read_patterns,
            "deny_write_patterns": self.deny_write_patterns,
        }
        config_path.write_text(json.dumps(data, indent=2))


class PermissionManager:
    """Manages permission checks and prompts."""

    def __init__(
        self,
        config: PermissionConfig,
        prompt_callback: Callable[[str, str, dict], PermissionDecision] | None = None,
    ):
        self.config = config
        self.prompt_callback = prompt_callback
        # Track "always allow" decisions for this session
        self._session_cache: dict[str, PermissionLevel] = {}

    def _match_patterns(self, path: str, patterns: list[str]) -> bool:
        """Check if path matches any pattern (with ! exceptions)."""
        matched = False
        for pattern in patterns:
            if pattern.startswith("!"):
                # Exception pattern
                if fnmatch.fnmatch(path, pattern[1:]):
                    return False
            else:
                if fnmatch.fnmatch(path, pattern):
                    matched = True
        return matched

    def _is_dangerous_bash(self, command: str) -> tuple[bool, str | None]:
        """Check if bash command matches dangerous patterns."""
        for pattern in DANGEROUS_BASH_PATTERNS:
            if re.search(pattern, command, re.IGNORECASE):
                return True, pattern
        return False, None

    def check_tool(
        self, tool_name: str, args: dict
    ) -> PermissionDecision:
        """Check if tool execution is allowed."""

        # Get base permission level
        base_level = self.config.tool_defaults.get(tool_name, PermissionLevel.ASK)

        # Check session cache first
        cache_key = f"{tool_name}:{json.dumps(args, sort_keys=True)}"
        if cache_key in self._session_cache:
            cached = self._session_cache[cache_key]
            if cached == PermissionLevel.ALLOW:
                return PermissionDecision(True, "Allowed (session cache)", "session_cache")

        # Tool-specific checks
        if tool_name == "read":
            path = args.get("path", "")
            filename = Path(path).name
            if self._match_patterns(filename, self.config.deny_read_patterns):
                return PermissionDecision(
                    False,
                    f"Reading '{path}' is denied by pattern rules",
                    "deny_read_patterns",
                )

        elif tool_name == "write":
            path = args.get("path", "")
            filename = Path(path).name
            if self._match_patterns(filename, self.config.deny_write_patterns):
                return PermissionDecision(
                    False,
                    f"Writing '{path}' is denied by pattern rules",
                    "deny_write_patterns",
                )

        elif tool_name == "bash":
            command = args.get("command", "")
            is_dangerous, pattern = self._is_dangerous_bash(command)
            if is_dangerous:
                return PermissionDecision(
                    False,
                    f"Dangerous bash command denied (matched: {pattern})",
                    "dangerous_bash",
                )

        # Apply base level
        if base_level == PermissionLevel.ALLOW:
            return PermissionDecision(True, "Allowed by default", "tool_default")

        if base_level == PermissionLevel.DENY:
            return PermissionDecision(False, "Denied by default", "tool_default")

        # ASK: need to prompt user
        if self.prompt_callback:
            decision = self.prompt_callback(tool_name, cache_key, args)
            return decision

        # No prompt callback, deny by default for safety
        return PermissionDecision(False, "Permission required but no prompt available", "no_prompt")

    def allow_always(self, cache_key: str) -> None:
        """Mark a tool+args combination as always allowed for this session."""
        self._session_cache[cache_key] = PermissionLevel.ALLOW

    def allow_tool_always(self, tool_name: str) -> None:
        """Mark a tool as always allowed for this session."""
        self.config.tool_defaults[tool_name] = PermissionLevel.ALLOW


class PlanModePermissions(PermissionManager):
    """Permission manager for plan mode: deny write operations."""

    def __init__(self, config: PermissionConfig, prompt_callback=None):
        super().__init__(config, prompt_callback)
        # Override write tools to deny in plan mode
        self.config.tool_defaults["write"] = PermissionLevel.DENY
        self.config.tool_defaults["edit"] = PermissionLevel.DENY
        self.config.tool_defaults["bash"] = PermissionLevel.DENY
