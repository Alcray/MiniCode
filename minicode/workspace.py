"""Workspace: filesystem operations + ignore rules."""

from __future__ import annotations

import difflib
import os
from pathlib import Path
from typing import Iterator

import pathspec


BINARY_EXTENSIONS = frozenset({
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z",
    ".exe", ".dll", ".so", ".dylib", ".class", ".jar",
    ".o", ".a", ".lib", ".obj", ".wasm",
    ".pyc", ".pyo",
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp",
    ".mp3", ".mp4", ".wav", ".avi", ".mov", ".mkv",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".bin", ".dat",
})

MAX_LINE_LENGTH = 2000
MAX_LINE_SUFFIX = f"... (truncated to {MAX_LINE_LENGTH} chars)"


# Always ignore these patterns (in addition to .gitignore)
BUILTIN_IGNORES = [
    ".git/",
    "node_modules/",
    "__pycache__/",
    ".pytest_cache/",
    "*.pyc",
    ".venv/",
    "venv/",
    ".env/",
    "dist/",
    "build/",
    ".minicode/",
    "*.egg-info/",
]


class Workspace:
    """Manages workspace filesystem with ignore rules."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self._spec: pathspec.PathSpec | None = None
        self._files_read: set[str] = set()  # tracks files the LLM has read

    def _load_ignore_spec(self) -> pathspec.PathSpec:
        """Load and combine ignore patterns."""
        if self._spec is not None:
            return self._spec

        patterns = list(BUILTIN_IGNORES)

        # Load .gitignore if present
        gitignore_path = self.root / ".gitignore"
        if gitignore_path.exists():
            try:
                patterns.extend(gitignore_path.read_text().splitlines())
            except Exception:
                pass

        self._spec = pathspec.PathSpec.from_lines("gitignore", patterns)
        return self._spec

    def is_ignored(self, path: str | Path) -> bool:
        """Check if path should be ignored."""
        spec = self._load_ignore_spec()
        rel_path = self._relative_path(path)
        if rel_path is None:
            return True
        return spec.match_file(str(rel_path))

    def _relative_path(self, path: str | Path) -> Path | None:
        """Get path relative to workspace root. Returns None if outside workspace."""
        try:
            abs_path = (self.root / path).resolve()
            rel = abs_path.relative_to(self.root)
            return rel
        except ValueError:
            return None

    def resolve_path(self, path: str | Path) -> Path | None:
        """Resolve path within workspace. Returns None if outside workspace."""
        rel = self._relative_path(path)
        if rel is None:
            return None
        return self.root / rel

    def list_dir(self, path: str = ".") -> list[dict]:
        """List directory contents, respecting ignore rules."""
        target = self.resolve_path(path)
        if target is None or not target.exists():
            return []

        if not target.is_dir():
            return []

        entries = []
        try:
            for entry in sorted(target.iterdir()):
                rel_path = entry.relative_to(self.root)
                if self.is_ignored(rel_path):
                    continue
                entries.append({
                    "name": entry.name,
                    "type": "dir" if entry.is_dir() else "file",
                    "path": str(rel_path),
                })
        except PermissionError:
            pass

        return entries

    def glob(self, pattern: str) -> list[str]:
        """Find files matching glob pattern, respecting ignore rules."""
        matches = []
        try:
            for match in self.root.glob(pattern):
                if match.is_file():
                    rel_path = match.relative_to(self.root)
                    if not self.is_ignored(rel_path):
                        matches.append(str(rel_path))
        except Exception:
            pass
        return sorted(matches)

    def grep(
        self, query: str, path: str = ".", glob_filter: str | None = None
    ) -> list[dict]:
        """Search for pattern in files, respecting ignore rules."""
        results = []
        target = self.resolve_path(path)
        if target is None:
            return results

        max_results = 100
        max_context_chars = 200

        def search_file(file_path: Path) -> Iterator[dict]:
            try:
                content = file_path.read_text(errors="ignore")
                lines = content.splitlines()
                for i, line in enumerate(lines, 1):
                    if query in line:
                        # Truncate context
                        idx = line.find(query)
                        start = max(0, idx - max_context_chars // 2)
                        end = min(len(line), idx + len(query) + max_context_chars // 2)
                        context = line[start:end]
                        if start > 0:
                            context = "..." + context
                        if end < len(line):
                            context = context + "..."
                        yield {
                            "file": str(file_path.relative_to(self.root)),
                            "line": i,
                            "context": context,
                        }
            except Exception:
                pass

        if target.is_file():
            files = [target]
        else:
            if glob_filter:
                files = list(target.glob(glob_filter))
            else:
                files = list(target.rglob("*"))

        for f in files:
            if not f.is_file():
                continue
            rel = f.relative_to(self.root)
            if self.is_ignored(rel):
                continue
            for match in search_file(f):
                results.append(match)
                if len(results) >= max_results:
                    return results

        return results

    def read_file(
        self, path: str, offset: int | None = None, limit: int | None = None
    ) -> dict:
        """Read file content with line number prefixes.

        Returns lines formatted as ``{line_number}: {content}`` so the LLM
        can reference exact locations when using the edit tool.

        Args:
            path: File path relative to workspace root.
            offset: 1-indexed line to start reading from (default 1).
            limit: Max lines to return (default 2000).
        """
        target = self.resolve_path(path)
        if target is None:
            return {"error": "Path outside workspace", "content": None}

        if not target.exists():
            hint = self._suggest_similar_files(path)
            msg = f"File not found: {path}"
            if hint:
                msg += f"\n\nDid you mean one of these?\n{hint}"
            return {"error": msg, "content": None}

        if not target.is_file():
            return {"error": f"Not a file: {path}", "content": None}

        if self._is_binary(target):
            return {"error": f"Cannot read binary file: {path}", "content": None}

        max_bytes = 50_000  # 50KB cap (matches OpenCode)
        read_limit = limit or 2000
        start = (offset or 1) - 1  # 1-indexed -> 0-indexed

        try:
            content = target.read_text(errors="replace")
            self._files_read.add(str(target.relative_to(self.root)))
            all_lines = content.splitlines()
            total_lines = len(all_lines)

            if start >= total_lines and not (total_lines == 0 and start == 0):
                return {
                    "error": f"Offset {start + 1} is out of range (file has {total_lines} lines)",
                    "content": None,
                }

            selected = all_lines[start : start + read_limit]

            # Format with line number prefixes and truncate long lines
            numbered: list[str] = []
            byte_count = 0
            truncated_by_bytes = False
            for i, line in enumerate(selected):
                if len(line) > MAX_LINE_LENGTH:
                    line = line[:MAX_LINE_LENGTH] + MAX_LINE_SUFFIX
                prefixed = f"{start + i + 1}: {line}"
                line_bytes = len(prefixed.encode("utf-8")) + 1
                if byte_count + line_bytes > max_bytes:
                    truncated_by_bytes = True
                    break
                numbered.append(prefixed)
                byte_count += line_bytes

            result_content = "\n".join(numbered)
            last_line = start + len(numbered)
            has_more = last_line < total_lines or truncated_by_bytes

            # Build pagination footer
            if truncated_by_bytes:
                footer = (
                    f"(Output capped at {max_bytes // 1024}KB. "
                    f"Showing lines {start + 1}-{last_line} of {total_lines}. "
                    f"Use offset={last_line + 1} to continue.)"
                )
            elif has_more:
                footer = (
                    f"(Showing lines {start + 1}-{last_line} of {total_lines}. "
                    f"Use offset={last_line + 1} to continue.)"
                )
            else:
                footer = f"(End of file - {total_lines} lines)"

            result_content += "\n\n" + footer

            return {
                "content": result_content,
                "total_lines": total_lines,
                "truncated": has_more,
                "bytes": byte_count,
            }
        except Exception as e:
            return {"error": str(e), "content": None}

    def _is_binary(self, path: Path) -> bool:
        """Check if a file is binary by extension or by sampling bytes."""
        if path.suffix.lower() in BINARY_EXTENSIONS:
            return True
        try:
            size = path.stat().st_size
            if size == 0:
                return False
            sample_size = min(4096, size)
            data = path.read_bytes()[:sample_size]
            if b"\x00" in data:
                return True
            non_printable = sum(
                1 for b in data if b < 9 or (13 < b < 32)
            )
            return non_printable / len(data) > 0.3
        except Exception:
            return False

    def _suggest_similar_files(self, path: str) -> str:
        """Search the parent directory for similarly-named files."""
        target = self.resolve_path(path)
        if target is None:
            return ""
        parent = target.parent
        if not parent.exists():
            return ""
        basename = target.name.lower()
        try:
            candidates = [
                str(e.relative_to(self.root))
                for e in parent.iterdir()
                if e.is_file()
                and (basename in e.name.lower() or e.name.lower() in basename)
            ]
            if not candidates:
                close = difflib.get_close_matches(
                    target.name,
                    [e.name for e in parent.iterdir() if e.is_file()],
                    n=3, cutoff=0.6,
                )
                candidates = [
                    str((parent / c).relative_to(self.root)) for c in close
                ]
            return "\n".join(candidates[:3])
        except Exception:
            return ""

    def write_file(self, path: str, content: str) -> dict:
        """Write content to file."""
        target = self.resolve_path(path)
        if target is None:
            return {"success": False, "error": "Path outside workspace"}

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
            rel = str(self._relative_path(path))
            self._files_read.add(rel)
            return {"success": True, "bytes": len(content), "path": rel}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def apply_patch(self, diff: str) -> dict:
        """Apply unified diff patch atomically."""
        import subprocess
        import tempfile

        # Write diff to temp file
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".patch", delete=False) as f:
                f.write(diff)
                patch_file = f.name

            # Try to apply with patch command
            result = subprocess.run(
                ["patch", "-p1", "--dry-run", "-i", patch_file],
                cwd=self.root,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                os.unlink(patch_file)
                return {
                    "success": False,
                    "error": f"Patch would fail: {result.stderr or result.stdout}",
                }

            # Actually apply
            result = subprocess.run(
                ["patch", "-p1", "-i", patch_file],
                cwd=self.root,
                capture_output=True,
                text=True,
                timeout=30,
            )

            os.unlink(patch_file)

            if result.returncode == 0:
                return {"success": True, "output": result.stdout}
            else:
                return {"success": False, "error": result.stderr or result.stdout}

        except FileNotFoundError:
            # patch command not available, try manual parsing
            return self._apply_patch_manual(diff)
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _apply_patch_manual(self, diff: str) -> dict:
        """Manually apply a simple unified diff."""
        lines = diff.splitlines()
        current_file = None
        changes = {}  # file -> list of (line_num, old_lines, new_lines)

        i = 0
        while i < len(lines):
            line = lines[i]

            # Detect file header
            if line.startswith("--- "):
                i += 1
                continue
            if line.startswith("+++ "):
                # Extract filename (remove a/ or b/ prefix if present)
                path = line[4:].strip()
                if path.startswith("b/"):
                    path = path[2:]
                current_file = path
                if current_file not in changes:
                    changes[current_file] = []
                i += 1
                continue

            # Detect hunk header
            if line.startswith("@@"):
                # Parse @@ -start,count +start,count @@
                import re
                match = re.match(r"@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@", line)
                if match and current_file:
                    old_start = int(match.group(1))
                    new_start = int(match.group(2))
                    i += 1

                    old_lines = []
                    new_lines = []

                    while i < len(lines) and not lines[i].startswith("@@") and not lines[i].startswith("--- "):
                        hunk_line = lines[i]
                        if hunk_line.startswith("-"):
                            old_lines.append(hunk_line[1:])
                        elif hunk_line.startswith("+"):
                            new_lines.append(hunk_line[1:])
                        elif hunk_line.startswith(" ") or hunk_line == "":
                            content = hunk_line[1:] if hunk_line.startswith(" ") else ""
                            old_lines.append(content)
                            new_lines.append(content)
                        i += 1

                    changes[current_file].append((old_start, old_lines, new_lines))
                    continue
            i += 1

        # Apply changes
        try:
            for filepath, hunks in changes.items():
                target = self.resolve_path(filepath)
                if target is None:
                    return {"success": False, "error": f"Path outside workspace: {filepath}"}

                if target.exists():
                    content = target.read_text().splitlines()
                else:
                    content = []

                # Apply hunks in reverse order to preserve line numbers
                for old_start, old_lines, new_lines in reversed(hunks):
                    start_idx = old_start - 1
                    # Replace old lines with new lines
                    content[start_idx : start_idx + len(old_lines)] = new_lines

                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("\n".join(content) + ("\n" if content else ""))

            return {"success": True, "files_modified": list(changes.keys())}
        except Exception as e:
            return {"success": False, "error": str(e)}
