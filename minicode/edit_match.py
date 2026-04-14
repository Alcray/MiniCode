"""Edit matching strategies modeled after OpenCode, Cline, and Gemini CLI.

Implements a cascading set of replacer strategies, each a generator that yields
candidate strings found in the file content. The first unique match wins.

Strategy cascade (in order of precision):
 1. SimpleReplacer         – exact substring
 2. LineTrimmedReplacer    – trim trailing whitespace per line
 3. BlockAnchorReplacer    – first/last line anchors + Levenshtein on middles
 4. WhitespaceNormalizedReplacer – collapse all whitespace to single spaces
 5. IndentationFlexibleReplacer  – strip min indent then match
 6. EscapeNormalizedReplacer     – handle literal \\n, \\t, \\" from JSON encoding
 7. TrimmedBoundaryReplacer      – trim leading/trailing blank lines of block
 8. ContextAwareReplacer         – anchor lines + >=50% inner line similarity
 9. EllipsisReplacer             – handle ... placeholders for skipped code

Plus diagnostic helpers for error messages.
"""

from __future__ import annotations

import re
from collections.abc import Generator
from difflib import SequenceMatcher
from typing import Protocol


# ---------------------------------------------------------------------------
# Replacer protocol – every strategy is a generator yielding candidate matches
# ---------------------------------------------------------------------------

class Replacer(Protocol):
    """Each strategy yields candidates. A candidate is either:
    - str: the matched text from content (replacement uses new_string as-is)
    - tuple[str, str]: (matched text, adjusted replacement) for strategies
      that need to transform the replacement (e.g. re-indenting)
    """
    def __call__(self, content: str, find: str) -> Generator[str | tuple[str, str], None, None]: ...


# ---------------------------------------------------------------------------
# Strategy 1: Exact substring
# ---------------------------------------------------------------------------

def simple_replacer(content: str, find: str) -> Generator[str, None, None]:
    yield find


# ---------------------------------------------------------------------------
# Strategy 2: Trim trailing whitespace per line, match by sliding window
# ---------------------------------------------------------------------------

def line_trimmed_replacer(content: str, find: str) -> Generator[str, None, None]:
    original_lines = content.split("\n")
    search_lines = find.split("\n")

    if search_lines and search_lines[-1] == "":
        search_lines.pop()

    for i in range(len(original_lines) - len(search_lines) + 1):
        if all(
            original_lines[i + j].rstrip() == search_lines[j].rstrip()
            for j in range(len(search_lines))
        ):
            start = _line_offset(original_lines, i)
            end = _line_offset(original_lines, i + len(search_lines))
            # Keep trailing newline only if present in the slice
            candidate = content[start:end]
            if candidate.endswith("\n") and not find.endswith("\n"):
                candidate = candidate[:-1]
            yield candidate


# ---------------------------------------------------------------------------
# Strategy 3: Block anchor – first/last line match, Levenshtein on middles
# ---------------------------------------------------------------------------

SINGLE_CANDIDATE_THRESHOLD = 0.0
MULTI_CANDIDATE_THRESHOLD = 0.3


def _levenshtein(a: str, b: str) -> int:
    if not a or not b:
        return max(len(a), len(b))
    m, n = len(a), len(b)
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[n]


def block_anchor_replacer(content: str, find: str) -> Generator[str, None, None]:
    original_lines = content.split("\n")
    search_lines = find.split("\n")

    if search_lines and search_lines[-1] == "":
        search_lines.pop()

    if len(search_lines) < 3:
        return

    first_trimmed = search_lines[0].strip()
    last_trimmed = search_lines[-1].strip()

    candidates: list[tuple[int, int]] = []
    for i, line in enumerate(original_lines):
        if line.strip() != first_trimmed:
            continue
        for j in range(i + 2, len(original_lines)):
            if original_lines[j].strip() == last_trimmed:
                candidates.append((i, j))
                break

    if not candidates:
        return

    def _score(start: int, end: int) -> float:
        actual_len = end - start + 1
        mid_count = min(len(search_lines) - 2, actual_len - 2)
        if mid_count <= 0:
            return 1.0
        total = 0.0
        for k in range(1, mid_count + 1):
            ol = original_lines[start + k].strip()
            sl = search_lines[k].strip()
            max_len = max(len(ol), len(sl))
            if max_len == 0:
                continue
            total += (1 - _levenshtein(ol, sl) / max_len) / mid_count
        return total

    if len(candidates) == 1:
        start, end = candidates[0]
        if _score(start, end) >= SINGLE_CANDIDATE_THRESHOLD:
            yield _extract_block(content, original_lines, start, end)
        return

    best, best_sim = None, -1.0
    for c in candidates:
        s = _score(*c)
        if s > best_sim:
            best_sim, best = s, c

    if best is not None and best_sim >= MULTI_CANDIDATE_THRESHOLD:
        yield _extract_block(content, original_lines, *best)


# ---------------------------------------------------------------------------
# Strategy 4: Whitespace-normalized (collapse all runs to single space)
# ---------------------------------------------------------------------------

def whitespace_normalized_replacer(content: str, find: str) -> Generator[str, None, None]:
    def _norm(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    norm_find = _norm(find)
    lines = content.split("\n")

    # Single-line matches
    for line in lines:
        if _norm(line) == norm_find:
            yield line
        elif norm_find in _norm(line):
            words = find.strip().split()
            if words:
                pattern = r"\s+".join(re.escape(w) for w in words)
                try:
                    m = re.search(pattern, line)
                    if m:
                        yield m.group(0)
                except re.error:
                    pass

    # Multi-line matches
    find_lines = find.split("\n")
    if len(find_lines) > 1:
        for i in range(len(lines) - len(find_lines) + 1):
            block = lines[i : i + len(find_lines)]
            if _norm("\n".join(block)) == norm_find:
                yield "\n".join(block)


# ---------------------------------------------------------------------------
# Strategy 5: Indentation-flexible (strip min indent, then match)
# ---------------------------------------------------------------------------

def indentation_flexible_replacer(content: str, find: str) -> Generator[str | tuple[str, str], None, None]:
    """Yields (candidate, adjusted_replacement_placeholder) tuples.

    The second element is a sentinel -- the actual adjusted replacement is
    computed in replace() using _adjust_indentation(), because the replacer
    doesn't know new_string. We yield the indent offset string so replace()
    can apply it.
    """
    def _min_indent(text: str) -> int:
        lines = text.split("\n")
        non_empty = [l for l in lines if l.strip()]
        if not non_empty:
            return 0
        return min(len(l) - len(l.lstrip()) for l in non_empty)

    def _remove_indent(text: str, n: int) -> str:
        lines = text.split("\n")
        return "\n".join(
            l[n:] if l.strip() else l for l in lines
        )

    find_base_indent = _min_indent(find)
    norm_find = _remove_indent(find, find_base_indent)
    content_lines = content.split("\n")
    find_lines = find.split("\n")

    for i in range(len(content_lines) - len(find_lines) + 1):
        block = content_lines[i : i + len(find_lines)]
        block_text = "\n".join(block)
        block_base_indent = _min_indent(block_text)
        if _remove_indent(block_text, block_base_indent) == norm_find:
            # The offset between the file's indent and the LLM's indent
            extra = block_base_indent - find_base_indent
            yield block_text, f"__indent_offset__:{extra}"


# ---------------------------------------------------------------------------
# Strategy 6: Escape-normalized (handle literal \n, \t, \" from JSON)
# ---------------------------------------------------------------------------

_ESCAPE_MAP = {
    "n": "\n", "t": "\t", "r": "\r",
    "'": "'", '"': '"', "`": "`",
    "\\": "\\", "\n": "\n", "$": "$",
}


def _unescape(text: str) -> str:
    def _repl(m: re.Match) -> str:
        return _ESCAPE_MAP.get(m.group(1), m.group(0))
    return re.sub(r"\\([ntr'\"`\\\n$])", _repl, text)


def escape_normalized_replacer(content: str, find: str) -> Generator[str, None, None]:
    unescaped = _unescape(find)
    if unescaped == find:
        return

    if unescaped in content:
        yield unescaped

    lines = content.split("\n")
    find_lines = unescaped.split("\n")
    for i in range(len(lines) - len(find_lines) + 1):
        block = lines[i : i + len(find_lines)]
        joined = "\n".join(block)
        if _unescape(joined) == unescaped:
            yield joined


# ---------------------------------------------------------------------------
# Strategy 7: Trimmed boundary (strip leading/trailing blank lines)
# ---------------------------------------------------------------------------

def trimmed_boundary_replacer(content: str, find: str) -> Generator[str, None, None]:
    trimmed = find.strip()
    if trimmed == find:
        return

    if trimmed in content:
        yield trimmed

    lines = content.split("\n")
    find_lines = find.split("\n")
    for i in range(len(lines) - len(find_lines) + 1):
        block = "\n".join(lines[i : i + len(find_lines)])
        if block.strip() == trimmed:
            yield block


# ---------------------------------------------------------------------------
# Strategy 8: Context-aware (anchor first/last lines, >=50% inner match)
# ---------------------------------------------------------------------------

def context_aware_replacer(content: str, find: str) -> Generator[str, None, None]:
    find_lines = find.split("\n")
    if len(find_lines) < 3:
        return

    if find_lines and find_lines[-1] == "":
        find_lines.pop()

    content_lines = content.split("\n")
    first_trimmed = find_lines[0].strip()
    last_trimmed = find_lines[-1].strip()

    for i, line in enumerate(content_lines):
        if line.strip() != first_trimmed:
            continue
        for j in range(i + 2, len(content_lines)):
            if content_lines[j].strip() != last_trimmed:
                continue
            block = content_lines[i : j + 1]
            if len(block) != len(find_lines):
                break

            matching = total = 0
            for k in range(1, len(block) - 1):
                bl = block[k].strip()
                fl = find_lines[k].strip()
                if bl or fl:
                    total += 1
                    if bl == fl:
                        matching += 1

            if total == 0 or matching / total >= 0.5:
                yield "\n".join(block)
            break


# ---------------------------------------------------------------------------
# Strategy 9: Ellipsis (...) expansion
# ---------------------------------------------------------------------------

def ellipsis_replacer(content: str, find: str) -> Generator[str, None, None]:
    """Not a standard replacer -- handled specially in replace().

    Yields the *whole transformed content* if ellipsis substitution succeeds,
    wrapped in a sentinel so the caller can distinguish it.
    """
    dots_re = re.compile(r"(^\s*\.\.\.\n)", re.MULTILINE | re.DOTALL)

    pieces = re.split(dots_re, find)
    if len(pieces) == 1:
        return

    # Ellipsis is present in the search block -- this strategy is only usable
    # from the special handling in replace(), not the generic cascade.
    yield from ()


# ---------------------------------------------------------------------------
# Ellipsis handling (standalone, not part of the generator cascade)
# ---------------------------------------------------------------------------

def try_ellipsis_replace(content: str, old: str, new: str) -> str | None:
    """Handle ... placeholders that LLMs use to skip unchanged code."""
    dots_re = re.compile(r"(^\s*\.\.\.\n)", re.MULTILINE | re.DOTALL)

    old_pieces = re.split(dots_re, old)
    new_pieces = re.split(dots_re, new)

    if len(old_pieces) != len(new_pieces):
        return None
    if len(old_pieces) == 1:
        return None

    if not all(
        old_pieces[i] == new_pieces[i] for i in range(1, len(old_pieces), 2)
    ):
        return None

    old_chunks = [old_pieces[i] for i in range(0, len(old_pieces), 2)]
    new_chunks = [new_pieces[i] for i in range(0, len(new_pieces), 2)]

    result = content
    for o_chunk, n_chunk in zip(old_chunks, new_chunks):
        if not o_chunk and not n_chunk:
            continue
        if not o_chunk and n_chunk:
            if not result.endswith("\n"):
                result += "\n"
            result += n_chunk
            continue
        if result.count(o_chunk) != 1:
            return None
        result = result.replace(o_chunk, n_chunk, 1)

    return result


# ---------------------------------------------------------------------------
# The ordered cascade of all strategies
# ---------------------------------------------------------------------------

REPLACER_CASCADE: list[tuple[str, Replacer]] = [
    ("exact", simple_replacer),
    ("line_trimmed", line_trimmed_replacer),
    ("indentation_flexible", indentation_flexible_replacer),
    ("block_anchor", block_anchor_replacer),
    ("whitespace_normalized", whitespace_normalized_replacer),
    ("escape_normalized", escape_normalized_replacer),
    ("trimmed_boundary", trimmed_boundary_replacer),
    ("context_aware", context_aware_replacer),
]


def replace(
    content: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
) -> tuple[str, str] | None:
    """Try to replace old_string with new_string in content.

    Returns (new_content, strategy_name) on success, or None if no strategy
    could find a unique match.

    Uses the OpenCode-style cascade: each strategy is a generator yielding
    candidate substrings of `content` that match `old_string`. For each
    candidate, we verify it actually appears in content, and (unless
    replace_all) that it appears exactly once.

    Strategies may yield plain strings or (candidate, metadata) tuples.
    The indentation_flexible strategy yields tuples with indent offset info
    so the replacement text gets properly re-indented.
    """
    if old_string == new_string:
        return None

    # Ellipsis handling first -- if old_string contains ..., that's
    # an explicit intent to skip unchanged code, not a regular match.
    if "\n...\n" in old_string or old_string.startswith("...\n"):
        result = try_ellipsis_replace(content, old_string, new_string)
        if result is not None:
            return result, "ellipsis"

    for strategy_name, replacer in REPLACER_CASCADE:
        for item in replacer(content, old_string):
            # Unpack: strategies can yield str or (str, metadata)
            if isinstance(item, tuple):
                candidate, metadata = item
            else:
                candidate, metadata = item, None

            idx = content.find(candidate)
            if idx == -1:
                continue

            # Determine the actual replacement text
            replacement = _apply_metadata(new_string, metadata)

            if replace_all:
                return content.replace(candidate, replacement), strategy_name

            # Uniqueness check: must appear exactly once
            if content.find(candidate, idx + len(candidate)) != -1:
                continue

            new_content = (
                content[:idx] + replacement + content[idx + len(candidate) :]
            )
            return new_content, strategy_name

    # Finally, try ellipsis handling (operates on the whole content)
    result = try_ellipsis_replace(content, old_string, new_string)
    if result is not None:
        return result, "ellipsis"

    return None


def _apply_metadata(new_string: str, metadata: str | None) -> str:
    """Adjust new_string based on strategy metadata (e.g. indentation offset)."""
    if metadata is None:
        return new_string

    if metadata.startswith("__indent_offset__:"):
        offset = int(metadata.split(":")[1])
        if offset == 0:
            return new_string
        lines = new_string.split("\n")
        adjusted = []
        for line in lines:
            if line.strip() and offset > 0:
                adjusted.append(" " * offset + line)
            elif line.strip() and offset < 0:
                # Remove leading spaces (don't go negative)
                remove = min(-offset, len(line) - len(line.lstrip()))
                adjusted.append(line[remove:])
            else:
                adjusted.append(line)
        return "\n".join(adjusted)

    return new_string


# ---------------------------------------------------------------------------
# Diagnostic helpers
# ---------------------------------------------------------------------------

def find_similar_lines(
    search_text: str, file_content: str, threshold: float = 0.6
) -> str:
    """Find the most similar block in the file to the failed search text.

    Uses character-level SequenceMatcher so a single typo doesn't tank the
    similarity ratio. Returns context lines for the LLM to self-correct.
    """
    search_lines = search_text.splitlines()
    content_lines = file_content.splitlines()

    if not search_lines or not content_lines:
        return ""

    window = len(search_lines)
    if window > len(content_lines):
        return ""

    search_joined = "\n".join(search_lines)
    best_ratio = 0.0
    best_match: list[str] | None = None
    best_idx = 0

    for i in range(len(content_lines) - window + 1):
        chunk = content_lines[i : i + window]
        ratio = SequenceMatcher(None, search_joined, "\n".join(chunk)).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = chunk
            best_idx = i

    if best_ratio < threshold or best_match is None:
        return ""

    if best_match[0] == search_lines[0] and best_match[-1] == search_lines[-1]:
        return "\n".join(best_match)

    ctx = 5
    start = max(0, best_idx - ctx)
    end = min(len(content_lines), best_idx + window + ctx)
    return "\n".join(content_lines[start:end])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _line_offset(lines: list[str], line_idx: int) -> int:
    """Byte offset of the start of line_idx in the original content."""
    offset = 0
    for k in range(min(line_idx, len(lines))):
        offset += len(lines[k]) + 1  # +1 for the \n
    return offset


def _extract_block(
    content: str, lines: list[str], start: int, end: int
) -> str:
    """Extract a block from content spanning lines[start] through lines[end]."""
    s = _line_offset(lines, start)
    e = _line_offset(lines, end + 1)
    candidate = content[s:e]
    if candidate.endswith("\n"):
        candidate = candidate[:-1]
    return candidate
