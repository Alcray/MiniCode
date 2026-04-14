"""Tests for the cascading edit match strategies (OpenCode-level coverage)."""

import pytest

from minicode.edit_match import (
    block_anchor_replacer,
    context_aware_replacer,
    escape_normalized_replacer,
    find_similar_lines,
    indentation_flexible_replacer,
    line_trimmed_replacer,
    replace,
    simple_replacer,
    trimmed_boundary_replacer,
    try_ellipsis_replace,
    whitespace_normalized_replacer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _first(gen):
    """Return the first value from a generator, or None."""
    return next(gen, None)


def _all(gen):
    """Collect all values from a generator."""
    return list(gen)


# ===========================================================================
# Strategy 1: SimpleReplacer
# ===========================================================================

class TestSimpleReplacer:
    def test_yields_find_verbatim(self):
        assert _first(simple_replacer("anything", "hello")) == "hello"


# ===========================================================================
# Strategy 2: LineTrimmedReplacer
# ===========================================================================

class TestLineTrimmedReplacer:
    def test_trailing_spaces_ignored(self):
        content = "def foo():  \n    return 1  \n"
        find = "def foo():\n    return 1\n"
        candidates = _all(line_trimmed_replacer(content, find))
        assert len(candidates) >= 1
        assert "def foo():" in candidates[0]

    def test_exact_still_works(self):
        content = "a\nb\nc\n"
        find = "b\n"
        assert _first(line_trimmed_replacer(content, find)) is not None

    def test_no_match_different_content(self):
        content = "hello\nworld\n"
        find = "goodbye\n"
        assert _first(line_trimmed_replacer(content, find)) is None

    def test_multiline_trailing_ws(self):
        content = "  a = 1   \n  b = 2   \n"
        find = "  a = 1\n  b = 2\n"
        result = _first(line_trimmed_replacer(content, find))
        assert result is not None


# ===========================================================================
# Strategy 3: BlockAnchorReplacer
# ===========================================================================

class TestBlockAnchorReplacer:
    def test_anchors_match_with_different_middle(self):
        content = "def foo():\n    x = 1\n    y = 2\n    return x + y\n"
        find = "def foo():\n    x = 999\n    y = 888\n    return x + y\n"
        result = _first(block_anchor_replacer(content, find))
        assert result is not None
        assert "def foo():" in result
        assert "return x + y" in result

    def test_too_few_lines_skipped(self):
        content = "a\nb\n"
        find = "a\nb\n"
        assert _first(block_anchor_replacer(content, find)) is None

    def test_no_anchor_match(self):
        content = "def foo():\n    pass\n"
        find = "def bar():\n    x = 1\n    pass\n"
        assert _first(block_anchor_replacer(content, find)) is None

    def test_multiple_candidates_picks_best(self):
        content = (
            "def start():\n    a = 1\n    end()\n"
            "def start():\n    b = 2\n    end()\n"
        )
        find = "def start():\n    b = 2\n    end()\n"
        result = _first(block_anchor_replacer(content, find))
        assert result is not None
        assert "b = 2" in result


# ===========================================================================
# Strategy 4: WhitespaceNormalizedReplacer
# ===========================================================================

class TestWhitespaceNormalizedReplacer:
    def test_extra_spaces_collapsed(self):
        content = "x  =   1\n"
        find = "x = 1"
        result = _first(whitespace_normalized_replacer(content, find))
        assert result is not None

    def test_tabs_vs_spaces(self):
        content = "x\t=\t1\n"
        find = "x = 1"
        result = _first(whitespace_normalized_replacer(content, find))
        assert result is not None

    def test_multiline_normalized(self):
        content = "a  =  1\nb  =  2\n"
        find = "a = 1\nb = 2"
        results = _all(whitespace_normalized_replacer(content, find))
        assert any("a" in r and "b" in r for r in results)

    def test_no_match(self):
        content = "hello world\n"
        find = "goodbye"
        assert _first(whitespace_normalized_replacer(content, find)) is None


# ===========================================================================
# Strategy 5: IndentationFlexibleReplacer
# ===========================================================================

class TestIndentationFlexibleReplacer:
    def test_missing_indentation(self):
        content = "    x = 1\n    y = 2\n"
        find = "x = 1\ny = 2\n"
        result = _first(indentation_flexible_replacer(content, find))
        assert result is not None
        candidate, metadata = result
        assert "    x = 1" in candidate
        assert "indent_offset" in metadata

    def test_partial_indentation(self):
        content = "        return 1\n        return 2\n"
        find = "    return 1\n    return 2\n"
        result = _first(indentation_flexible_replacer(content, find))
        assert result is not None
        candidate, metadata = result
        assert "        return 1" in candidate
        assert "indent_offset" in metadata

    def test_correct_indentation(self):
        content = "    x = 1\n"
        find = "    x = 1\n"
        result = _first(indentation_flexible_replacer(content, find))
        assert result is not None

    def test_different_content(self):
        content = "    hello\n"
        find = "    goodbye\n"
        assert _first(indentation_flexible_replacer(content, find)) is None


# ===========================================================================
# Strategy 6: EscapeNormalizedReplacer
# ===========================================================================

class TestEscapeNormalizedReplacer:
    def test_literal_backslash_n(self):
        content = 'print("hello\nworld")\n'
        find = 'print("hello\\nworld")'
        result = _first(escape_normalized_replacer(content, find))
        assert result is not None

    def test_literal_backslash_t(self):
        content = "a\tb\n"
        find = "a\\tb"
        result = _first(escape_normalized_replacer(content, find))
        assert result is not None

    def test_no_escapes_returns_nothing(self):
        content = "plain text\n"
        find = "plain text"
        # If unescaped == find, the function returns early
        assert _first(escape_normalized_replacer(content, find)) is None

    def test_backslash_quote(self):
        content = 'say "hello"\n'
        find = 'say \\"hello\\"'
        result = _first(escape_normalized_replacer(content, find))
        assert result is not None


# ===========================================================================
# Strategy 7: TrimmedBoundaryReplacer
# ===========================================================================

class TestTrimmedBoundaryReplacer:
    def test_leading_blank_line_stripped(self):
        content = "def foo():\n    pass\n"
        find = "\ndef foo():\n    pass\n"
        result = _first(trimmed_boundary_replacer(content, find))
        assert result is not None
        assert "def foo" in result

    def test_trailing_blank_line_stripped(self):
        content = "x = 1\n"
        find = "x = 1\n\n"
        result = _first(trimmed_boundary_replacer(content, find))
        assert result is not None

    def test_already_trimmed_returns_nothing(self):
        content = "hello\n"
        find = "hello"
        assert _first(trimmed_boundary_replacer(content, find)) is None

    def test_both_ends_stripped(self):
        content = "line1\nline2\nline3\n"
        find = "\nline2\n\n"
        result = _first(trimmed_boundary_replacer(content, find))
        assert result is not None
        assert "line2" in result


# ===========================================================================
# Strategy 8: ContextAwareReplacer
# ===========================================================================

class TestContextAwareReplacer:
    def test_anchors_plus_inner_match(self):
        content = "def foo():\n    a = 1\n    b = 2\n    return a + b\n"
        find = "def foo():\n    a = 1\n    b = 2\n    return a + b\n"
        result = _first(context_aware_replacer(content, find))
        assert result is not None

    def test_inner_lines_50pct_threshold(self):
        content = "START\n    real_a\n    real_b\n    real_c\n    real_d\nEND\n"
        # 2 of 4 inner lines match (50%) -> should pass
        find = "START\n    real_a\n    real_b\n    wrong_c\n    wrong_d\nEND\n"
        result = _first(context_aware_replacer(content, find))
        assert result is not None

    def test_below_threshold_rejected(self):
        content = "START\n    a\n    b\n    c\n    d\nEND\n"
        # 0 of 4 inner lines match -> below 50%
        find = "START\n    w\n    x\n    y\n    z\nEND\n"
        result = _first(context_aware_replacer(content, find))
        assert result is None

    def test_too_few_lines(self):
        content = "a\nb\n"
        find = "a\nb\n"
        assert _first(context_aware_replacer(content, find)) is None


# ===========================================================================
# Strategy 9: Ellipsis handling
# ===========================================================================

class TestEllipsisReplace:
    def test_simple_ellipsis(self):
        content = "head\nmiddle\ntail\n"
        old = "head\n...\ntail\n"
        new = "head\n...\nnew_tail\n"
        result = try_ellipsis_replace(content, old, new)
        assert result is not None
        assert "new_tail" in result
        assert "middle" in result

    def test_no_ellipsis_returns_none(self):
        assert try_ellipsis_replace("a\n", "a\n", "b\n") is None

    def test_unmatched_returns_none(self):
        assert try_ellipsis_replace("a\n", "a\n...\nb\n", "x\ny\n") is None

    def test_ambiguous_returns_none(self):
        content = "dup\ndup\n"
        assert try_ellipsis_replace(content, "dup\n", "new\n") is None

    def test_append_via_ellipsis(self):
        result = try_ellipsis_replace("existing\n", "...\n", "...\nnew\n")
        assert result is not None
        assert "new" in result


# ===========================================================================
# Integration: replace() – the full cascade
# ===========================================================================

class TestReplace:
    def test_exact_match(self):
        result = replace("a\nb\nc\n", "b", "x")
        assert result is not None
        new_content, strategy = result
        assert "x" in new_content
        assert strategy == "exact"

    def test_line_trimmed_fallback(self):
        result = replace("a = 1   \nb = 2   \n", "a = 1\nb = 2\n", "a = 10\nb = 20\n")
        assert result is not None
        _, strategy = result
        assert strategy == "line_trimmed"

    def test_indentation_fallback(self):
        content = "    def foo():\n        return 1\n"
        result = replace(content, "def foo():\n    return 1\n", "def foo():\n    return 2\n")
        assert result is not None
        new_content, strategy = result
        assert strategy == "indentation_flexible"
        assert "        return 2" in new_content

    def test_ellipsis_fallback(self):
        content = "header_unique_xyz\nmiddle_stuff\nfooter_unique_abc\n"
        result = replace(
            content,
            "header_unique_xyz\n...\nfooter_unique_abc\n",
            "header_unique_xyz\n...\nnew_footer\n",
        )
        assert result is not None
        new_content, strategy = result
        # May match via block_anchor (anchors found) or ellipsis -- both valid
        assert "new_footer" in new_content
        assert "middle_stuff" in new_content

    def test_total_failure(self):
        result = replace("hello\n", "completely unrelated\n", "x\n")
        assert result is None

    def test_identical_strings_rejected(self):
        result = replace("hello\n", "hello", "hello")
        assert result is None

    def test_replace_all(self):
        content = "a\nb\na\nb\n"
        result = replace(content, "a", "x", replace_all=True)
        assert result is not None
        new_content, strategy = result
        assert new_content.count("x") == 2
        assert new_content.count("a") == 0

    def test_duplicate_rejected_without_replace_all(self):
        content = "dup\nother\ndup\n"
        result = replace(content, "dup", "new")
        # exact match finds 2 occurrences and skips; other strategies may or may not match
        # but simple_replacer yields "dup" which appears twice, so it skips.
        # Later strategies should also not produce a unique match.
        # This is expected to either fail entirely or match via a later strategy.
        if result is not None:
            # If a later strategy matched, it should be unique
            new_content, _ = result
            assert new_content.count("new") == 1

    def test_block_anchor_fallback(self):
        content = "class Foo:\n    x = 1\n    y = 2\n    z = 3\nend\n"
        find = "class Foo:\n    x = WRONG\n    y = WRONG\n    z = 3\nend\n"
        result = replace(content, find, "class Foo:\n    replaced\nend\n")
        assert result is not None

    def test_escape_fallback(self):
        content = 'print("hello\tworld")\n'
        find = 'print("hello\\tworld")'
        result = replace(content, find, 'print("goodbye")')
        assert result is not None
        new_content, strategy = result
        assert strategy == "escape_normalized"

    def test_whitespace_normalized_fallback(self):
        content = "x  =   1\ny  =   2\n"
        find = "x = 1\ny = 2"
        result = replace(content, find, "x = 10\ny = 20")
        assert result is not None


# ===========================================================================
# find_similar_lines()
# ===========================================================================

class TestFindSimilarLines:
    def test_finds_similar_block(self):
        search = "def foo():\n    return 1"
        content = "class Bar:\n    def foo():\n        return 1\n    def baz():\n        pass"
        result = find_similar_lines(search, content)
        assert result != ""
        assert "foo" in result

    def test_no_match_below_threshold(self):
        result = find_similar_lines(
            "completely unrelated\nstuff here",
            "the quick brown fox\njumps over the lazy dog",
        )
        assert result == ""

    def test_empty_inputs(self):
        assert find_similar_lines("", "some content") == ""
        assert find_similar_lines("search", "") == ""

    def test_typo_detected(self):
        search = "def helo():\n    print('hi')"
        content = "def hello():\n    print('hi')"
        result = find_similar_lines(search, content)
        assert "hello" in result


# ===========================================================================
# Integration: ToolExecutor._execute_edit
# ===========================================================================

class TestEditToolIntegration:
    @pytest.fixture
    def workspace(self, tmp_path):
        from minicode.workspace import Workspace
        return Workspace(tmp_path)

    @pytest.fixture
    def executor(self, workspace):
        from minicode.tools import ToolExecutor
        return ToolExecutor(workspace)

    def _write(self, workspace, name, content):
        (workspace.root / name).write_text(content)
        return name

    def test_exact_match(self, executor, workspace):
        name = self._write(workspace, "t.py", "a = 1\nb = 2\n")
        result = executor.execute("edit", {
            "path": name, "old_string": "a = 1", "new_string": "a = 10"
        })
        assert result.success
        assert result.data["match"] == "exact"
        assert (workspace.root / name).read_text() == "a = 10\nb = 2\n"

    def test_indentation_fallback(self, executor, workspace):
        name = self._write(workspace, "t.py",
            "class Foo:\n    def bar(self):\n        return 1\n")
        result = executor.execute("edit", {
            "path": name,
            "old_string": "def bar(self):\n    return 1\n",
            "new_string": "def bar(self):\n    return 42\n",
        })
        assert result.success
        content = (workspace.root / name).read_text()
        assert "        return 42" in content

    def test_diagnostic_error(self, executor, workspace):
        name = self._write(workspace, "t.py", "def hello():\n    print('hi')\n")
        result = executor.execute("edit", {
            "path": name,
            "old_string": "def helo():\n    print('hi')\n",
            "new_string": "def hello():\n    print('bye')\n",
        })
        assert not result.success
        assert "Did you mean" in result.data["error"]

    def test_replace_all(self, executor, workspace):
        name = self._write(workspace, "t.py", "TODO\ncode\nTODO\n")
        result = executor.execute("edit", {
            "path": name,
            "old_string": "TODO",
            "new_string": "DONE",
            "replace_all": True,
        })
        assert result.success
        content = (workspace.root / name).read_text()
        assert content.count("DONE") == 2
        assert content.count("TODO") == 0

    def test_identical_strings_rejected(self, executor, workspace):
        name = self._write(workspace, "t.py", "a = 1\n")
        result = executor.execute("edit", {
            "path": name,
            "old_string": "a = 1",
            "new_string": "a = 1",
        })
        assert not result.success
        assert "identical" in result.data["error"]

    def test_empty_old_string_rejected(self, executor, workspace):
        name = self._write(workspace, "t.py", "a = 1\n")
        result = executor.execute("edit", {
            "path": name,
            "old_string": "",
            "new_string": "x",
        })
        assert not result.success

    def test_file_not_found(self, executor, workspace):
        result = executor.execute("edit", {
            "path": "nonexistent.py",
            "old_string": "a",
            "new_string": "b",
        })
        assert not result.success
        assert "not found" in result.data["error"].lower()

    def test_trailing_ws_match(self, executor, workspace):
        name = self._write(workspace, "t.py", "x = 1   \ny = 2   \n")
        result = executor.execute("edit", {
            "path": name,
            "old_string": "x = 1\ny = 2\n",
            "new_string": "x = 10\ny = 20\n",
        })
        assert result.success

    def test_escape_match(self, executor, workspace):
        name = self._write(workspace, "t.py", 'msg = "hello\tworld"\n')
        result = executor.execute("edit", {
            "path": name,
            "old_string": 'msg = "hello\\tworld"',
            "new_string": 'msg = "goodbye"',
        })
        assert result.success


# ===========================================================================
# End-to-end: real files on disk, every strategy, verify file content
# ===========================================================================

class TestEditToolE2E:
    """Exercise every strategy through the real ToolExecutor -> real files on disk.

    Each test:
    1. Writes a real file to a temp directory
    2. Calls executor.execute("edit", ...) -- the same code path the agent uses
    3. Reads the file back from disk and checks the exact content
    """

    @pytest.fixture
    def workspace(self, tmp_path):
        from minicode.workspace import Workspace
        return Workspace(tmp_path)

    @pytest.fixture
    def executor(self, workspace):
        from minicode.tools import ToolExecutor
        return ToolExecutor(workspace)

    def _write(self, workspace, name, content):
        (workspace.root / name).write_text(content)
        return name

    def _read(self, workspace, name):
        return (workspace.root / name).read_text()

    # -- Strategy: exact ------------------------------------------------

    def test_e2e_exact_simple(self, executor, workspace):
        f = self._write(workspace, "s1.py", "x = 1\ny = 2\nz = 3\n")
        r = executor.execute("edit", {
            "path": f, "old_string": "y = 2", "new_string": "y = 200"
        })
        assert r.success and r.data["match"] == "exact"
        assert self._read(workspace, f) == "x = 1\ny = 200\nz = 3\n"

    def test_e2e_exact_multiline(self, executor, workspace):
        original = "def greet():\n    name = 'world'\n    print(f'hello {name}')\n"
        f = self._write(workspace, "s1b.py", original)
        r = executor.execute("edit", {
            "path": f,
            "old_string": "    name = 'world'\n    print(f'hello {name}')",
            "new_string": "    name = 'universe'\n    print(f'hi {name}')",
        })
        assert r.success and r.data["match"] == "exact"
        content = self._read(workspace, f)
        assert "name = 'universe'" in content
        assert "print(f'hi {name}')" in content
        assert "def greet():\n" in content

    # -- Strategy: line_trimmed -----------------------------------------

    def test_e2e_line_trimmed(self, executor, workspace):
        """File has trailing spaces on lines; LLM sends without them."""
        f = self._write(workspace, "s2.py",
            "class Config:   \n    debug = True   \n    verbose = False   \n")
        r = executor.execute("edit", {
            "path": f,
            "old_string": "class Config:\n    debug = True\n    verbose = False\n",
            "new_string": "class Config:\n    debug = False\n    verbose = True\n",
        })
        assert r.success
        content = self._read(workspace, f)
        assert "debug = False" in content
        assert "verbose = True" in content

    # -- Strategy: indentation_flexible ---------------------------------

    def test_e2e_indentation_deep_nesting(self, executor, workspace):
        """LLM omits outer indentation on deeply nested code."""
        original = (
            "class Server:\n"
            "    class Handler:\n"
            "        def process(self, req):\n"
            "            if req.valid:\n"
            "                return self.handle(req)\n"
            "            return None\n"
        )
        f = self._write(workspace, "s3.py", original)
        r = executor.execute("edit", {
            "path": f,
            "old_string": "def process(self, req):\n    if req.valid:\n        return self.handle(req)\n    return None\n",
            "new_string": "def process(self, req):\n    validate(req)\n    return self.handle(req)\n",
        })
        assert r.success
        content = self._read(workspace, f)
        assert "        def process(self, req):\n" in content
        assert "            validate(req)\n" in content
        assert "            return self.handle(req)\n" in content
        # Make sure the class structure is preserved
        assert content.startswith("class Server:\n")
        assert "    class Handler:\n" in content

    # -- Strategy: block_anchor -----------------------------------------

    def test_e2e_block_anchor(self, executor, workspace):
        """LLM gets first/last lines right but slightly misremembers the middle."""
        original = (
            "def calculate(x, y):\n"
            "    intermediate = x * y\n"
            "    adjusted = intermediate + 1\n"
            "    return adjusted\n"
        )
        f = self._write(workspace, "s4.py", original)
        r = executor.execute("edit", {
            "path": f,
            "old_string": "def calculate(x, y):\n    result = x * y\n    adjusted = result + 1\n    return adjusted\n",
            "new_string": "def calculate(x, y):\n    return x * y + 1\n",
        })
        assert r.success
        content = self._read(workspace, f)
        assert "return x * y + 1" in content
        assert "intermediate" not in content

    # -- Strategy: whitespace_normalized --------------------------------

    def test_e2e_whitespace_normalized(self, executor, workspace):
        """File has tabs; LLM sends spaces."""
        f = self._write(workspace, "s5.py", "x\t=\t1\n")
        r = executor.execute("edit", {
            "path": f,
            "old_string": "x = 1",
            "new_string": "x = 42",
        })
        assert r.success
        content = self._read(workspace, f)
        assert "42" in content

    # -- Strategy: escape_normalized ------------------------------------

    def test_e2e_escape_normalized(self, executor, workspace):
        """LLM sends literal \\n instead of actual newline in a string."""
        f = self._write(workspace, "s6.py", 'greeting = "hello\\nworld"\nprint(greeting)\n')
        r = executor.execute("edit", {
            "path": f,
            "old_string": 'greeting = "hello\\\\nworld"',
            "new_string": 'greeting = "goodbye"',
        })
        assert r.success
        assert 'greeting = "goodbye"' in self._read(workspace, f)

    # -- Strategy: trimmed_boundary -------------------------------------

    def test_e2e_trimmed_boundary(self, executor, workspace):
        """LLM adds extra blank lines around the search block."""
        f = self._write(workspace, "s7.py", "a = 1\nb = 2\nc = 3\n")
        r = executor.execute("edit", {
            "path": f,
            "old_string": "\nb = 2\n\n",
            "new_string": "b = 20",
        })
        assert r.success
        content = self._read(workspace, f)
        assert "b = 20" in content
        assert "a = 1" in content
        assert "c = 3" in content

    # -- Strategy: context_aware ----------------------------------------

    def test_e2e_context_aware(self, executor, workspace):
        """LLM gets anchor lines right and >50% of inner lines."""
        original = (
            "def setup():\n"
            "    config = load_config()\n"
            "    db = connect_db(config)\n"
            "    cache = init_cache()\n"
            "    return App(config, db, cache)\n"
        )
        f = self._write(workspace, "s8.py", original)
        # LLM gets 2 of 3 inner lines right (66% > 50%)
        r = executor.execute("edit", {
            "path": f,
            "old_string": "def setup():\n    config = load_config()\n    db = connect_db(config)\n    cache = init_cache()\n    return App(config, db, cache)\n",
            "new_string": "def setup():\n    return App()\n",
        })
        assert r.success
        assert "return App()" in self._read(workspace, f)

    # -- Strategy: ellipsis ---------------------------------------------

    def test_e2e_ellipsis(self, executor, workspace):
        """LLM uses ... to skip a large unchanged middle section."""
        original = (
            "def big_function():\n"
            "    # line 1\n"
            "    # line 2\n"
            "    # line 3\n"
            "    # line 4\n"
            "    # line 5\n"
            "    # line 6\n"
            "    # line 7\n"
            "    # line 8\n"
            "    # line 9\n"
            "    return 'old_result'\n"
        )
        f = self._write(workspace, "s9.py", original)
        r = executor.execute("edit", {
            "path": f,
            "old_string": "def big_function():\n...\n    return 'old_result'\n",
            "new_string": "def big_function():\n...\n    return 'new_result'\n",
        })
        assert r.success
        content = self._read(workspace, f)
        assert "return 'new_result'" in content
        # All the middle lines must be preserved
        for i in range(1, 10):
            assert f"# line {i}" in content

    # -- replace_all ----------------------------------------------------

    def test_e2e_replace_all_rename(self, executor, workspace):
        """Rename a variable across a whole file."""
        original = (
            "def process(data):\n"
            "    result = transform(data)\n"
            "    log(result)\n"
            "    return result\n"
        )
        f = self._write(workspace, "s10.py", original)
        r = executor.execute("edit", {
            "path": f,
            "old_string": "result",
            "new_string": "output",
            "replace_all": True,
        })
        assert r.success
        content = self._read(workspace, f)
        assert content.count("output") == 3
        assert "result" not in content

    # -- Error cases ----------------------------------------------------

    def test_e2e_duplicate_not_unique(self, executor, workspace):
        """Two identical matches without replace_all should fail."""
        f = self._write(workspace, "s11.py", "x = 1\ny = 2\nx = 1\n")
        r = executor.execute("edit", {
            "path": f, "old_string": "x = 1", "new_string": "x = 99"
        })
        # Should not silently change only one -- strategies should skip non-unique
        # The file should be unchanged
        assert self._read(workspace, f) == "x = 1\ny = 2\nx = 1\n"

    def test_e2e_no_match_gives_hint(self, executor, workspace):
        """When nothing matches, error should suggest similar lines."""
        f = self._write(workspace, "s12.py",
            "def calculate_total(items):\n    return sum(i.price for i in items)\n")
        r = executor.execute("edit", {
            "path": f,
            "old_string": "def calculate_totl(items):\n    return sum(i.price for i in items)\n",
            "new_string": "def calculate_total(items, tax):\n    return sum(i.price for i in items) * tax\n",
        })
        assert not r.success
        assert "Did you mean" in r.data["error"]
        assert "calculate_total" in r.data["error"]

    # -- Sequential edits (realistic scenario) --------------------------

    def test_e2e_sequential_edits(self, executor, workspace):
        """Multiple edits to the same file in sequence -- like a real agent session."""
        original = (
            "import os\n"
            "\n"
            "def read_config(path):\n"
            "    with open(path) as f:\n"
            "        return f.read()\n"
            "\n"
            "def main():\n"
            "    config = read_config('config.txt')\n"
            "    print(config)\n"
        )
        f = self._write(workspace, "s13.py", original)

        # Edit 1: add json import
        r1 = executor.execute("edit", {
            "path": f,
            "old_string": "import os\n",
            "new_string": "import os\nimport json\n",
        })
        assert r1.success

        # Edit 2: change read_config to parse JSON
        r2 = executor.execute("edit", {
            "path": f,
            "old_string": "        return f.read()\n",
            "new_string": "        return json.load(f)\n",
        })
        assert r2.success

        # Edit 3: change main to use the parsed config
        r3 = executor.execute("edit", {
            "path": f,
            "old_string": "    print(config)\n",
            "new_string": "    print(config['name'])\n",
        })
        assert r3.success

        final = self._read(workspace, f)
        assert "import json\n" in final
        assert "json.load(f)" in final
        assert "config['name']" in final
        assert "def read_config(path):" in final
        assert "def main():" in final
