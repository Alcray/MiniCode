"""Tests for the read and write tools (end-to-end through ToolExecutor)."""

import pytest

from minicode.tools import ToolExecutor
from minicode.workspace import Workspace


@pytest.fixture
def workspace(tmp_path):
    return Workspace(tmp_path)


@pytest.fixture
def executor(workspace):
    return ToolExecutor(workspace)


def _write_raw(workspace, name, content):
    """Write a file directly to disk (bypasses the tool)."""
    p = workspace.root / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return name


def _write_raw_bytes(workspace, name, data: bytes):
    p = workspace.root / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return name


# ===========================================================================
# Read tool
# ===========================================================================

class TestReadTool:

    # -- Line number prefixes -------------------------------------------

    def test_lines_have_number_prefixes(self, executor, workspace):
        name = _write_raw(workspace, "hello.py", "a = 1\nb = 2\nc = 3\n")
        r = executor.execute("read", {"path": name})
        assert r.success
        content = r.data["content"]
        assert "1: a = 1" in content
        assert "2: b = 2" in content
        assert "3: c = 3" in content

    def test_single_line_file(self, executor, workspace):
        name = _write_raw(workspace, "one.txt", "only line")
        r = executor.execute("read", {"path": name})
        assert r.success
        assert "1: only line" in r.data["content"]

    def test_empty_file(self, executor, workspace):
        name = _write_raw(workspace, "empty.txt", "")
        r = executor.execute("read", {"path": name})
        assert r.success
        assert "0 lines" in r.data["content"]

    # -- Offset and limit -----------------------------------------------

    def test_offset_starts_from_given_line(self, executor, workspace):
        lines = "\n".join(f"line {i}" for i in range(1, 11)) + "\n"
        name = _write_raw(workspace, "ten.txt", lines)
        r = executor.execute("read", {"path": name, "offset": 5})
        assert r.success
        content = r.data["content"]
        assert "5: line 5" in content
        assert "1: line 1" not in content

    def test_limit_caps_output(self, executor, workspace):
        lines = "\n".join(f"line {i}" for i in range(1, 101)) + "\n"
        name = _write_raw(workspace, "hundred.txt", lines)
        r = executor.execute("read", {"path": name, "limit": 10})
        assert r.success
        content = r.data["content"]
        assert "10: line 10" in content
        assert "11: line 11" not in content
        assert "offset=11" in content

    def test_offset_beyond_file_returns_error(self, executor, workspace):
        name = _write_raw(workspace, "short.txt", "a\nb\n")
        r = executor.execute("read", {"path": name, "offset": 999})
        assert not r.success
        assert "out of range" in r.data["error"]

    def test_offset_and_limit_together(self, executor, workspace):
        lines = "\n".join(f"L{i}" for i in range(1, 21)) + "\n"
        name = _write_raw(workspace, "twenty.txt", lines)
        r = executor.execute("read", {"path": name, "offset": 5, "limit": 3})
        assert r.success
        content = r.data["content"]
        assert "5: L5" in content
        assert "7: L7" in content
        assert "8: L8" not in content

    # -- Pagination footer ----------------------------------------------

    def test_footer_shows_end_of_file(self, executor, workspace):
        name = _write_raw(workspace, "small.txt", "a\nb\n")
        r = executor.execute("read", {"path": name})
        assert "End of file" in r.data["content"]

    def test_footer_shows_continuation(self, executor, workspace):
        lines = "\n".join(f"x{i}" for i in range(1, 50)) + "\n"
        name = _write_raw(workspace, "big.txt", lines)
        r = executor.execute("read", {"path": name, "limit": 10})
        assert "offset=11" in r.data["content"]

    # -- Long line truncation -------------------------------------------

    def test_long_lines_truncated(self, executor, workspace):
        long_line = "x" * 5000
        name = _write_raw(workspace, "wide.txt", long_line + "\n")
        r = executor.execute("read", {"path": name})
        assert r.success
        assert "truncated to 2000 chars" in r.data["content"]
        assert len(r.data["content"]) < 5000

    # -- Binary detection -----------------------------------------------

    def test_binary_by_extension(self, executor, workspace):
        name = _write_raw(workspace, "image.png", "not really a png")
        r = executor.execute("read", {"path": name})
        assert not r.success
        assert "binary" in r.data["error"].lower()

    def test_binary_by_null_bytes(self, executor, workspace):
        name = _write_raw_bytes(workspace, "data.custom", b"hello\x00world")
        r = executor.execute("read", {"path": name})
        assert not r.success
        assert "binary" in r.data["error"].lower()

    def test_text_file_with_odd_extension(self, executor, workspace):
        name = _write_raw(workspace, "config.xyz", "key = value\n")
        r = executor.execute("read", {"path": name})
        assert r.success
        assert "1: key = value" in r.data["content"]

    # -- File not found suggestions -------------------------------------

    def test_not_found_suggests_similar(self, executor, workspace):
        _write_raw(workspace, "config.py", "x = 1")
        r = executor.execute("read", {"path": "confg.py"})
        assert not r.success
        assert "config.py" in r.data["error"]

    def test_not_found_no_suggestions(self, executor, workspace):
        r = executor.execute("read", {"path": "totally_nonexistent_xyz.py"})
        assert not r.success
        assert "not found" in r.data["error"].lower()

    # -- Error cases ----------------------------------------------------

    def test_path_outside_workspace(self, executor, workspace):
        r = executor.execute("read", {"path": "../../etc/passwd"})
        assert not r.success

    def test_read_directory_returns_error(self, executor, workspace):
        (workspace.root / "subdir").mkdir()
        r = executor.execute("read", {"path": "subdir"})
        assert not r.success
        assert "not a file" in r.data["error"].lower()

    # -- Tracks files for write staleness check -------------------------

    def test_read_tracks_file_for_write(self, executor, workspace):
        name = _write_raw(workspace, "tracked.txt", "original\n")
        executor.execute("read", {"path": name})
        # Now write should work because we read it
        r = executor.execute("write", {"path": name, "content": "new\n"})
        assert r.success


# ===========================================================================
# Write tool
# ===========================================================================

class TestWriteTool:

    # -- Basic write ----------------------------------------------------

    def test_create_new_file(self, executor, workspace):
        r = executor.execute("write", {"path": "new.txt", "content": "hello\n"})
        assert r.success
        assert (workspace.root / "new.txt").read_text() == "hello\n"

    def test_create_with_nested_dirs(self, executor, workspace):
        r = executor.execute("write", {
            "path": "a/b/c/deep.txt", "content": "deep\n"
        })
        assert r.success
        assert (workspace.root / "a/b/c/deep.txt").read_text() == "deep\n"

    def test_overwrite_after_read(self, executor, workspace):
        name = _write_raw(workspace, "exist.txt", "old\n")
        executor.execute("read", {"path": name})
        r = executor.execute("write", {"path": name, "content": "new\n"})
        assert r.success
        assert (workspace.root / name).read_text() == "new\n"

    # -- Read-before-write enforcement ----------------------------------

    def test_overwrite_existing_file(self, executor, workspace):
        name = _write_raw(workspace, "exist.txt", "original\n")
        r = executor.execute("write", {"path": name, "content": "overwrite\n"})
        assert r.success
        assert (workspace.root / name).read_text() == "overwrite\n"

    def test_new_file_no_read_required(self, executor, workspace):
        r = executor.execute("write", {"path": "brand_new.txt", "content": "hi\n"})
        assert r.success

    # -- Error cases ----------------------------------------------------

    def test_path_outside_workspace(self, executor, workspace):
        r = executor.execute("write", {
            "path": "../../evil.txt", "content": "hack\n"
        })
        assert not r.success

    def test_returns_path_and_bytes(self, executor, workspace):
        r = executor.execute("write", {"path": "info.txt", "content": "data\n"})
        assert r.success
        assert r.data["bytes"] == 5
        assert r.data["path"] == "info.txt"


# ===========================================================================
# Integration: read -> edit -> read (full round trip)
# ===========================================================================

class TestReadEditRoundTrip:
    def test_read_edit_read_cycle(self, executor, workspace):
        """Simulate a real agent session: read a file, edit it, read it back."""
        name = _write_raw(workspace, "app.py",
            "def main():\n    print('hello')\n    return 0\n")

        # Step 1: read the file
        r1 = executor.execute("read", {"path": name})
        assert r1.success
        assert "1: def main():" in r1.data["content"]
        assert "2:     print('hello')" in r1.data["content"]

        # Step 2: edit using content from the read
        r2 = executor.execute("edit", {
            "path": name,
            "old_string": "    print('hello')",
            "new_string": "    print('goodbye')",
        })
        assert r2.success

        # Step 3: read again to verify
        r3 = executor.execute("read", {"path": name})
        assert r3.success
        assert "2:     print('goodbye')" in r3.data["content"]
        assert "hello" not in r3.data["content"]

    def test_read_write_read_cycle(self, executor, workspace):
        """Read -> write (full overwrite) -> read back."""
        name = _write_raw(workspace, "data.txt", "old content\n")

        r1 = executor.execute("read", {"path": name})
        assert r1.success

        r2 = executor.execute("write", {
            "path": name, "content": "new content\n"
        })
        assert r2.success

        r3 = executor.execute("read", {"path": name})
        assert r3.success
        assert "1: new content" in r3.data["content"]

    def test_multi_file_session(self, executor, workspace):
        """Agent reads multiple files, edits one, writes another."""
        _write_raw(workspace, "a.py", "x = 1\n")
        _write_raw(workspace, "b.py", "y = 2\n")

        executor.execute("read", {"path": "a.py"})
        executor.execute("read", {"path": "b.py"})

        r1 = executor.execute("edit", {
            "path": "a.py", "old_string": "x = 1", "new_string": "x = 10"
        })
        assert r1.success

        r2 = executor.execute("write", {
            "path": "b.py", "content": "y = 20\n"
        })
        assert r2.success

        assert (workspace.root / "a.py").read_text() == "x = 10\n"
        assert (workspace.root / "b.py").read_text() == "y = 20\n"
