"""Tests for AgentIndex and ToolMgr.create_index_tools."""

from __future__ import annotations

import os
import tempfile

import pytest

from simple_agent.index.indexer import AgentIndex
from simple_agent.tool.tool_mgr import ToolMgr


class TestAgentIndexCRUD:
    """Tests for AgentIndex update, remove, and tree operations."""

    def _make_index(self, db_path: str) -> AgentIndex:
        return AgentIndex(db_path=db_path)

    def test_update_creates_entry(self):
        """update() should create an entry and persist it."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            idx = self._make_index(db_path)
            idx.update("src/main.py", type="file", description="App entry point")

            output = idx.tree()
            assert "main.py" in output
            assert "App entry point" in output
        finally:
            os.unlink(db_path)

    def test_update_upserts_existing_entry(self):
        """update() on existing path should overwrite the description."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            idx = self._make_index(db_path)
            idx.update("src/main.py", type="file", description="Old description")
            idx.update("src/main.py", type="file", description="New description")

            output = idx.tree()
            assert "New description" in output
            assert "Old description" not in output
        finally:
            os.unlink(db_path)

    def test_remove_deletes_entry_and_children(self):
        """remove() should delete the entry and all descendants."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            idx = self._make_index(db_path)
            idx.update("src/old/", type="folder", description="Old module")
            idx.update("src/old/module.py", type="file", description="Old file")
            idx.update("src/old/module.py:old_func", type="function", description="Old function")
            idx.update("src/other.py", type="file", description="Other file")

            idx.remove("src/old/")

            output = idx.tree()
            assert "Old module" not in output
            assert "Old file" not in output
            assert "Old function" not in output
            assert "Other file" in output
        finally:
            os.unlink(db_path)

    def test_tree_shows_hierarchy(self):
        """tree() should render nested parent-child structure."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            idx = self._make_index(db_path)
            idx.update("src/__init__.py", type="file", description="Package init")
            idx.update("src/process/", type="folder", description="Process modules")
            idx.update("src/process/agent_process.py", type="file", description="Agent process")

            output = idx.tree()
            assert "src/" in output
            assert "process/" in output
            assert "agent_process.py" in output
            assert "Package init" in output
            assert "Process modules" in output
            assert "Agent process" in output
        finally:
            os.unlink(db_path)

    def test_tree_depth_limit(self):
        """tree() with depth param should limit recursion."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            idx = self._make_index(db_path)
            idx.update("src/", type="folder", description="Source")
            idx.update("src/process/", type="folder", description="Processes")
            idx.update("src/process/file.py", type="file", description="File")

            output = idx.tree(depth=1)
            assert "src/" in output
            assert "process/" not in output
            assert "file.py" not in output
        finally:
            os.unlink(db_path)

    def test_tree_filter_matches_name(self):
        """tree() with filter should show only matching names."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            idx = self._make_index(db_path)
            idx.update("src/explore.py", type="file", description="Explore module")
            idx.update("src/process.py", type="file", description="Process module")
            idx.update("src/other.py", type="file", description="Other module")

            output = idx.tree(filter="explore")
            assert "explore.py" in output
            assert "process.py" not in output
            assert "other.py" not in output
        finally:
            os.unlink(db_path)

    def test_tree_scoped_subtree(self):
        """tree() with path should render only a subtree."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            idx = self._make_index(db_path)
            idx.update("src/state/", type="folder", description="State module")
            idx.update("src/state/models.py", type="file", description="Data models")
            idx.update("tests/", type="folder", description="Test suite")

            output = idx.tree(path="src/state/")
            assert "models.py" in output
            assert "Test suite" not in output
        finally:
            os.unlink(db_path)

    def test_symbol_entry_colon_separator(self):
        """update() with colon-separated path should create symbol entry under file parent."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            idx = self._make_index(db_path)
            idx.update("src/main.py", type="file", description="Entry point")
            idx.update("src/main.py:main", type="function", description="Main function")

            output = idx.tree(path="src/main.py")
            assert "main" in output
            assert "Main function" in output
        finally:
            os.unlink(db_path)


class TestAgentIndexRealSrc:
    """Tests that walk the real src/ directory."""

    def _make_index(self, db_path: str) -> AgentIndex:
        return AgentIndex(db_path=db_path)

    def _add_src_entries(self, idx: AgentIndex, root: str, max_depth: int) -> None:
        """Walk *root* on disk and add every file/dir up to *max_depth*."""
        for dirpath, dirnames, filenames in os.walk(root):
            rel = os.path.relpath(dirpath, root)
            if rel == ".":
                rel = ""

            for name in filenames:
                entry_path = os.path.join(rel, name) if rel else name
                depth = entry_path.count(os.sep)
                if depth >= max_depth:
                    continue
                full = os.path.join(dirpath, name)
                suffix = os.path.splitext(name)[1]
                entry_type = "file"
                if suffix == ".html":
                    entry_type = "template"
                idx.update(entry_path, type=entry_type, description=full)

            for name in dirnames:
                if name.startswith("__pycache__") or name.startswith("."):
                    dirnames.remove(name)  # skip pycache / hidden in walk
                    continue
                entry_path = os.path.join(rel, name) if rel else name
                depth = entry_path.count(os.sep)
                if depth >= max_depth:
                    dirnames.remove(name)  # don't descend further
                    continue
                idx.update(entry_path + "/", type="directory", description="")

    def test_src_tree_max_depth_3(self):
        """Index the real src/ directory with max depth 3 and print the tree."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            idx = self._make_index(db_path)
            self._add_src_entries(idx, "src", max_depth=4)

            output = idx.tree(path="simple_agent", depth=4)
            # output = idx.tree(depth=3)

            print()
            print(output)

            # Directories at depth 1
            assert "simple_agent/" in output

            # Top-level files
            assert "__init__.py" in output
            assert "models.py" in output
            assert "stream.py" in output
            assert "format.py" in output

            # Depth 2 directories
            assert "process/" in output
            assert "state/" in output
            assert "index/" in output
            assert "tool/" in output
            assert "db/" in output
            assert "web/" in output
            assert "cli/" in output
            assert "session/" in output
            assert "snapshot/" in output

            # Depth 2 files
            assert "indexer.py" in output
            assert "tool_mgr.py" in output
            assert "agent_process.py" in output

            # Depth 3 directories (grandchildren of root)
            assert "templates/" in output

            # Templates files appear via parent directory entries from _ensure_parents
            assert "templates" in output

            # Render result for inspection
            print("\n===== tree output (max depth 3) =====")

        finally:
            os.unlink(db_path)

    def test_src_tree_pattern_filter(self):
        """Only show *.py files with pattern filter."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            idx = self._make_index(db_path)
            self._add_src_entries(idx, "src", max_depth=3)

            output = idx.tree(pattern="*.py", depth=3)

            assert "indexer.py" in output
            assert "tool_mgr.py" in output
            # HTML files excluded both by depth=3 (at depth 4) and by pattern
            assert "task_detail.html" not in output
            assert "base.html" not in output

            print("\n===== tree output (pattern *.py) =====")
            print(output)

        finally:
            os.unlink(db_path)


class TestDiffRangeParsing:
    """Tests for _parse_diff_ranges and _ranges_overlap."""

    def _make_index(self, db_path: str) -> AgentIndex:
        return AgentIndex(db_path=db_path)

    def test_parse_single_hunk(self):
        """Single hunk should yield (old_s, old_e, new_s, new_e)."""
        diff = "@@ -10,5 +10,6 @@\n-old\n+new\n"
        ranges = AgentIndex._parse_diff_ranges(diff)
        assert ranges == [(10, 14, 10, 15)]

    def test_parse_multiple_hunks(self):
        """Multiple hunks should yield all ranges."""
        diff = "@@ -3,4 +3,5 @@\n...\n@@ -15,3 +18,3 @@\n..."
        ranges = AgentIndex._parse_diff_ranges(diff)
        assert ranges == [(3, 6, 3, 7), (15, 17, 18, 20)]

    def test_skip_pure_add_hunk(self):
        """Hunk with old-count of 0 should be skipped."""
        diff = "@@ -5,0 +5,4 @@\n+new line\n+another\n"
        ranges = AgentIndex._parse_diff_ranges(diff)
        assert ranges == []

    def test_empty_diff(self):
        """Empty or malformed diff should return empty list."""
        assert AgentIndex._parse_diff_ranges("") == []
        assert AgentIndex._parse_diff_ranges("no hunks here") == []

    def test_ranges_overlap_full(self):
        """Full containment should overlap."""
        assert AgentIndex._ranges_overlap(10, 20, 12, 16) is True

    def test_ranges_overlap_boundary(self):
        """Shared boundary line should overlap."""
        assert AgentIndex._ranges_overlap(10, 15, 15, 18) is True

    def test_ranges_overlap_no_overlap(self):
        """Disjoint ranges should not overlap."""
        assert AgentIndex._ranges_overlap(10, 15, 16, 20) is False

    def test_ranges_overlap_adjacent(self):
        """Adjacent ranges (no shared line) should not overlap."""
        assert AgentIndex._ranges_overlap(10, 15, 16, 20) is False


class TestInvalidateStale:
    """Tests for AgentIndex.invalidate_stale()."""

    def _make_index(self, db_path: str) -> AgentIndex:
        return AgentIndex(db_path=db_path)

    def test_overlapping_entry_deleted(self):
        """Entry whose line range overlaps a hunk should be deleted."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            idx = self._make_index(db_path)
            idx.update("src/main.py", type="file", description="Main")
            idx.update("src/main.py:setup", type="function", description="Setup", line_start=9, line_end=14)
            idx.update("src/main.py:process", type="function", description="Process", line_start=15, line_end=22)

            diff = "@@ -10,3 +10,4 @@\n context\n-old\n+new\n\n"
            deleted = idx.invalidate_stale("src/main.py", diff)

            assert deleted == 1
            output = idx.tree(path="src/main.py")
            assert "setup" not in output
            assert "process" in output
        finally:
            os.unlink(db_path)

    def test_non_overlapping_kept(self):
        """Entry outside all hunk ranges should be kept."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            idx = self._make_index(db_path)
            idx.update("src/main.py", type="file", description="Main")
            idx.update("src/main.py:teardown", type="function", description="Cleanup", line_start=23, line_end=30)

            diff = "@@ -10,3 +10,4 @@\n context\n-old\n+new\n\n"
            deleted = idx.invalidate_stale("src/main.py", diff)

            assert deleted == 0
            output = idx.tree(path="src/main.py")
            assert "teardown" in output
        finally:
            os.unlink(db_path)

    def test_file_entry_survives(self):
        """File-level entry (no line range) should never be deleted."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            idx = self._make_index(db_path)
            idx.update("src/main.py", type="file", description="Main")

            diff = "@@ -1,5 +1,5 @@\n-old\n+new\n\n"
            deleted = idx.invalidate_stale("src/main.py", diff)

            assert deleted == 0
            output = idx.tree()
            assert "main.py" in output
        finally:
            os.unlink(db_path)

    def test_no_symbol_entries(self):
        """File with no symbol entries should return 0."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            idx = self._make_index(db_path)
            idx.update("src/utils.py", type="file", description="Utils")

            diff = "@@ -3,2 +3,2 @@\n-old\n+new\n\n"
            deleted = idx.invalidate_stale("src/utils.py", diff)

            assert deleted == 0
        finally:
            os.unlink(db_path)

    def test_no_hunks_returns_zero(self):
        """Empty diff should return 0 and keep all entries."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            idx = self._make_index(db_path)
            idx.update("src/main.py:func", type="function", description="Func", line_start=5, line_end=10)

            deleted = idx.invalidate_stale("src/main.py", "")
            assert deleted == 0
        finally:
            os.unlink(db_path)

    def test_multiple_hunks_one_match(self):
        """With multiple hunks, only one matching should delete the entry."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            idx = self._make_index(db_path)
            idx.update("src/main.py:setup", type="function", description="Setup", line_start=9, line_end=14)
            idx.update("src/main.py:process", type="function", description="Process", line_start=15, line_end=22)

            diff = "@@ -4,2 +4,2 @@\n-old\n+new\n\n@@ -16,3 +16,4 @@\n x\n-y\n+z\n\n"
            deleted = idx.invalidate_stale("src/main.py", diff)

            assert deleted == 1
            output = idx.tree(path="src/main.py")
            assert "setup" in output
            assert "process" not in output
        finally:
            os.unlink(db_path)


class TestUpdateWithLineRange:
    """Tests for AgentIndex.update() with line_start and line_end."""

    def _make_index(self, db_path: str) -> AgentIndex:
        return AgentIndex(db_path=db_path)

    def test_update_with_line_range(self):
        """update() should store line_start and line_end."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            idx = self._make_index(db_path)
            idx.update("src/main.py:main", type="function", description="Main", line_start=10, line_end=25)

            output = idx.tree(path="src/main.py")
            assert "main" in output
            assert "Main" in output
        finally:
            os.unlink(db_path)

    def test_update_without_line_range(self):
        """update() without line range should leave fields as None."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            idx = self._make_index(db_path)
            idx.update("src/main.py:main", type="function", description="Main")

            output = idx.tree()
            assert "main" in output
        finally:
            os.unlink(db_path)

    def test_update_overwrites_line_range(self):
        """update() on existing entry should overwrite line range."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            idx = self._make_index(db_path)
            idx.update("src/main.py:main", type="function", description="Main", line_start=10, line_end=25)
            idx.update("src/main.py:main", type="function", description="Main v2", line_start=12, line_end=28)

            # Old range [10, 25] should be overwritten — diff at [10, 11] should not match new [12, 28]
            diff = "@@ -10,2 +10,2 @@\n-old\n+new\n\n"
            deleted = idx.invalidate_stale("src/main.py", diff)
            assert deleted == 0
        finally:
            os.unlink(db_path)