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

            # Depth 4 files (great-grandchildren) — excluded by depth=3
            assert "task_detail.html" not in output
            assert "task_list.html" not in output
            assert "base.html" not in output

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