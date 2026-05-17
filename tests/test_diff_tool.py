"""Tests for diff tool — RepoWatcher and ToolMgr.create_diff_tool."""

from __future__ import annotations

import os
import tempfile

import pytest
from git import Repo

from simple_agent.snapshot.ghost_indexer import RepoWatcher
from simple_agent.tool.tool_mgr import ToolMgr


class TestRepoWatcherGetFileDiff:
    """Tests for RepoWatcher.get_file_diff()."""

    def test_get_file_diff_returns_single_file_patch(self):
        """get_file_diff() should return a diff for only the specified file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Repo.init(tmpdir)
            file_path = os.path.join(tmpdir, "test.py")
            with open(file_path, "w") as f:
                f.write("print('hello')\n")
            repo.index.add(["test.py"])
            repo.index.commit("initial")

            watcher = RepoWatcher(tmpdir, os.path.join(tmpdir, "shadow"))
            start = watcher.take_snapshot()

            with open(file_path, "w") as f:
                f.write("print('hello')\nprint('world')\n")
            repo.index.add(["test.py"])
            repo.index.commit("second")

            end = watcher.take_snapshot()

            diff_output = watcher.get_file_diff(start, end, "test.py")

            assert "test.py" in diff_output
            assert "print('world')" in diff_output

    def test_get_file_diff_nonexistent_file_returns_empty(self):
        """get_file_diff() should return empty for nonexistent file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Repo.init(tmpdir)
            file_path = os.path.join(tmpdir, "test.py")
            with open(file_path, "w") as f:
                f.write("print('hello')\n")
            repo.index.add(["test.py"])
            repo.index.commit("initial")

            watcher = RepoWatcher(tmpdir, os.path.join(tmpdir, "shadow"))
            start = watcher.take_snapshot()
            end = watcher.take_snapshot()

            diff_output = watcher.get_file_diff(start, end, "nonexistent.py")

            assert diff_output == "" or diff_output is not None


class TestToolMgrCreateDiffTool:
    """Tests for ToolMgr.create_diff_tool()."""

    @pytest.mark.asyncio
    async def test_diff_tool_full_repo_diff(self):
        """The diff tool should return a git diff between two snapshots."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Repo.init(tmpdir)
            file_path = os.path.join(tmpdir, "test.py")
            with open(file_path, "w") as f:
                f.write("print('v1')\n")
            repo.index.add(["test.py"])
            repo.index.commit("initial")

            watcher = RepoWatcher(tmpdir, os.path.join(tmpdir, "shadow"))
            start = watcher.take_snapshot()

            with open(file_path, "w") as f:
                f.write("print('v2')\n")
            repo.index.add(["test.py"])
            repo.index.commit("second")

            end = watcher.take_snapshot()

            mgr = ToolMgr()
            diff_tool = mgr.create_diff_tool(watcher)

            result = await diff_tool.execute("call_1", {"start": start, "end": end})
            output = result.content[0].text

            assert "test.py" in output

    @pytest.mark.asyncio
    async def test_diff_tool_single_file_diff(self):
        """The diff tool with path param should return diff for a single file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Repo.init(tmpdir)
            for name in ["a.py", "b.py"]:
                file_path = os.path.join(tmpdir, name)
                with open(file_path, "w") as f:
                    f.write(f"# {name}\n")
                repo.index.add([name])
            repo.index.commit("initial")

            watcher = RepoWatcher(tmpdir, os.path.join(tmpdir, "shadow"))
            start = watcher.take_snapshot()

            for name in ["a.py", "b.py"]:
                file_path = os.path.join(tmpdir, name)
                with open(file_path, "w") as f:
                    f.write(f"# {name} modified\n")
                repo.index.add([name])
            repo.index.commit("modified both")

            end = watcher.take_snapshot()

            mgr = ToolMgr()
            diff_tool = mgr.create_diff_tool(watcher)

            result = await diff_tool.execute("call_1", {"start": start, "end": end, "path": "a.py"})
            output = result.content[0].text

            assert "a.py" in output
            assert "b.py" not in output
