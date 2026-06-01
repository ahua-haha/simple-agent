"""Tests for AgentIndex."""

from __future__ import annotations

import os
import tempfile

import pytest

from simple_agent.index.indexer import AgentIndex, IndexEntry


class TestAgentIndexCRUD:
    """Tests for AgentIndex update, remove, and tree operations."""

    @staticmethod
    def _make_index(db_path: str, base_dir: str = ".") -> AgentIndex:
        return AgentIndex(db_path, base_dir=base_dir)

    @staticmethod
    def _make_workspace(base: str, files: dict[str, str]) -> None:
        """Create files under *base*. *files* maps relpath → content."""
        for rel, content in files.items():
            full = os.path.join(base, rel)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "w") as fh:
                fh.write(content or "")

    def test_update_creates_entry(self, tmp_path):
        """update() should store a description that appears in tree()."""
        db_path = str(tmp_path / "index.db")
        ws = str(tmp_path / "ws")
        self._make_workspace(ws, {"main.py": "print('hello')"})

        idx = self._make_index(db_path, base_dir=ws)
        idx.update(path="main.py", type="file", description="App entry point")

        output = idx.tree()
        assert "main.py" in output
        assert "App entry point" in output

    def test_update_upserts_existing_entry(self, tmp_path):
        """update() on existing path should overwrite the description."""
        db_path = str(tmp_path / "index.db")
        ws = str(tmp_path / "ws")
        self._make_workspace(ws, {"main.py": "x"})

        idx = self._make_index(db_path, base_dir=ws)
        idx.update(path="main.py", type="file", description="Old description")
        idx.update(path="main.py", type="file", description="New description")

        output = idx.tree()
        assert "New description" in output
        assert "Old description" not in output

    def test_remove_clears_descriptions(self, tmp_path):
        """remove() should delete DB entries so files appear without descriptions."""
        db_path = str(tmp_path / "index.db")
        ws = str(tmp_path / "ws")
        self._make_workspace(ws, {
            "old/module.py": "x",
            "other.py": "x",
        })

        idx = self._make_index(db_path, base_dir=ws)
        idx.update(path="old", type="directory", description="Old dir")
        idx.update(path="old/module.py", type="file", description="Old file")
        idx.update(path="other.py", type="file", description="Other file")

        idx.remove("old")

        output = idx.tree()
        # Descriptions cleared
        assert "Old dir" not in output
        assert "Old file" not in output
        # Other file description preserved
        assert "Other file" in output

    def test_tree_shows_hierarchy(self, tmp_path):
        """tree() should render nested parent-child structure from filesystem."""
        db_path = str(tmp_path / "index.db")
        ws = str(tmp_path / "ws")
        self._make_workspace(ws, {
            "src/__init__.py": "",
            "src/process/agent_process.py": "",
        })

        idx = self._make_index(db_path, base_dir=ws)
        idx.update(path="src/__init__.py", type="file", description="Package init")
        idx.update(path="src/process", type="directory", description="Process modules")
        idx.update(path="src/process/agent_process.py", type="file", description="Agent process")

        output = idx.tree()
        assert "src/" in output
        assert "process/" in output
        assert "agent_process.py" in output
        assert "Package init" in output
        assert "Process modules" in output
        assert "Agent process" in output

    def test_tree_depth_limit(self, tmp_path):
        """tree() with depth param should limit recursion."""
        db_path = str(tmp_path / "index.db")
        ws = str(tmp_path / "ws")
        self._make_workspace(ws, {
            "src/process/file.py": "",
        })

        idx = self._make_index(db_path, base_dir=ws)
        idx.update(path="src", type="directory", description="Source")
        idx.update(path="src/process", type="directory", description="Processes")
        idx.update(path="src/process/file.py", type="file", description="File")

        output = idx.tree(depth=1)
        # depth=1: root + direct children (src/); grandchildren hidden
        assert "src/" in output
        assert "process/" not in output
        assert "file.py" not in output

    def test_tree_scoped_subtree(self, tmp_path):
        """tree() with path should render only a subtree."""
        db_path = str(tmp_path / "index.db")
        ws = str(tmp_path / "ws")
        self._make_workspace(ws, {
            "src/state/models.py": "",
            "tests/conftest.py": "",
        })

        idx = self._make_index(db_path, base_dir=ws)
        idx.update(path="src/state", type="directory", description="State module")
        idx.update(path="src/state/models.py", type="file", description="Data models")
        idx.update(path="tests", type="directory", description="Test suite")

        output = idx.tree(path="src/state")
        assert "models.py" in output
        assert "Test suite" not in output

    def test_symbol_entry_stored_in_db(self, tmp_path):
        """Symbol entries with colon-separated paths are stored in the DB."""
        db_path = str(tmp_path / "index.db")
        idx = self._make_index(db_path)
        idx.update(path="main.py", type="file", description="Entry point")
        idx.update(path="main.py:main", type="function", description="Main function")

        session = idx._get_session()
        try:
            sym = session.get(IndexEntry, "main.py:main")
            assert sym is not None
            assert sym.description == "Main function"
        finally:
            session.close()


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
                idx.update(path=entry_path, type=entry_type, description=full)

            for name in dirnames:
                if name.startswith("__pycache__") or name.startswith("."):
                    dirnames.remove(name)  # skip pycache / hidden in walk
                    continue
                entry_path = os.path.join(rel, name) if rel else name
                depth = entry_path.count(os.sep)
                if depth >= max_depth:
                    dirnames.remove(name)  # don't descend further
                    continue
                idx.update(path=entry_path + "/", type="directory", description="")

    def test_src_tree_max_depth_3(self):
        """Index the real src/ directory with max depth 3 and print the tree."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            idx = self._make_index(db_path)
            self._add_src_entries(idx, "src", max_depth=4)

            output = idx.tree(path="src/simple_agent", depth=4)

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
            assert "common_tools.py" in output
            assert "agent_process.py" in output

            # Render result for inspection
            print("\n===== tree output (max depth 3) =====")

        finally:
            os.unlink(db_path)


class TestIndexMeta:
    """Tests for IndexMeta hash storage."""

    def _make_index(self, db_path: str) -> AgentIndex:
        return AgentIndex(db_path=db_path)

    def test_hash_none_before_first_sync(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            idx = self._make_index(db_path)
            assert idx._get_hash() is None
        finally:
            os.unlink(db_path)

    def test_hash_stored_and_retrieved(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            idx = self._make_index(db_path)
            idx._set_hash("abc123")
            assert idx._get_hash() == "abc123"
        finally:
            os.unlink(db_path)

    def test_hash_overwritten(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            idx = self._make_index(db_path)
            idx._set_hash("abc123")
            idx._set_hash("def456")
            assert idx._get_hash() == "def456"
        finally:
            os.unlink(db_path)


class TestHandleDeletes:
    """Tests for AgentIndex._handle_deletes()."""

    def _make_index(self, db_path: str) -> AgentIndex:
        return AgentIndex(db_path=db_path)

    def _make_watcher(self, dir_exists=True):
        class _W:
            def path_exists_in_tree(self, _hash, _path):
                return dir_exists
            def get_file_diff(self, _old, _new, _path, context_lines=0):
                return ""
        return _W()

    def test_deletes_entry_and_returns_for_propagate(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            idx = self._make_index(db_path)
            idx.update(path="src/config.py", type="file", description="Config")
            idx.update(path="src", type="directory", description="Source")
            watcher = self._make_watcher(dir_exists=True)

            result = idx._handle_deletes(
                [("D", "src/config.py", None)], watcher, "h2",
            )
            assert result == ["src/config.py"]

            # Entry removed from DB
            session = idx._get_session()
            try:
                assert session.get(IndexEntry, "src/config.py") is None
                src_entry = session.get(IndexEntry, "src")
                assert src_entry is not None
                assert src_entry.description == "Source"
            finally:
                session.close()
        finally:
            os.unlink(db_path)

    def test_deletes_orphan_directory(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            idx = self._make_index(db_path)
            idx.update(path="old", type="directory", description="Old")
            idx.update(path="old/legacy.py", type="file", description="Legacy")
            watcher = self._make_watcher(dir_exists=False)

            result = idx._handle_deletes(
                [("D", "old/legacy.py", None)], watcher, "h2",
            )
            assert "old/legacy.py" in result
            assert "old" in result

            # Entries removed from DB
            session = idx._get_session()
            try:
                assert session.get(IndexEntry, "old/legacy.py") is None
                assert session.get(IndexEntry, "old") is None
            finally:
                session.close()
        finally:
            os.unlink(db_path)

    def test_missing_entry_added_and_no_error(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            idx = self._make_index(db_path)
            watcher = self._make_watcher()

            result = idx._handle_deletes(
                [("D", "nonexistent.py", None)], watcher, "h2",
            )
            assert result == ["nonexistent.py"]  # added, DELETE silently no-ops
        finally:
            os.unlink(db_path)

    def test_non_delete_status_ignored(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            idx = self._make_index(db_path)
            idx.update(path="mod.py", type="file", description="Mod")
            watcher = self._make_watcher()

            result = idx._handle_deletes(
                [("M", "mod.py", None), ("A", "new.py", None), ("R100", "old.py", "new.py")],
                watcher, "h2",
            )
            assert result == []
        finally:
            os.unlink(db_path)



class TestHandleModified:
    """Tests for AgentIndex._handle_modified()."""

    def _make_index(self, db_path: str) -> AgentIndex:
        return AgentIndex(db_path=db_path)

    def _make_watcher(self, diff_text=""):
        class _W:
            def path_exists_in_tree(self, _hash, _path):
                return True
            def get_file_diff(self, _old, _new, _path, context_lines=0):
                return diff_text
        return _W()

    def test_decrements_counter_on_modify(self):
        """Modified file decrements propagation_count and preserves description."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            idx = self._make_index(db_path)
            idx.update(path="src", type="directory", description="Source")
            idx.update(path="src/mod.py", type="file", description="Mod")
            watcher = self._make_watcher()

            result = idx._handle_modified(
                [("M", "src/mod.py", None)], watcher, "h1", "h2",
            )
            assert result == ["src/mod.py"]

            session = idx._get_session()
            try:
                entry = session.get(IndexEntry, "src/mod.py")
                assert entry.propagation_count == 3  # 4→3
                assert entry.description == "Mod"     # preserved
            finally:
                session.close()
        finally:
            os.unlink(db_path)

    def test_counter_zero_removes_file_entry(self):
        """When counter reaches 0, the file entry is deleted."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            idx = self._make_index(db_path)
            idx.update(path="old.py", type="file", description="Old")

            session = idx._get_session()
            try:
                entry = session.get(IndexEntry, "old.py")
                entry.propagation_count = 1
                session.commit()
            finally:
                session.close()

            watcher = self._make_watcher()
            result = idx._handle_modified(
                [("M", "old.py", None)], watcher, "h1", "h2",
            )
            assert result == ["old.py"]

            session = idx._get_session()
            try:
                assert session.get(IndexEntry, "old.py") is None
            finally:
                session.close()
        finally:
            os.unlink(db_path)

    def test_multiple_files_collected_for_propagation(self):
        """Every modified file in the index is collected for propagation."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            idx = self._make_index(db_path)
            idx.update(path="a.py", type="file", description="A")
            idx.update(path="b.py", type="file", description="B")
            watcher = self._make_watcher()

            result = idx._handle_modified(
                [("M", "a.py", None), ("M", "b.py", None)],
                watcher, "h1", "h2",
            )
            assert result == ["a.py", "b.py"]

            session = idx._get_session()
            try:
                assert session.get(IndexEntry, "a.py").propagation_count == 3
                assert session.get(IndexEntry, "b.py").propagation_count == 3
            finally:
                session.close()
        finally:
            os.unlink(db_path)

    def test_non_modified_status_ignored(self):
        """Only M status is processed; D and A are ignored."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            idx = self._make_index(db_path)
            idx.update(path="keep.py", type="file", description="Keep")
            watcher = self._make_watcher()

            result = idx._handle_modified(
                [("D", "keep.py", None), ("A", "new.py", None)],
                watcher, "h1", "h2",
            )
            assert result == []

            session = idx._get_session()
            try:
                entry = session.get(IndexEntry, "keep.py")
                assert entry.propagation_count == 4  # unchanged
                assert entry.description == "Keep"
            finally:
                session.close()
        finally:
            os.unlink(db_path)

    def test_missing_file_skipped(self):
        """Files not present in the index are silently skipped."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            idx = self._make_index(db_path)
            watcher = self._make_watcher()

            result = idx._handle_modified(
                [("M", "nonexistent.py", None)], watcher, "h1", "h2",
            )
            assert result == []
        finally:
            os.unlink(db_path)


class TestHandleAppended:
    """Tests for AgentIndex._handle_appended()."""

    def _make_index(self, db_path: str) -> AgentIndex:
        return AgentIndex(db_path=db_path)

    def test_creates_entry_and_returns_parent(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            idx = self._make_index(db_path)
            result = idx._handle_appended(
                [("A", "src/new.py", None)],
            )
            assert result == ["src/new.py"]
        finally:
            os.unlink(db_path)

    def test_top_level_file_collected(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            idx = self._make_index(db_path)
            result = idx._handle_appended(
                [("A", "top_level.py", None)],
            )
            assert result == ["top_level.py"]
        finally:
            os.unlink(db_path)

    def test_non_append_status_ignored(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            idx = self._make_index(db_path)
            result = idx._handle_appended(
                [("M", "mod.py", None), ("D", "gone.py", None)],
            )
            assert result == []
        finally:
            os.unlink(db_path)


class TestSyncOrphanDirectory:
    """Integration test: sync removes orphan directory entries."""

    def _make_index(self, db_path: str) -> AgentIndex:
        return AgentIndex(db_path=db_path)

    def test_sync_removes_directory_when_all_files_deleted(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            from git import Repo
            from simple_agent.snapshot.ghost_indexer import RepoWatcher

            repo = Repo.init(tmpdir)
            db_path = os.path.join(tmpdir, "index.db")
            idx = self._make_index(db_path)

            subdir = os.path.join(tmpdir, "mylib")
            os.makedirs(subdir)
            with open(os.path.join(subdir, "util.py"), "w") as fh:
                fh.write("x\n")
            repo.index.add(["mylib/util.py"])
            repo.index.commit("init")

            watcher = RepoWatcher(tmpdir, os.path.join(tmpdir, "shadow"))
            h1 = watcher.take_snapshot()

            idx.update(path="mylib/util.py", type="file", description="Util")
            idx.sync(None, h1, watcher)

            tree_before = idx.tree(path=tmpdir)
            assert "mylib/" in tree_before

            os.unlink(os.path.join(subdir, "util.py"))
            os.rmdir(subdir)
            repo.index.remove(["mylib/util.py"])
            repo.index.commit("deleted")
            h2 = watcher.take_snapshot()

            processed = idx.sync(h1, h2, watcher)
            assert processed >= 1

            tree_after = idx.tree(path=tmpdir)
            assert "mylib/" not in tree_after
            assert "util.py" not in tree_after



class TestPropagateStale:
    """Tests for AgentIndex._propagate_stale()."""

    def _make_index(self, db_path: str) -> AgentIndex:
        return AgentIndex(db_path=db_path)

    def test_single_change_below_threshold(self):
        """One file change (score=1.0) does not decrement counter."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            idx = self._make_index(db_path)
            idx.update(path="src", type="directory", description="Core source")

            idx._propagate_stale(["src/a.py"])

            session = idx._get_session()
            try:
                entry = session.get(IndexEntry, "src")
                assert entry.propagation_count == 4  # unchanged (1.0 < 3.0)
                assert entry.description == "Core source"
            finally:
                session.close()
        finally:
            os.unlink(db_path)

    def test_three_direct_children_trigger_decrement(self):
        """Three direct children (score=3.0) decrement counter by 1."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            idx = self._make_index(db_path)
            idx.update(path="lib", type="directory", description="Library")
            paths = [f"lib/{n}" for n in ["a.py", "b.py", "c.py"]]

            idx._propagate_stale(paths)

            session = idx._get_session()
            try:
                entry = session.get(IndexEntry, "lib")
                assert entry.propagation_count == 3  # 4→3
                assert entry.description == "Library"  # not cleared
            finally:
                session.close()
        finally:
            os.unlink(db_path)

    def test_deep_changes_decay(self):
        """Files at depth 2 contribute factor^1 = 0.7 each."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            idx = self._make_index(db_path)
            idx.update(path="src", type="directory", description="Source")
            idx.update(path="src/sub", type="directory", description="Sub")

            # 5 files at depth 2 → 5 × 0.7 = 3.5 ≥ 3 → decrement
            paths = [f"src/sub/{n}" for n in ["a.py", "b.py", "c.py", "d.py", "e.py"]]
            idx._propagate_stale(paths)

            session = idx._get_session()
            try:
                sub = session.get(IndexEntry, "src/sub")
                assert sub.propagation_count == 3  # 5.0 ≥ 3 → 4→3

                src = session.get(IndexEntry, "src")
                assert src.propagation_count == 3  # 3.5 ≥ 3 → 4→3
            finally:
                session.close()
        finally:
            os.unlink(db_path)

    def test_counter_zero_removes_entry(self):
        """When counter reaches 0, the entry is deleted from the DB."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            idx = self._make_index(db_path)
            idx.update(path="pkg", type="directory", description="Package")

            session = idx._get_session()
            try:
                entry = session.get(IndexEntry, "pkg")
                entry.propagation_count = 1
                session.commit()
            finally:
                session.close()

            # 3 files → score 3.0 ≥ 3 → counter 1→0 → entry removed
            idx._propagate_stale(["pkg/a.py", "pkg/b.py", "pkg/c.py"])

            session = idx._get_session()
            try:
                assert session.get(IndexEntry, "pkg") is None
            finally:
                session.close()
        finally:
            os.unlink(db_path)

    def test_update_description_resets_counter(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            idx = self._make_index(db_path)
            idx.update(path="dir", type="directory", description="Old")

            session = idx._get_session()
            try:
                entry = session.get(IndexEntry, "dir")
                entry.propagation_count = 0
                session.commit()
            finally:
                session.close()

            idx.update(path="dir", description="New description")

            session = idx._get_session()
            try:
                entry = session.get(IndexEntry, "dir")
                assert entry.propagation_count == 4
                assert entry.description == "New description"
            finally:
                session.close()
        finally:
            os.unlink(db_path)


class TestTreeSmoke:
    """Smoketest: render the real repo tree for visual inspection."""

    def test_render_real_repo_tree(self):
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            idx = AgentIndex(db_path)
            idx.update(path="src/simple_agent/index", type="directory",
                       description="Project index")
            idx.update(path="src/simple_agent/index/indexer.py", type="file",
                       description="AgentIndex and tree renderer")
            idx.update(path="main.py", type="file",
                       description="Application entry point")

            print()
            print(idx.tree("src/simple", depth=3))
        finally:
            os.unlink(db_path)


class TestTreeRenderPython:
    """Render Python files with tree-sitter symbol extraction."""

    def test_render_python_file_tree(self):
        from pathlib import Path
        from simple_agent.index.tree import walk_file, render_tree

        node = walk_file(Path("src/simple_agent/process/agent_process.py"))
        assert node is not None
        out = render_tree(node)
        print()
        print(out)
        assert "AgentProcess" in out
