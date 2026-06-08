"""Tests for AgentIndex."""

from __future__ import annotations

import os
import json
import sqlite3
import tempfile

import pytest
from sqlalchemy import delete
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from simple_agent.index.indexer import (
    AgentIndex,
    DirectoryNode,
    FileNode,
    IndexEntry,
    IndexMeta,
    IndexNodeRecord,
    SymbolNode,
)


def upsert_index_entry(
    idx: AgentIndex,
    path: str,
    *,
    type: str | None = None,
    description: str | None = None,
) -> None:
    entry_type = type or "file"
    kind = "symbol" if entry_type in {"class", "function", "method"} else entry_type
    if kind == "folder":
        kind = "directory"
    metadata = {"description": description or ""}
    if kind == "symbol":
        metadata["symbol_type"] = entry_type

    session = idx._get_session()
    try:
        stmt = sqlite_insert(IndexNodeRecord).values(
            path=path.rstrip("/"),
            kind=kind,
            metadata_json=json.dumps(metadata, separators=(",", ":")),
            status="updated",
        )
        set_vals = {
            "status": stmt.excluded.status,
            "updated_at": stmt.excluded.updated_at,
        }
        if type is not None:
            set_vals["kind"] = stmt.excluded.kind
        if description is not None:
            set_vals["metadata"] = stmt.excluded.metadata
            set_vals["propagation_count"] = stmt.excluded.propagation_count
        session.exec(
            stmt.on_conflict_do_update(
                index_elements=[IndexNodeRecord.path],
                set_=set_vals,
            )
        )
        session.commit()
    finally:
        session.close()


def remove_index_entry(idx: AgentIndex, path: str) -> None:
    clean = path.rstrip("/")
    session = idx._get_session()
    try:
        session.exec(
            delete(IndexNodeRecord).where(
                (IndexNodeRecord.path == clean) |
                (IndexNodeRecord.path.startswith(clean + "/")) |
                (IndexNodeRecord.path.startswith(clean + ":"))
            )
        )
        session.commit()
    finally:
        session.close()


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

    def test_create_tools_only_defines_update_entry_tool_for_now(self, tmp_path):
        db_path = str(tmp_path / "index.db")
        idx = self._make_index(db_path)

        tools = idx.create_tools()

        assert [tool.name for tool in tools] == ["index_update"]

    def test_update_creates_entry(self, tmp_path):
        """update() should store a description that appears in tree()."""
        db_path = str(tmp_path / "index.db")
        ws = str(tmp_path / "ws")
        self._make_workspace(ws, {"main.py": "print('hello')"})

        idx = self._make_index(db_path, base_dir=ws)
        upsert_index_entry(idx, path="main.py", type="file", description="App entry point")

        output = idx.tree()
        assert "main.py" in output
        assert "App entry point" in output

    def test_update_upserts_existing_entry(self, tmp_path):
        """update() on existing path should overwrite the description."""
        db_path = str(tmp_path / "index.db")
        ws = str(tmp_path / "ws")
        self._make_workspace(ws, {"main.py": "x"})

        idx = self._make_index(db_path, base_dir=ws)
        upsert_index_entry(idx, path="main.py", type="file", description="Old description")
        upsert_index_entry(idx, path="main.py", type="file", description="New description")

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
        upsert_index_entry(idx, path="old", type="directory", description="Old dir")
        upsert_index_entry(idx, path="old/module.py", type="file", description="Old file")
        upsert_index_entry(idx, path="other.py", type="file", description="Other file")

        remove_index_entry(idx, "old")

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
        upsert_index_entry(idx, path="src/__init__.py", type="file", description="Package init")
        upsert_index_entry(idx, path="src/process", type="directory", description="Process modules")
        upsert_index_entry(idx, path="src/process/agent_process.py", type="file", description="Agent process")

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
        upsert_index_entry(idx, path="src", type="directory", description="Source")
        upsert_index_entry(idx, path="src/process", type="directory", description="Processes")
        upsert_index_entry(idx, path="src/process/file.py", type="file", description="File")

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
        upsert_index_entry(idx, path="src/state", type="directory", description="State module")
        upsert_index_entry(idx, path="src/state/models.py", type="file", description="Data models")
        upsert_index_entry(idx, path="tests", type="directory", description="Test suite")

        output = idx.tree(path="src/state")
        assert "models.py" in output
        assert "Test suite" not in output

    def test_symbol_entry_stored_in_db(self, tmp_path):
        """Symbol entries with colon-separated paths are stored in the DB."""
        db_path = str(tmp_path / "index.db")
        idx = self._make_index(db_path)
        upsert_index_entry(idx, path="main.py", type="file", description="Entry point")
        upsert_index_entry(idx, path="main.py:main", type="function", description="Main function")

        session = idx._get_session()
        try:
            sym = session.get(IndexNodeRecord, "main.py:main")
            assert sym is not None
            assert sym.kind == "symbol"
            assert SymbolNode.from_record(sym).description == "Main function"
            assert SymbolNode.from_record(sym).symbol_type == "function"
        finally:
            session.close()

    def test_index_node_record_has_generic_metadata_schema(self, tmp_path):
        db_path = str(tmp_path / "index.db")
        self._make_index(db_path)

        with sqlite3.connect(db_path) as conn:
            columns = conn.execute("PRAGMA table_info(index_nodes)").fetchall()

        names = {column[1] for column in columns}
        assert names == {"path", "kind", "metadata", "expired", "propagation_count", "updated_at"}

    def test_file_node_roundtrips_description_through_metadata(self, tmp_path):
        db_path = str(tmp_path / "index.db")
        idx = self._make_index(db_path)

        upsert_index_entry(idx, path="main.py", type="file", description="Entry point")

        session = idx._get_session()
        try:
            record = session.get(IndexNodeRecord, "main.py")
            assert record is not None
            assert record.kind == "file"
            node = FileNode.from_record(record)
            assert node.description == "Entry point"
        finally:
            session.close()

    def test_update_marks_entry_not_expired(self, tmp_path):
        db_path = str(tmp_path / "index.db")
        idx = self._make_index(db_path)

        upsert_index_entry(idx, path="main.py", type="file", description="Old")
        session = idx._get_session()
        try:
            record = session.get(IndexNodeRecord, "main.py")
            assert record is not None
            record.expired = True
            session.commit()
        finally:
            session.close()

        upsert_index_entry(idx, path="main.py", type="file", description="New")

        session = idx._get_session()
        try:
            record = session.get(IndexNodeRecord, "main.py")
            assert record is not None
            assert record.expired is False
        finally:
            session.close()

    def test_directory_node_roundtrips_description_through_metadata(self, tmp_path):
        db_path = str(tmp_path / "index.db")
        idx = self._make_index(db_path)

        upsert_index_entry(idx, path="src", type="directory", description="Source directory")

        session = idx._get_session()
        try:
            record = session.get(IndexNodeRecord, "src")
            assert record is not None
            assert record.kind == "directory"
            node = DirectoryNode.from_record(record)
            assert node.description == "Source directory"
        finally:
            session.close()

    def test_update_description_resets_counter(self, tmp_path):
        db_path = str(tmp_path / "index.db")
        idx = self._make_index(db_path)

        upsert_index_entry(idx, path="dir", type="directory", description="Old")

        session = idx._get_session()
        try:
            entry = session.get(IndexEntry, "dir")
            assert entry is not None
            entry.propagation_count = 0
            session.commit()
        finally:
            session.close()

        upsert_index_entry(idx, path="dir", description="New description")

        session = idx._get_session()
        try:
            entry = session.get(IndexEntry, "dir")
            assert entry is not None
            assert entry.propagation_count == 4
            assert entry.description == "New description"
        finally:
            session.close()

    def test_upsert_entry_merges_metadata_json(self, tmp_path):
        db_path = str(tmp_path / "index.db")
        idx = self._make_index(db_path)

        idx.upsert_entry(
            "main.py",
            {"kind": "file", "description": "Entry point", "owner": "runtime"},
        )
        idx.upsert_entry(
            "main.py",
            {"description": "Updated entry"},
        )

        session = idx._get_session()
        try:
            entry = session.get(IndexNodeRecord, "main.py")
            assert entry is not None
            assert json.loads(entry.metadata_json) == {
                "description": "Updated entry",
                "owner": "runtime",
            }
            assert entry.status == "updated"
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
                upsert_index_entry(idx, path=entry_path, type=entry_type, description=full)

            for name in dirnames:
                if name.startswith("__pycache__") or name.startswith("."):
                    dirnames.remove(name)  # skip pycache / hidden in walk
                    continue
                entry_path = os.path.join(rel, name) if rel else name
                depth = entry_path.count(os.sep)
                if depth >= max_depth:
                    dirnames.remove(name)  # don't descend further
                    continue
                upsert_index_entry(idx, path=entry_path + "/", type="directory", description="")

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
    """Tests for AgentIndex commit metadata and status workflow."""

    def _make_index(self, db_path: str) -> AgentIndex:
        return AgentIndex(db_path=db_path)

    def test_parse_diff_returns_changed_files(self):
        class _Watcher:
            def get_changed_files_with_rename(self, from_commit, to_commit):
                assert from_commit == "abc123"
                assert to_commit == "def456"
                return [
                    ("M", "src/app.py", None),
                    ("A", "src/new.py", None),
                    ("D", "src/old.py", None),
                    ("R100", "src/name.py", "src/renamed.py"),
                ]

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            idx = AgentIndex(db_path=db_path, repo_watcher=_Watcher())

            assert idx.parse_diff("abc123", "def456") == [
                "src/app.py",
                "src/new.py",
                "src/old.py",
                "src/name.py",
                "src/renamed.py",
            ]
        finally:
            os.unlink(db_path)

    def test_auto_commit_marks_changed_entries_expired_and_sets_commit(self):
        class _Watcher:
            def get_changed_files_with_rename(self, from_commit, to_commit):
                assert from_commit == "abc123"
                assert to_commit == "def456"
                return [("M", "src/app.py", None)]

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            idx = AgentIndex(db_path=db_path, repo_watcher=_Watcher())
            upsert_index_entry(idx, path="src/app.py", type="file", description="App")
            upsert_index_entry(idx, path="src/app.py:main", type="function", description="Main")
            upsert_index_entry(idx, path="src/other.py", type="file", description="Other")
            session = idx._get_session()
            try:
                session.add(IndexMeta(key="current_commit", value="abc123"))
                session.commit()
            finally:
                session.close()

            idx.auto_commit("def456")

            session = idx._get_session()
            try:
                app = session.get(IndexNodeRecord, "src/app.py")
                symbol = session.get(IndexNodeRecord, "src/app.py:main")
                other = session.get(IndexNodeRecord, "src/other.py")
                meta = session.get(IndexMeta, "current_commit")

                assert app is not None
                assert app.status == "expired"
                assert symbol is not None
                assert symbol.status == "expired"
                assert other is not None
                assert other.status == "updated"
                assert meta is not None
                assert meta.value == "def456"
            finally:
                session.close()
        finally:
            os.unlink(db_path)

    def test_auto_commit_marks_directory_descendants_expired(self):
        class _Watcher:
            def get_changed_files_with_rename(self, from_commit, to_commit):
                return [("M", "src/pkg", None)]

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            idx = AgentIndex(db_path=db_path, repo_watcher=_Watcher())
            upsert_index_entry(idx, path="src/pkg/file.py", type="file", description="File")
            upsert_index_entry(idx, path="src/other.py", type="file", description="Other")
            session = idx._get_session()
            try:
                session.add(IndexMeta(key="current_commit", value="abc123"))
                session.commit()
            finally:
                session.close()

            idx.auto_commit("def456")

            expired_paths = {entry.path for entry in idx.list_expired_entries()}
            updated_paths = {entry.path for entry in idx.list_updated_entries()}
            assert expired_paths == {"src/pkg/file.py"}
            assert updated_paths == {"src/other.py"}
        finally:
            os.unlink(db_path)

    def test_auto_commit_marks_renamed_old_and_new_paths_expired(self):
        class _Watcher:
            def get_changed_files_with_rename(self, from_commit, to_commit):
                return [("R100", "src/old.py", "src/new.py")]

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            idx = AgentIndex(db_path=db_path, repo_watcher=_Watcher())
            upsert_index_entry(idx, path="src/old.py", type="file", description="Old")
            upsert_index_entry(idx, path="src/new.py", type="file", description="New")
            session = idx._get_session()
            try:
                session.add(IndexMeta(key="current_commit", value="abc123"))
                session.commit()
            finally:
                session.close()

            idx.auto_commit("def456")

            assert {entry.path for entry in idx.list_expired_entries()} == {
                "src/old.py",
                "src/new.py",
            }
        finally:
            os.unlink(db_path)

    def test_auto_commit_sets_initial_commit_without_diff(self):
        class _Watcher:
            def get_changed_files_with_rename(self, from_commit, to_commit):
                raise AssertionError("first auto_commit should not parse diff")

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            idx = AgentIndex(db_path=db_path, repo_watcher=_Watcher())
            upsert_index_entry(idx, path="src/app.py", type="file", description="App")

            idx.auto_commit("def456")

            session = idx._get_session()
            try:
                app = session.get(IndexNodeRecord, "src/app.py")
                meta = session.get(IndexMeta, "current_commit")
                assert app is not None
                assert app.status == "updated"
                assert meta is not None
                assert meta.value == "def456"
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
            upsert_index_entry(idx, path="src/simple_agent/index", type="directory",
                       description="Project index")
            upsert_index_entry(idx, path="src/simple_agent/index/indexer.py", type="file",
                       description="AgentIndex and tree renderer")
            upsert_index_entry(idx, path="main.py", type="file",
                       description="Application entry point")

            print()
            print(idx.tree("src/simple", depth=3))
        finally:
            os.unlink(db_path)


class TestTreeRenderPython:
    """Render Python files with tree-sitter symbol extraction."""

    def test_node_formats_name_and_comment(self):
        from simple_agent.index.models import SymbolNode
        from simple_agent.index.tree import render_tree

        node = SymbolNode(
            path="src/app.py:run()",
            description="Runs the app",
            symbol_type="function",
        )

        assert node.format_node() == "app.py:run()  # Runs the app [function]"
        assert "# Runs the app [function]" in render_tree(node)

    def test_walk_file_accepts_root_node_and_returns_same_node(self):
        from pathlib import Path
        from simple_agent.index.models import FileNode
        from simple_agent.index.tree import WalkOptions, walk_file

        path = Path("src/simple_agent/process/agent_process.py")
        root = FileNode(path=str(path))

        assert walk_file(root, WalkOptions()) is root

    def test_build_tree_accepts_file_path(self):
        from simple_agent.index.models import FileNode
        from simple_agent.index.tree import WalkOptions, build_tree

        root = FileNode(path="src/simple_agent/index/tree.py")
        node = build_tree(root, WalkOptions())

        assert node is root
        assert node.path.endswith("src/simple_agent/index/tree.py")

    def test_render_python_file_tree(self):
        from pathlib import Path
        pytest.importorskip("tree_sitter")
        pytest.importorskip("tree_sitter_python")
        from simple_agent.index.models import FileNode
        from simple_agent.index.tree import WalkOptions, render_tree, walk_file

        path = Path("src/simple_agent/process/agent_process.py")
        node = walk_file(FileNode(path=str(path)), WalkOptions())
        assert node is not None
        out = render_tree(node)
        print()
        print(out)
        assert "AgentProcess" in out
