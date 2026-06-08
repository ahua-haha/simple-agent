"""AgentIndex — persistent tree-structured project index for agent memory."""

from __future__ import annotations

import json
import os
import time

from collections.abc import Callable
from pathlib import Path

from pi.agent import AgentTool, AgentToolResult
from pi.ai.types import TextContent
from sqlmodel import SQLModel, Field, Session, create_engine, select

from simple_agent.index.models import (
    BaseNode,
    DirectoryNode,
    FileNode,
    IndexNodeRecord,
    SymbolNode,
)
from simple_agent.snapshot.ghost_indexer import RepoWatcher
from simple_agent.index.tree import WalkOptions, build_tree, render_tree
import pathspec


IndexEntry = IndexNodeRecord


class IndexMeta(SQLModel, table=True):
    __tablename__ = "index_meta"

    key: str = Field(primary_key=True)
    value: str = Field(default="")


class AgentIndex:
    """Tree-structured index of project files, folders, and symbols.

    Bound to a repo at *base_dir*. One index per repo, stored at
    *db_path*.
    """

    def __init__(self, db_path: str = "./data/agent_index.db", *,
                 base_dir: str = "."):
        self._base_dir = Path(base_dir).resolve()
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._engine = create_engine(
            f"sqlite:///{db_path}",
            connect_args={"check_same_thread": False},
        )
        SQLModel.metadata.create_all(self._engine)

    def _get_session(self) -> Session:
        return Session(self._engine)

    def create_index_tools(self) -> list[AgentTool]:
        tree_tool = AgentTool(
            name="index_tree",
            description="Render the project index as a tree with # descriptions. Use to review what's known about the codebase structure.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Subtree path to render (default: root)"},
                    "depth": {"type": "integer", "description": "Max depth to render (default: unlimited)"},
                },
                "required": [],
            },
        )

        async def tree_execute(tool_call_id: str, params: dict, cancel_event=None, on_update=None) -> AgentToolResult:
            output = self.tree(path=params.get("path", ""), depth=params.get("depth"))
            return AgentToolResult(content=[TextContent(text=output)])

        tree_tool.execute = tree_execute

        update_tool = AgentTool(
            name="index_update",
            description="Add or update an entry in the project index. Use after discovering a new file, class, function, or module.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Entry path, e.g. 'src/main.py' or 'src/main.py:main()'"},
                    "type": {"type": "string", "description": "Entry type: folder, file, class, function, method"},
                    "description": {"type": "string", "description": "Text description of what this entry does"},
                },
                "required": ["path", "type", "description"],
            },
        )

        async def update_execute(tool_call_id: str, params: dict, cancel_event=None, on_update=None) -> AgentToolResult:
            self.update(
                path=params["path"],
                type=params.get("type", "file"),
                description=params.get("description", ""),
            )
            return AgentToolResult(content=[TextContent(text="ok")])

        update_tool.execute = update_execute

        remove_tool = AgentTool(
            name="index_remove",
            description="Remove an entry and all its children from the project index.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to remove, e.g. 'src/old_module/'"},
                },
                "required": ["path"],
            },
        )

        async def remove_execute(tool_call_id: str, params: dict, cancel_event=None, on_update=None) -> AgentToolResult:
            self.remove(path=params["path"])
            return AgentToolResult(content=[TextContent(text="ok")])

        remove_tool.execute = remove_execute

        return [tree_tool, update_tool, remove_tool]

    def _get_hash(self, _session: Session | None = None) -> str | None:
        sess = _session or self._get_session()
        meta = sess.exec(
            select(IndexMeta).where(IndexMeta.key == "repo_hash")
        ).first()
        if _session is None:
            sess.close()
        return meta.value if meta else None

    def _set_hash(self, hash_value: str, _session: Session | None = None) -> None:
        _close = _session is None
        sess = _session or self._get_session()
        meta = sess.exec(
            select(IndexMeta).where(IndexMeta.key == "repo_hash")
        ).first()
        if meta:
            meta.value = hash_value
        else:
            sess.add(IndexMeta(key="repo_hash", value=hash_value))
        if _close:
            sess.commit()
            sess.close()

    @staticmethod
    def _get_parent(path: str) -> str | None:
        """Return the parent directory of *path*, or None at root.

        ``'src/utils/helper.py'`` → ``'src/utils'``
        ``'src'`` → ``''``
        ``''`` → ``None``
        """
        if not path:
            return None
        if "/" not in path:
            return ""
        return path.rsplit("/", 1)[0]

    def update(
        self,
        path: str,
        *,
        type: str | None = None,
        description: str | None = None,
        _session: Session | None = None,
    ) -> None:
        """Upsert a single entry by path. Only non-None fields are updated."""
        _close = _session is None
        sess = _session or self._get_session()
        p = path.rstrip("/")

        from sqlalchemy.dialects.sqlite import insert as sqlite_insert

        now = int(time.time())
        entry_type = type or "file"
        kind = "symbol" if entry_type in {"class", "function", "method"} else entry_type
        if kind == "folder":
            kind = "directory"
        metadata = {"description": description or ""}
        if kind == "symbol":
            metadata["symbol_type"] = entry_type

        stmt = sqlite_insert(IndexNodeRecord).values(
            path=p,
            kind=kind,
            metadata_json=json.dumps(metadata, separators=(",", ":")),
            updated_at=now,
        )

        update_cols: set[str] = {"updated_at"}
        if type is not None:
            update_cols.add("kind")
        if description is not None:
            update_cols.add("metadata_json")
            update_cols.add("propagation_count")

        set_vals = {}
        for column_name in update_cols:
            if column_name == "metadata_json":
                set_vals["metadata"] = stmt.excluded.metadata
            else:
                set_vals[column_name] = getattr(stmt.excluded, column_name)

        sess.exec(stmt.on_conflict_do_update(
            index_elements=[IndexNodeRecord.path],
            set_=set_vals,
        ))

        if _close:
            sess.commit()
            sess.close()

    def remove(self, path: str, _session: Session | None = None) -> None:
        """Remove an entry and all its descendants."""
        _close = _session is None
        sess = _session or self._get_session()
        clean = path.rstrip("/")
        from sqlalchemy import delete
        sess.exec(
            delete(IndexNodeRecord).where(
                (IndexNodeRecord.path == clean) |
                (IndexNodeRecord.path.startswith(clean + "/")) |
                (IndexNodeRecord.path.startswith(clean + ":"))
            )
        )
        if _close:
            sess.commit()
            sess.close()

    def _handle_deletes(
        self,
        changes: list[tuple[str, str, str | None]],
        repo_watcher: RepoWatcher,
        new_hash: str,
        _session: Session | None = None,
    ) -> list[str]:
        """Remove deleted entries from the index and return their paths
        for propagation.

        Phase 1 collects all paths to remove (files + orphan directories).
        Phase 2 batch-deletes each path and its children.
        """
        _close = _session is None
        sess = _session or self._get_session()

        def _dir_exists(path: str) -> bool:
            return repo_watcher.path_exists_in_tree(new_hash, path)

        # Phase 1: collect
        removed: set[str] = set()

        for status, old_path, _ in changes:
            if status != "D":
                continue

            removed.add(old_path)

            parent = self._get_parent(old_path)
            while parent is not None:
                if parent and not _dir_exists(parent):
                    removed.add(parent)
                parent = self._get_parent(parent)

        # Phase 2: batch delete
        from sqlalchemy import delete as sql_delete
        for path in removed:
            sess.exec(
                sql_delete(IndexNodeRecord).where(
                    (IndexNodeRecord.path == path) |
                    (IndexNodeRecord.path.startswith(path + "/")) |
                    (IndexNodeRecord.path.startswith(path + ":"))
                )
            )

        if _close:
            sess.commit()
            sess.close()
        return list(removed)

    def _parse_file_tree(
        self,
        file_path: str,
    ) -> list[IndexNodeRecord]:
        """Parse *file_path* and return symbol ``IndexNodeRecord`` objects.

        TODO: implement language-specific parsing.
        """
        return []

    def parse_file_diff(
        self,
        file_path: str,
        repo_watcher: RepoWatcher,
        old_hash: str,
        new_hash: str,
    ) -> list[IndexNodeRecord]:
        """Parse the diff for *file_path* and return entries to delete.

        Fetches the file diff internally from *repo_watcher*, parses it
        to identify which indexed symbol entries were deleted or modified,
        and returns them for deletion.

        TODO: implement diff-based symbol detection.
        """
        _diff = repo_watcher.get_file_diff(old_hash, new_hash, file_path, context_lines=0)
        return []

    def _handle_modified(
        self,
        changes: list[tuple[str, str, str | None]],
        repo_watcher: RepoWatcher,
        old_hash: str,
        new_hash: str,
        _session: Session | None = None,
    ) -> list[str]:
        """Handle modified files: use ``parse_file_diff`` to identify deleted
        symbol entries, decrement the file's ``propagation_count``, and
        remove the file entry when the counter reaches 0.

        Returns the collected paths for propagation to parent directories."""
        _close = _session is None
        sess = _session or self._get_session()

        from sqlalchemy import delete as sql_delete

        collected: list[str] = []

        for status, old_path, _ in changes:
            if status != "M":
                continue

            # Delete symbol entries identified by parse_file_diff
            deleted = self.parse_file_diff(old_path, repo_watcher, old_hash, new_hash)
            for d in deleted:
                sess.exec(
                    sql_delete(IndexNodeRecord).where(IndexNodeRecord.path == d.path)
                )

            # Decrement file counter; remove when exhausted
            entry = sess.get(IndexNodeRecord, old_path)
            if entry is None:
                continue
            entry.propagation_count -= 1
            if entry.propagation_count <= 0:
                sess.delete(entry)
            collected.append(old_path)

        if _close:
            sess.commit()
            sess.close()
        return collected

    @staticmethod
    def _handle_appended(
        changes: list[tuple[str, str, str | None]],
    ) -> list[str]:
        """Collect appended file paths for propagation."""
        appended: list[str] = []
        for status, old_path, _ in changes:
            if status == "A":
                appended.append(old_path)
        return appended

    def _propagate_stale(
        self,
        paths: list[str],
        _session: Session | None = None,
        factor: float = 0.7,
        threshold: float = 3.0,
    ) -> None:
        """Apply distance-decay scoring to ancestor directories.

        Each changed path contributes ``factor^(d-1)`` to its ancestor
        at distance *d* (direct child = 1).  When an ancestor's
        accumulated score meets *threshold*, its ``propagation_count``
        is decremented by 1.  When the counter reaches 0, the entry is
        removed from the database.
        """
        _close = _session is None
        sess = _session or self._get_session()

        # Accumulate decay-weighted scores per ancestor
        scores: dict[str, float] = {}
        for path in paths:
            d = 1
            parent = self._get_parent(path)
            while parent is not None:
                scores[parent] = scores.get(parent, 0.0) + factor ** (d - 1)
                d += 1
                parent = self._get_parent(parent)

        # Decrement counters for ancestors that pass threshold
        from sqlalchemy import delete as sql_delete
        for dir_path, score in scores.items():
            if score < threshold:
                continue
            entry = sess.get(IndexNodeRecord, dir_path)
            if entry is None:
                continue
            entry.propagation_count -= 1
            if entry.propagation_count <= 0:
                sess.delete(entry)

        if _close:
            sess.commit()
            sess.close()

    def sync(self, old_hash: str | None, new_hash: str, repo_watcher: RepoWatcher) -> int:
        """Sync the index to *new_hash* by processing all file changes since *old_hash*.

        Uses rename detection (``-M50%``) to handle renames before deletions
        and modifications. All operations run in a single transaction.
        Returns the count of files processed.
        """
        if old_hash is None:
            self._set_hash(new_hash)
            return 0

        changes = repo_watcher.get_changed_files_with_rename(old_hash, new_hash)
        if not changes:
            self._set_hash(new_hash)
            return 0

        processed = 0
        session = self._get_session()
        try:
            # Phase 1: deletes
            deleted = self._handle_deletes(changes, repo_watcher, new_hash, _session=session)
            processed += len(deleted)

            # Phase 2: modifications
            modified = self._handle_modified(changes, repo_watcher, old_hash, new_hash, _session=session)

            # Phase 3: appended
            appended = self._handle_appended(changes)

            # Phase 4: propagate
            self._propagate_stale(deleted + modified + appended, _session=session)

            self._set_hash(new_hash, _session=session)
            session.commit()
        finally:
            session.close()
        return processed

    def _load_gitignore_spec(self) -> pathspec.PathSpec | None:
        gitignore = self._base_dir / ".gitignore"
        try:
            with gitignore.open() as f:
                return pathspec.PathSpec.from_lines("gitignore", f)
        except FileNotFoundError:
            return None

    @staticmethod
    def _format_path(path: Path) -> str:
        """Normalize *path* to a string with ``/`` suffix for directories."""
        s = str(path)
        if path.is_dir():
            s += "/"
        return s

    def _make_tree_filter(self) -> Callable[[Path], bool]:
        spec = self._load_gitignore_spec()
        base = self._base_dir

        def filter_fn(abs_path: Path) -> bool:
            if spec is not None:
                try:
                    rel = abs_path.relative_to(base)
                except ValueError:
                    return False
                if spec.match_file(self._format_path(rel)):
                    return True
            return False

        return filter_fn

    def _load_descriptions(self) -> dict[str, str]:
        sess = self._get_session()
        try:
            entries = sess.exec(select(IndexNodeRecord)).all()
            return {e.path: e.description for e in entries if e.description}
        finally:
            sess.close()

    @staticmethod
    def _format_comments(node: BaseNode, descs: dict[str, str], base: Path) -> None:
        """Walk the tree and set node descriptions from stored index descriptions."""
        try:
            rel = str(Path(node.path).relative_to(base))
        except ValueError:
            rel = node.path
        desc = descs.get(rel, "")
        if desc:
            node.description = desc

        for child in node.children:
            AgentIndex._format_comments(child, descs, base)

    def tree(
        self,
        path: str = "",
        depth: int | None = None,
    ) -> str:
        """Render the index as a tree with # descriptions.

        *path* is relative to *base_dir*. Structure comes from the
        filesystem; descriptions from the database.
        """
        filter_fn = self._make_tree_filter()
        full_path = (self._base_dir / path) if path else self._base_dir
        if full_path.is_dir():
            root: BaseNode = DirectoryNode(path=str(full_path))
        elif full_path.is_file():
            root = FileNode(path=str(full_path))
        else:
            return "(empty)"

        root = build_tree(root, WalkOptions(depth=depth, filter_fn=filter_fn))
        descs = self._load_descriptions()
        self._format_comments(root, descs, self._base_dir)
        return render_tree(root)
