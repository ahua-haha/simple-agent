"""AgentIndex — persistent tree-structured project index for agent memory."""

from __future__ import annotations

import json
import os
import time

from collections import deque
from collections.abc import Callable
from pathlib import Path

from pi.agent import AgentTool, AgentToolResult
from pi.ai.types import TextContent
from sqlalchemy import or_, update
from sqlmodel import SQLModel, Field, Session, create_engine, select

from simple_agent.index.models import (
    BaseNode,
    DirectoryNode,
    FileNode,
    IndexNodeRecord,
    SymbolNode,
    index_node_from_record,
)
from simple_agent.snapshot.ghost_indexer import RepoWatcher
from simple_agent.index.tree import WalkOptions, build_tree, render_tree
import pathspec


IndexEntry = IndexNodeRecord


class IndexUpdateError(Exception):
    """Raised when an index update cannot be committed."""


class IndexMeta(SQLModel, table=True):
    __tablename__ = "index_meta"

    key: str = Field(primary_key=True)
    value: str = Field(default="")


class AgentIndex:
    """Tree-structured index of project files, folders, and symbols.

    Bound to a repo at *base_dir*. One index per repo, stored at
    *db_path*.
    """

    def __init__(
        self,
        db_path: str = "./data/agent_index.db",
        *,
        base_dir: str = ".",
        repo_watcher: RepoWatcher | None = None,
    ):
        self._base_dir = Path(base_dir).resolve()
        self._repo_watcher = repo_watcher
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._engine = create_engine(
            f"sqlite:///{db_path}",
            connect_args={"check_same_thread": False},
        )
        SQLModel.metadata.create_all(self._engine)

    def _get_session(self) -> Session:
        return Session(self._engine)

    def create_tools(self) -> list[AgentTool]:
        return [
            self.create_tree_tool(),
            self.create_upsert_tool(),
        ]

    def create_tree_tool(self) -> AgentTool:
        tool = AgentTool(
            name="index_tree",
            description=(
                "Explore the repository tree and inspect repo structure. "
                "Use this to inspect the corresponding AgentIndex memory for each entry."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Optional path under the indexed repo to render.",
                    },
                    "depth": {
                        "type": "integer",
                        "description": "Optional maximum depth to render.",
                    },
                    "entry_limit": {
                        "type": "integer",
                        "description": (
                            "Optional maximum number of tree entries to render. "
                            "If exceeded, render depth is reduced until the entry count fits."
                        ),
                    },
                },
            },
        )

        async def execute(tool_call_id, params, cancel_event=None, on_update=None):
            output = self.tree(
                path=params.get("path", ""),
                depth=params.get("depth"),
                entry_limit=params.get("entry_limit"),
            )
            return AgentToolResult(content=[TextContent(text=output)])

        tool.execute = execute
        return tool

    def create_upsert_tool(self) -> AgentTool:
        tool = AgentTool(
            name="index_upsert",
            description=(
                "Create or update one repo memory entry in AgentIndex. "
                "Only write concise factual metadata for paths you inspected."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path_id": {
                        "type": "string",
                        "description": "Index path id, such as src/app.py or src/app.py:ClassName.",
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Metadata JSON to merge into the entry. Include kind and description.",
                    },
                },
                "required": ["path_id", "metadata"],
            },
        )

        async def execute(tool_call_id, params, cancel_event=None, on_update=None):
            path_id = params["path_id"]
            metadata = params["metadata"]
            if not isinstance(metadata, dict):
                raise ValueError("index_upsert metadata must be an object")
            self.upsert_entry(path_id, metadata)
            return AgentToolResult(content=[TextContent(text=f"Updated index entry: {path_id}")])

        tool.execute = execute
        return tool

    def parse_diff(self, from_commit: str, to_commit: str) -> list[str]:
        if self._repo_watcher is None:
            return []
        changes = self._repo_watcher.get_changed_files_with_rename(from_commit, to_commit)
        changed_files: list[str] = []
        for _status, old_path, new_path in changes:
            changed_files.append(old_path)
            if new_path is not None:
                changed_files.append(new_path)
        return changed_files

    def auto_commit(self, target_commit: str) -> None:
        with self._get_session() as session:
            meta = session.get(IndexMeta, "current_commit")
            current_commit = meta.value if meta is not None else None
            changed_paths = (
                self.parse_diff(current_commit, target_commit)
                if current_commit is not None
                else []
            )

            if changed_paths:
                self._expire_changed_entries(changed_paths, session)

            if meta is None:
                session.add(IndexMeta(key="current_commit", value=target_commit))
            else:
                meta.value = target_commit
            session.commit()

    def upsert_entry(self, path_id: str, update_json: dict) -> None:
        update_data = dict(update_json)
        kind = str(update_data.pop("kind", "file"))
        with self._get_session() as session:
            existing = session.get(IndexNodeRecord, path_id)
            if existing is None:
                entry = IndexNodeRecord(
                    path=path_id,
                    kind=kind,
                    metadata_json=_metadata_json(update_data),
                    status="updated",
                    updated_at=int(time.time()),
                )
                session.add(entry)
            else:
                existing.kind = kind
                existing.metadata_json = _metadata_json(
                    _metadata_dict(existing.metadata_json) | update_data
                )
                existing.status = "updated"
                existing.updated_at = int(time.time())
            session.commit()

    def list_expired_entries(self, session: Session | None = None) -> list[IndexNodeRecord]:
        statement = select(IndexNodeRecord).where(IndexNodeRecord.status == "expired")
        if session is not None:
            return list(session.exec(statement).all())
        with self._get_session() as own_session:
            return list(own_session.exec(statement).all())

    def list_updated_entries(self, session: Session | None = None) -> list[IndexNodeRecord]:
        statement = select(IndexNodeRecord).where(IndexNodeRecord.status == "updated")
        if session is not None:
            return list(session.exec(statement).all())
        with self._get_session() as own_session:
            return list(own_session.exec(statement).all())

    def commit(self, target_commit: str) -> None:
        with self._get_session() as session:
            meta = session.get(IndexMeta, "current_commit")
            current_commit = meta.value if meta is not None else None
            changed_paths = (
                self.parse_diff(current_commit, target_commit)
                if current_commit is not None
                else []
            )

            if changed_paths:
                self._expire_changed_entries(changed_paths, session)

            self._check_updated_entries_exist(target_commit, session)

            if meta is None:
                session.add(IndexMeta(key="current_commit", value=target_commit))
            else:
                meta.value = target_commit
            session.commit()

    def _expire_changed_entries(self, changed_paths: list[str], session: Session) -> None:
        conditions = []
        for path in changed_paths:
            clean_path = path.rstrip("/")
            conditions.extend(
                [
                    IndexNodeRecord.path == clean_path,
                    IndexNodeRecord.path.startswith(clean_path + "/"),
                    IndexNodeRecord.path.startswith(clean_path + ":"),
                ]
            )
        if not conditions:
            return
        session.exec(
            update(IndexNodeRecord)
            .where(or_(*conditions))
            .where(IndexNodeRecord.status != "updated")
            .where(IndexNodeRecord.status != "expired")
            .values(status="expired")
        )

    def _check_updated_entries_exist(self, target_commit: str, session: Session) -> None:
        # TODO: verify every updated entry still exists in target_commit.
        return

    def _load_gitignore_spec(self) -> pathspec.PathSpec:
        lines = [
            ".git/",
            "**/.git/",
            ".venv/",
            "**/.venv/",
            "__pycache__/",
            "**/__pycache__/",
            ".pytest_cache/",
            "**/.pytest_cache/",
        ]
        gitignore = self._base_dir / ".gitignore"
        try:
            with gitignore.open() as f:
                lines.extend(f)
        except FileNotFoundError:
            pass
        return pathspec.PathSpec.from_lines("gitignore", lines)

    def _make_tree_filter(self) -> Callable[[Path], bool]:
        spec = self._load_gitignore_spec()
        base = self._base_dir

        def filter_fn(abs_path: Path) -> bool:
            try:
                rel = abs_path.relative_to(base)
            except ValueError:
                return False
            rel_text = rel.as_posix()
            if abs_path.is_dir():
                rel_text += "/"
            return spec.match_file(rel_text)

        return filter_fn

    def _load_entries(self) -> dict[str, IndexNodeRecord]:
        with self._get_session() as session:
            entries = session.exec(select(IndexNodeRecord)).all()
            return {entry.path: entry for entry in entries}

    def _fill_tree_metadata(self, node: BaseNode, entries: dict[str, IndexNodeRecord]) -> None:
        """Walk the tree and fill matching nodes with stored index metadata."""
        entry = entries.get(self._node_path_id(node))
        if entry is not None:
            self._apply_entry_metadata(node, entry)

        for child in node.children:
            self._fill_tree_metadata(child, entries)

    def _node_path_id(self, node: BaseNode) -> str:
        path = node.path
        base = str(self._base_dir)
        if path == base:
            return "."
        prefix = base + os.sep
        if path.startswith(prefix):
            return path[len(prefix):]
        return path

    @staticmethod
    def _apply_entry_metadata(node: BaseNode, entry: IndexNodeRecord) -> None:
        entry_node = index_node_from_record(entry)
        node.description = entry_node.description
        node.propagation_count = entry_node.propagation_count
        node.updated_at = entry_node.updated_at
        if hasattr(node, "symbol_type") and hasattr(entry_node, "symbol_type"):
            node.symbol_type = entry_node.symbol_type

    def tree(
        self,
        path: str = "",
        depth: int | None = None,
        entry_limit: int | None = 48,
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
            return f'Error: path not found in repo: "{path or "."}"'

        root = build_tree(root, WalkOptions(depth=depth, filter_fn=filter_fn))
        entries = self._load_entries()
        self._fill_tree_metadata(root, entries)
        render_depth = (
            _max_render_depth_for_entry_limit(root, entry_limit)
            if entry_limit is not None
            else None
        )
        return render_tree(root, depth=render_depth)


def _max_render_depth_for_entry_limit(root: BaseNode, entry_limit: int) -> int:
    if entry_limit <= 0:
        return 0
    counts_by_depth: dict[int, int] = {}
    queue = deque([(root, 0)])
    while queue:
        node, depth = queue.popleft()
        counts_by_depth[depth] = counts_by_depth.get(depth, 0) + 1
        for child in node.children:
            queue.append((child, depth + 1))

    total = 0
    best_depth = 0
    for depth in sorted(counts_by_depth):
        next_total = total + counts_by_depth[depth]
        if next_total > entry_limit:
            return best_depth
        total = next_total
        best_depth = depth
    return best_depth


def _metadata_dict(metadata_json: str) -> dict:
    metadata = json.loads(metadata_json or "{}")
    if not isinstance(metadata, dict):
        raise ValueError("Index node metadata must be a JSON object")
    return metadata


def _metadata_json(metadata: dict) -> str:
    return json.dumps(metadata, sort_keys=True, separators=(",", ":"))
