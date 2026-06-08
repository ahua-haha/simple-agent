"""AgentIndex — persistent tree-structured project index for agent memory."""

from __future__ import annotations

import json
import os
import time

from collections.abc import Callable
from pathlib import Path

from pi.agent import AgentTool
from sqlalchemy import or_, update
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
        return

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


def _metadata_dict(metadata_json: str) -> dict:
    metadata = json.loads(metadata_json or "{}")
    if not isinstance(metadata, dict):
        raise ValueError("Index node metadata must be a JSON object")
    return metadata


def _metadata_json(metadata: dict) -> str:
    return json.dumps(metadata, sort_keys=True, separators=(",", ":"))
