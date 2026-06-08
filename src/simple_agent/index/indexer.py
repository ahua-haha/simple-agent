"""AgentIndex — persistent tree-structured project index for agent memory."""

from __future__ import annotations

import os

from collections.abc import Callable
from pathlib import Path

from pi.agent import AgentTool
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

    def create_tools(self) -> list[AgentTool]:
        return

    def parse_diff(self, from_commit: str, to_commit: str) -> list[str]:
        return

    def mark_expired(self, from_commit: str, to_commit: str) -> None:
        return

    def upsert_entry(self, entry: IndexNodeRecord) -> None:
        return

    def list_expired_entries(self, session: Session | None = None) -> list[IndexNodeRecord]:
        return

    def list_updated_entries(self, session: Session | None = None) -> list[IndexNodeRecord]:
        return

    def commit(self, target_commit: str) -> None:
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
