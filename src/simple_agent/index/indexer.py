"""AgentIndex — persistent tree-structured project index for agent memory."""

from __future__ import annotations

import os
import time

from collections.abc import Callable
from pathlib import Path
from typing import Any

from sqlmodel import SQLModel, Field, Session, create_engine, select

from simple_agent.snapshot.ghost_indexer import RepoWatcher
import pathspec


class IndexEntry(SQLModel, table=True):
    __tablename__ = "index_entries"

    path: str = Field(primary_key=True)
    type: str = Field(default="file")
    description: str = Field(default="")
    propagation_count: int = Field(default=4)
    updated_at: int = Field(default_factory=lambda: int(time.time()))


class IndexMeta(SQLModel, table=True):
    __tablename__ = "index_meta"

    key: str = Field(primary_key=True)
    value: str = Field(default="")


class TreeNode:
    """A node in the in-memory tree for rendering, with name, comment, and children."""

    def __init__(self, name: str, is_dir: bool = False, comment: str = "",
                 children: list[TreeNode] | None = None,
                 metadata: dict[str, Any] | None = None):
        self.name = name
        self.is_dir = is_dir
        self.comment = comment
        self.children = children or []
        self.metadata = metadata or {}


def walk_dir(
    dir_path: Path,
    *,
    depth: int | None = None,
    current_depth: int = 0,
    filter_fn: Callable[[Path], bool] | None = None,
) -> TreeNode | None:
    """Walk a directory and return a TreeNode with its children.

    Sets ``metadata["abs_path"]``. Does NOT access the database."""

    if filter_fn is not None and filter_fn(dir_path):
        return None

    node = TreeNode(name=dir_path.name or str(dir_path), is_dir=True,
                    metadata={"abs_path": str(dir_path)})

    if depth is not None and current_depth >= depth:
        return node

    try:
        entries = sorted(dir_path.iterdir())
    except OSError:
        return node

    for entry in entries:
        if entry.is_dir():
            child = walk_dir(entry, depth=depth,
                             current_depth=current_depth + 1,
                             filter_fn=filter_fn)
            if child is not None:
                node.children.append(child)
        elif entry.is_file():
            child = walk_file(entry, filter_fn=filter_fn)
            if child is not None:
                node.children.append(child)

    return node


def walk_file(
    file_path: Path,
    *,
    filter_fn: Callable[[Path], bool] | None = None,
) -> TreeNode | None:
    """Walk a file and return a TreeNode with symbol children.

    Routes to type-specific walkers based on file extension.
    Sets ``metadata["abs_path"]``. Does NOT access the database."""

    if filter_fn is not None and filter_fn(file_path):
        return None

    suffix = file_path.suffix
    if suffix == ".py":
        return _walk_python_file(file_path)
    elif suffix in (".md", ".mdx", ".markdown"):
        return _walk_markdown_file(file_path)
    else:
        return _walk_generic_file(file_path)


def _walk_python_file(file_path: Path) -> TreeNode:
    """Walk a Python file and return a TreeNode with symbol children.

    TODO: implement Python symbol parsing."""
    return TreeNode(name=file_path.name, is_dir=False,
                    metadata={"abs_path": str(file_path)})


def _walk_markdown_file(file_path: Path) -> TreeNode:
    """Walk a Markdown file and return a TreeNode with heading children.

    TODO: implement Markdown heading parsing."""
    return TreeNode(name=file_path.name, is_dir=False,
                    metadata={"abs_path": str(file_path)})


def _walk_generic_file(file_path: Path) -> TreeNode:
    """Walk a generic file and return a TreeNode."""
    return TreeNode(name=file_path.name, is_dir=False,
                    metadata={"abs_path": str(file_path)})


def build_tree(
    base_path: str = "",
    *,
    depth: int | None = None,
    filter_fn: Callable[[Path], bool] | None = None,
) -> TreeNode | None:
    """Walk the filesystem under *base_path*, return a ``TreeNode``.

    Does NOT access the database."""

    full = Path(base_path).resolve() if base_path else Path(".").resolve()
    if not full.is_dir():
        return None

    return walk_dir(full, depth=depth, filter_fn=filter_fn)


def _render_tree(node: TreeNode) -> str:
    """Render a TreeNode tree as an ASCII tree.

    Pure output. Uses ``node.comment`` for the trailing annotation."""
    if node is None:
        return "(empty)"

    def _render(n: TreeNode, prefix: str = "", is_last: bool = True,
                is_root: bool = True) -> str:
        output = ""

        if is_root:
            pointer = ""
            current_prefix = ""
        else:
            pointer = "└── " if is_last else "├── "
            current_prefix = prefix + pointer

        suffix = "/" if n.is_dir else ""
        comment = f"  # {n.comment}" if n.comment else ""

        if is_root:
            output += f"{n.name}{suffix}{comment}\n"
        else:
            output += f"{current_prefix}{n.name}{suffix}{comment}\n"

        if n.children:
            sorted_children = sorted(
                n.children,
                key=lambda c: (0 if c.is_dir else 1, c.name.lower())
            )

            next_prefix = prefix if is_root else prefix + ("    " if is_last else "│   ")

            num_children = len(sorted_children)
            for index, child in enumerate(sorted_children):
                is_child_last = (index == num_children - 1)
                output += _render(
                    child,
                    prefix=next_prefix,
                    is_last=is_child_last,
                    is_root=False
                )

        return output

    return _render(node)


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

        stmt = sqlite_insert(IndexEntry).values(
            path=p,
            type=type or "file",
            description=description or "",
            updated_at=now,
        )

        update_cols: set[str] = {"updated_at"}
        if type is not None:
            update_cols.add("type")
        if description is not None:
            update_cols.add("description")
            update_cols.add("propagation_count")

        set_vals = {c: getattr(stmt.excluded, c) for c in update_cols}

        sess.exec(stmt.on_conflict_do_update(
            index_elements=[IndexEntry.path],
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
            delete(IndexEntry).where(
                (IndexEntry.path == clean) |
                (IndexEntry.path.startswith(clean + "/")) |
                (IndexEntry.path.startswith(clean + ":"))
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
                sql_delete(IndexEntry).where(
                    (IndexEntry.path == path) |
                    (IndexEntry.path.startswith(path + "/")) |
                    (IndexEntry.path.startswith(path + ":"))
                )
            )

        if _close:
            sess.commit()
            sess.close()
        return list(removed)

    def _parse_file_tree(
        self,
        file_path: str,
    ) -> list[IndexEntry]:
        """Parse *file_path* and return symbol ``IndexEntry`` objects.

        TODO: implement language-specific parsing.
        """
        return []

    def parse_file_diff(
        self,
        file_path: str,
        repo_watcher: RepoWatcher,
        old_hash: str,
        new_hash: str,
    ) -> list[IndexEntry]:
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
                    sql_delete(IndexEntry).where(IndexEntry.path == d.path)
                )

            # Decrement file counter; remove when exhausted
            entry = sess.get(IndexEntry, old_path)
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
            entry = sess.get(IndexEntry, dir_path)
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
                return pathspec.PathSpec.from_lines("gitwildmatch", f)
        except FileNotFoundError:
            return None

    def _make_tree_filter(self) -> Callable[[Path], bool]:
        spec = self._load_gitignore_spec()
        base = self._base_dir

        def filter_fn(abs_path: Path) -> bool:
            if spec is not None:
                try:
                    rel = str(abs_path.relative_to(base))
                except ValueError:
                    return False
                if spec.match_file(rel):
                    return True
            return False

        return filter_fn

    def _load_descriptions(self) -> dict[str, str]:
        sess = self._get_session()
        try:
            entries = sess.exec(select(IndexEntry)).all()
            return {e.path: e.description for e in entries if e.description}
        finally:
            sess.close()

    @staticmethod
    def _format_comments(node: TreeNode, descs: dict[str, str], base: Path) -> None:
        """Walk the tree and set ``comment`` on each node from descriptions.

        Derives the relative path from ``metadata["abs_path"]`` and *base*."""
        abs_path = node.metadata.get("abs_path", "")
        try:
            rel = str(Path(abs_path).relative_to(base))
        except ValueError:
            rel = abs_path
        parts: list[str] = []
        desc = descs.get(rel, "")
        if desc:
            parts.append(desc)
        node_type = node.metadata.get("type", "")
        if node_type:
            parts.append(f"[{node_type}]")
        node.comment = " ".join(parts) if parts else ""

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
        full_path = str(self._base_dir / path) if path else str(self._base_dir)
        root = build_tree(full_path, depth=depth, filter_fn=filter_fn)
        if root is None:
            return "(empty)"

        descs = self._load_descriptions()
        self._format_comments(root, descs, self._base_dir)
        return _render_tree(root)
