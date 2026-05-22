"""AgentIndex — persistent tree-structured project index for agent memory."""

from __future__ import annotations

import os
import time

from typing import Any
from sqlmodel import SQLModel, Field, Session, create_engine, select

from simple_agent.snapshot.ghost_indexer import RepoWatcher


class IndexEntry(SQLModel, table=True):
    __tablename__ = "index_entries"

    path: str = Field(primary_key=True)
    type: str = Field(default="file")
    description: str = Field(default="")
    propagation_count: int = Field(default=4)
    line_start: int | None = Field(default=None)
    line_end: int | None = Field(default=None)
    updated_at: int = Field(default_factory=lambda: int(time.time()))


class IndexMeta(SQLModel, table=True):
    __tablename__ = "index_meta"

    key: str = Field(primary_key=True)
    value: str = Field(default="")


class TreeNode:
    """A node in the in-memory tree for rendering, with name, description, and children."""

    def __init__(self, name: str, is_dir: bool = False, description: str = "",
                 children: list[TreeNode] | None = None):
        self.name = name
        self.is_dir = is_dir
        self.description = description
        self.children = children or []


class AgentIndex:
    """Tree-structured index of project files, folders, and symbols.

    One index per repo, stored at ./data/agent_index.db.
    """

    def __init__(self, db_path: str = "./data/agent_index.db"):
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
        line_start: int | None = None,
        line_end: int | None = None,
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
            line_start=line_start,
            line_end=line_end,
            updated_at=now,
        )

        update_cols: set[str] = {"updated_at"}
        if type is not None:
            update_cols.add("type")
        if description is not None:
            update_cols.add("description")
            update_cols.add("propagation_count")
        if line_start is not None:
            update_cols.add("line_start")
        if line_end is not None:
            update_cols.add("line_end")

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

    def _walk_dir(self, dir_path: str, base: str, descs: dict[str, str],
                   depth: int | None, current_depth: int) -> TreeNode:
        """Walk a directory and return a TreeNode with its children."""
        rel = os.path.relpath(dir_path, base)
        name = os.path.basename(dir_path)
        node = TreeNode(name=name, is_dir=True, description=descs.get(rel, ""))

        if depth is not None and current_depth >= depth:
            return node

        try:
            names = sorted(os.listdir(dir_path))
        except OSError:
            return node

        for n in names:
            if n.startswith(".") or n == "__pycache__":
                continue
            full = os.path.join(dir_path, n)
            if os.path.isdir(full):
                node.children.append(self._walk_dir(full, base, descs, depth, current_depth + 1))
            elif os.path.isfile(full):
                node.children.append(self._walk_file(full, base, descs))

        return node

    def _walk_file(self, file_path: str, base: str, descs: dict[str, str]) -> TreeNode:
        """Walk a file and return a TreeNode with symbol children."""
        rel = os.path.relpath(file_path, base)
        name = os.path.basename(file_path)
        node = TreeNode(name=name, is_dir=False, description=descs.get(rel, ""))

        for sym in self._parse_file_tree(rel):
            sym_rel = rel + ":" + sym.name
            sym_node = TreeNode(name=sym.name, is_dir=False,
                                description=descs.get(sym_rel, sym.description))
            node.children.append(sym_node)

        return node

    def _build_tree(self, base_path: str, depth: int | None = None) -> TreeNode | None:
        """Walk the filesystem under *base_path*, attach descriptions from DB,
        and return a ``TreeNode`` for rendering."""
        base = base_path.rstrip("/") if base_path else "."
        if not os.path.isdir(base):
            return None

        # Collect all DB descriptions in one query
        sess = self._get_session()
        descs: dict[str, str] = {}
        try:
            entries = sess.exec(select(IndexEntry)).all()
            for e in entries:
                if e.description:
                    descs[e.path] = e.description
        finally:
            sess.close()

        root = self._walk_dir(base, base, descs, depth, 0)
        # Root description uses the full path argument, not the relative path
        root.description = descs.get(base_path.rstrip("/"), root.description)
        return root

    @staticmethod
    def _parse_diff_ranges(diff_text: str) -> list[tuple[int, int, int, int]]:
        """Extract changed blocks from unified diff hunk headers.

        Each ``@@ -old_s,old_n +new_s,new_n @@`` header yields
        ``(old_start, old_end, new_start, new_end)``.
        When the count is omitted (e.g. ``@@ -89 +91,8 @@``) it defaults to 1.
        Hunks with old-count of 0 (pure additions) are skipped.
        """
        import re
        ranges: list[tuple[int, int, int, int]] = []
        for m in re.finditer(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@', diff_text):
            old_s = int(m.group(1))
            old_n = int(m.group(2)) if m.group(2) else 1
            new_s = int(m.group(3))
            new_n = int(m.group(4)) if m.group(4) else 1
            if old_n > 0:
                ranges.append((old_s, old_s + old_n - 1, new_s, new_s + new_n - 1))
        return ranges

    @staticmethod
    def _ranges_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
        """Check whether two line intervals intersect."""
        return a_start <= b_end and b_start <= a_end

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

    def _reset_file_ranges(
        self,
        file_path: str,
        repo_watcher: RepoWatcher,
        old_hash: str,
        new_hash: str,
        _session: Session,
    ) -> str | None:
        """Reset symbol entries for one modified file.

        Deletes old entries overlapping diff hunks, transfers descriptions
        from old to new by matching path, then replaces all old entries
        with the regenerated tree.

        Returns *file_path* if the file description should be cleared
        (overlapping entries exceeded threshold), or ``None``.
        """
        # 1. Get old entries and hunk ranges
        old_entries = _session.exec(
            select(IndexEntry).where(
                IndexEntry.path.startswith(file_path + ":")
            )
        ).all()

        diff_text = repo_watcher.get_file_diff(old_hash, new_hash, file_path, context_lines=0)
        ranges = self._parse_diff_ranges(diff_text)

        # 2. Count overlapping entries
        overlap_count = 0
        if ranges:
            for old in old_entries:
                if old.line_start is None or old.line_end is None:
                    continue
                for old_s, old_e, _new_s, _new_e in ranges:
                    if self._ranges_overlap(old.line_start, old.line_end, old_s, old_e):
                        overlap_count += 1
                        break

        # 3. Transfer descriptions from old to new by matching path
        old_by_path: dict[str, str] = {}
        for old in old_entries:
            if old.description:
                old_by_path[old.path] = old.description

        new_entries = self._parse_file_tree(file_path)
        for new in new_entries:
            if new.path in old_by_path:
                new.description = old_by_path[new.path]

        # 4. Delete all old entries, insert new ones
        from sqlalchemy import delete as sql_delete
        _session.exec(
            sql_delete(IndexEntry).where(
                IndexEntry.path.startswith(file_path + ":")
            )
        )
        for new in new_entries:
            _session.add(new)

        # Return file path if enough entries changed
        total = len(old_entries)
        if total == 0 or (total > 0 and overlap_count / total >= 0.3):
            return file_path
        return None

    def _handle_modified(
        self,
        changes: list[tuple[str, str, str | None]],
        repo_watcher: RepoWatcher,
        old_hash: str,
        new_hash: str,
        _session: Session | None = None,
    ) -> list[str]:
        """Handle modified files: run ``_reset_file_ranges`` for each file
        and collect paths whose descriptions should be cleared.

        Returns the collected paths for propagation.
        """
        _close = _session is None
        sess = _session or self._get_session()

        from sqlalchemy import update as sql_update

        collected: list[str] = []

        for status, old_path, _ in changes:
            if status != "M":
                continue

            if sess.get(IndexEntry, old_path) is None:
                continue

            result = self._reset_file_ranges(old_path, repo_watcher, old_hash, new_hash, _session=sess)
            if result is not None:
                collected.append(result)

        for path in collected:
            sess.exec(
                sql_update(IndexEntry)
                .where(IndexEntry.path == path)
                .values(description="")
            )

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
    ) -> None:
        """Group changes by ancestor directory and decrement counters.

        Each directory's ``propagation_count`` is reduced by the number
        of changes under it.  When a counter hits zero or below, the
        description is cleared, the counter reset, and the overflow
        cascades to the grandparent.
        """
        _close = _session is None
        sess = _session or self._get_session()

        # Count changes per ancestor directory
        dir_counts: dict[str, int] = {}
        for path in paths:
            parent = self._get_parent(path)
            while parent is not None:
                dir_counts[parent] = dir_counts.get(parent, 0) + 1
                parent = self._get_parent(parent)

        # Decrement each directory's counter
        for dir_path, count in dir_counts.items():
            entry = sess.get(IndexEntry, dir_path)
            if entry is None:
                continue
            entry.propagation_count -= count
            if entry.propagation_count <= 0:
                if entry.description:
                    entry.description = ""
                entry.propagation_count = 4

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

    def tree(
        self,
        path: str = "",
        depth: int | None = None,
    ) -> str:
        """Render the index as a tree with # descriptions.

        Structure comes from the filesystem; descriptions from the database.
        """
        root = self._build_tree(path, depth=depth)

        if root is None:
            return "(empty)"

        return self.render_tree(root)

    def render_tree(self, node: TreeNode, prefix: str = "", is_last: bool = True, is_root: bool = True) -> str:
        """Recursively renders a TreeNode structure using terminal-style graphics."""
        output = ""

        if is_root:
            pointer = ""
            current_prefix = ""
        else:
            pointer = "└── " if is_last else "├── "
            current_prefix = prefix + pointer

        suffix = "/" if node.is_dir else ""
        comment = f"  # {node.description}" if node.description else ""

        if is_root:
            output += f"{node.name}{suffix}{comment}\n"
        else:
            output += f"{current_prefix}{node.name}{suffix}{comment}\n"

        if node.children:
            sorted_children = sorted(
                node.children,
                key=lambda c: (0 if c.is_dir else 1, c.name.lower())
            )

            next_prefix = prefix if is_root else prefix + ("    " if is_last else "│   ")

            num_children = len(sorted_children)
            for index, child in enumerate(sorted_children):
                is_child_last = (index == num_children - 1)
                output += self.render_tree(
                    child,
                    prefix=next_prefix,
                    is_last=is_child_last,
                    is_root=False
                )

        return output
