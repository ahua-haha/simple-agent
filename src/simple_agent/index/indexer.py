"""AgentIndex — persistent tree-structured project index for agent memory."""

from __future__ import annotations

import fnmatch
import os
import time

from typing import Any, Optional
from sqlmodel import SQLModel, Field, Session, create_engine, select

from simple_agent.snapshot.ghost_indexer import RepoWatcher


class IndexEntry(SQLModel, table=True):
    __tablename__ = "index_entries"

    path: str = Field(primary_key=True)
    parent_path: str = Field(index=True, default="")
    name: str = Field(index=True)
    type: str = Field(default="file")
    description: str = Field(default="")
    line_start: int | None = Field(default=None)
    line_end: int | None = Field(default=None)
    updated_at: int = Field(default_factory=lambda: int(time.time()))


class IndexMeta(SQLModel, table=True):
    __tablename__ = "index_meta"

    key: str = Field(primary_key=True)
    value: str = Field(default="")


class TreeNode:
    """A node in the in-memory index tree, holding an entry and its children."""

    def __init__(self, entry: IndexEntry, children: list[TreeNode] | None = None):
        self.entry = entry
        self.children = children or []

    @property
    def is_dir(self) -> bool:
        return self.entry.type in ("directory", "folder")


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

    def _derive_parent_and_name(self, path: str) -> tuple[str, str]:
        """Derive parent_path and name from a path.

        File example: 'src/process/file.py' → ('src/process', 'file.py')
        Symbol example: 'src/process/file.py:ClassName' → ('src/process/file.py', 'ClassName')
        Directory example: 'src/process/' → ('src', 'process')
        """
        if ":" in path:
            file_part, symbol = path.rsplit(":", 1)
            return file_part, symbol
        if "/" in path:
            parent, name = path.rsplit("/", 1)
            return parent, name
        return "", path

    def _ensure_parents(self, path: str, _session: Session | None = None) -> None:
        """Create intermediate folder entries for all ancestors of the given path."""
        if not path:
            return
        _close = _session is None
        sess = _session or self._get_session()
        parts = path.split("/")
        for i in range(1, len(parts) + 1):
            ancestor = "/".join(parts[:i])
            existing = sess.exec(
                select(IndexEntry).where(IndexEntry.path == ancestor)
            ).first()
            if not existing:
                parent_path, name = self._derive_parent_and_name(ancestor)
                sess.add(IndexEntry(
                    path=ancestor, parent_path=parent_path, name=name, type="directory", description="",
                ))
        if _close:
            sess.commit()
            sess.close()

    def update(
        self,
        path: str,
        type: str = "file",
        description: str = "",
        line_start: int | None = None,
        line_end: int | None = None,
        _session: Session | None = None,
    ) -> None:
        """Upsert a single index entry. Delegates to :meth:`update_batch`."""
        self.update_batch(
            [IndexEntry(
                path=path, type=type, description=description,
                line_start=line_start, line_end=line_end,
            )],
            _session=_session,
        )

    def update_batch(
        self,
        entries: list[IndexEntry],
        fields: list | None = None,
        _session: Session | None = None,
    ) -> None:
        """Upsert multiple index entries in a single SQL statement.

        *fields* specifies which columns to update on conflict, using the
        ``IndexEntry`` column attributes (e.g. ``[IndexEntry.description]``).
        If ``None``, all columns are updated. ``updated_at`` is always updated.
        """
        if not entries:
            return
        _close = _session is None
        sess = _session or self._get_session()

        from sqlalchemy.dialects.sqlite import insert as sqlite_insert
        from sqlalchemy.orm.attributes import InstrumentedAttribute
        # Resolve which fields get updated on conflict
        _update_keys: set[InstrumentedAttribute] | None = {IndexEntry.updated_at}
        for f in fields:
            _update_keys.add(f)

        t = int(time.time())
        rows: list[dict[str, Any]] = []
        for e in entries:
            p = e.path.rstrip("/")
            parent_path, name = self._derive_parent_and_name(p)
            self._ensure_parents(parent_path, _session=sess)
            rows.append(dict(
                path=p, parent_path=parent_path, name=name,
                type=e.type, description=e.description,
                line_start=e.line_start, line_end=e.line_end,
                updated_at=t,
            ))

        ins = sqlite_insert(IndexEntry).values(rows)

        set_vals: dict = {}
        for col in _update_keys:
            set_vals[col.name] = getattr(ins.excluded, col.name)

        stmt = ins.on_conflict_do_update(
            index_elements=[IndexEntry.path],
            set_=set_vals,
        )
        sess.exec(stmt)

        if _close:
            sess.commit()
            sess.close()

    def remove(self, path: str, _session: Session | None = None) -> None:
        """Remove an entry and all its descendants."""
        _close = _session is None
        sess = _session or self._get_session()
        clean = path.rstrip("/")
        entries = sess.exec(
            select(IndexEntry).where(
                (IndexEntry.path == clean) |
                (IndexEntry.path.startswith(clean + "/")) |
                (IndexEntry.path.startswith(clean + ":"))
            )
        ).all()
        for entry in entries:
            sess.delete(entry)
        if _close:
            sess.commit()
            sess.close()

    def _load_tree(self, root_path: str, _session: Session | None = None) -> TreeNode | None:
        """Load all descendants of *root_path* in one query, returning the root node."""
        sess = _session or self._get_session()
        if root_path:
            entries = sess.exec(
                select(IndexEntry).where(
                    (IndexEntry.path == root_path) |
                    (IndexEntry.path.startswith(root_path + "/")) |
                    (IndexEntry.path.startswith(root_path + ":"))
                ).order_by(IndexEntry.type.desc(), IndexEntry.name)
            ).all()
        else:
            entries = sess.exec(
                select(IndexEntry).order_by(IndexEntry.type.desc(), IndexEntry.name)
            ).all()
        if _session is None:
            sess.close()

        if not entries and root_path:
            return None

        by_parent: dict[str, list[TreeNode]] = {}
        root_entry: IndexEntry | None = None
        for e in entries:
            node = TreeNode(entry=e)
            by_parent.setdefault(e.parent_path, []).append(node)
            if e.path == root_path:
                root_entry = e

        def _attach(parent: str) -> list[TreeNode]:
            children = by_parent.get(parent, [])
            for child in children:
                child.children = _attach(child.entry.path)
            return children

        if root_path and root_entry:
            root = TreeNode(entry=root_entry)
            root.children = _attach(root_path)
        else:
            root = TreeNode(entry=IndexEntry(path="", name=".", type="directory"))
            root.children = _attach("")

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

    def invalidate_stale(self, file_path: str, diff_text: str, _session: Session | None = None) -> int:
        """Delete symbol entries under *file_path* whose line ranges overlap any diff hunk.

        File-level entries (no line range) are never deleted. Returns the count
        of deleted entries.

        The *diff_text* should be produced with ``-U0`` so each hunk header
        maps exactly to the changed lines with no context padding.
        """
        ranges = self._parse_diff_ranges(diff_text)
        if not ranges:
            return 0

        _close = _session is None
        sess = _session or self._get_session()
        deleted = 0
        entries = sess.exec(
            select(IndexEntry).where(
                IndexEntry.path.startswith(file_path + ":")
            )
        ).all()
        for entry in entries:
            if entry.line_start is None or entry.line_end is None:
                continue
            for old_s, old_e, _new_s, _new_e in ranges:
                if self._ranges_overlap(entry.line_start, entry.line_end, old_s, old_e):
                    sess.delete(entry)
                    deleted += 1
                    break
        if _close:
            sess.commit()
            sess.close()
        return deleted

    def rename(self, old_path: str, new_path: str, _session: Session | None = None) -> int:
        """Rename a file entry and all its symbol children.

        Every entry whose path equals *old_path* or starts with ``old_path:``
        has its path, parent_path, and name updated to reflect *new_path*.
        Returns the count of renamed entries.
        """
        _close = _session is None
        sess = _session or self._get_session()
        old_path = old_path.rstrip("/")
        new_path = new_path.rstrip("/")
        renamed = 0
        entries = sess.exec(
            select(IndexEntry).where(
                (IndexEntry.path == old_path) |
                (IndexEntry.path.startswith(old_path + ":"))
            )
        ).all()
        for entry in entries:
            updated_path = new_path + entry.path[len(old_path):]
            updated_parent, updated_name = self._derive_parent_and_name(updated_path)
            sess.delete(entry)
            sess.add(IndexEntry(
                path=updated_path,
                parent_path=updated_parent,
                name=updated_name,
                type=entry.type,
                description=entry.description,
                line_start=entry.line_start,
                line_end=entry.line_end,
                updated_at=int(time.time()),
            ))
            renamed += 1
        if _close:
            sess.commit()
            sess.close()
        return renamed

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
            # Phase 1: renames (status is like "R100", "R087")
            for status, old, new in changes:
                if status.startswith("R") and new is not None:
                    self.rename(old, new, _session=session)
                    processed += 1

            # Phase 2: deletes
            for status, old, _new in changes:
                if status == "D":
                    self.remove(old, _session=session)
                    processed += 1

            # Phase 3: modifications
            for status, old, _new in changes:
                if status == "M":
                    diff_text = repo_watcher.get_file_diff(old_hash, new_hash, old, context_lines=0)
                    self.invalidate_stale(old, diff_text, _session=session)
                    processed += 1

            self._set_hash(new_hash, _session=session)
            session.commit()
        finally:
            session.close()
        return processed

    def tree(
        self,
        path: str = "",
        depth: int | None = None,
        filter: str | None = None,
        pattern: str | None = None,
        prune: bool = False,
        type: str | None = None,
    ) -> str:
        """Render the index as a tree with # descriptions.

        Args:
            path: Root path for subtree rendering.
            depth: Maximum recursion depth.
            filter: Case-insensitive substring match on entry name.
            pattern: Glob pattern match on entry name (like ``tree -P``).
            prune: When True with pattern/type, hide directories that contain
                   no matching entries (like ``tree --prune``).
            type: Only show entries of this type (e.g. ``"file"``, ``"function"``).
        """
        root_path = path.rstrip("/") if path else ""
        root = self._load_tree(root_path)

        if root is None:
            return "(empty)"

        root = self.filter_tree(root, pattern, depth)
        if root is None:
            return "(empty)"

        return self.render_tree(root)

    def filter_tree(
        self,
        node: TreeNode, 
        pattern: Optional[str] = None, 
        max_depth: Optional[int] = None, 
        _current_relative_depth: int = 0
    ) -> Optional[TreeNode]:
        """
        Filters an in-memory TreeNode system uniformly based on name patterns and depth bounds.
        
        Args:
            node: The tree node to evaluate.
            pattern: Wildcard glob pattern (e.g., "*.py", "auth*").
            max_depth: Maximum relative depth to traverse down from the starting node.
        """
        # Rule 1: Depth Boundary Check
        if max_depth is not None and _current_relative_depth > max_depth:
            return None

        # Rule 2: Uniform Pattern Verification (Checks every node type)
        node_self_matches = True
        if pattern is not None:
            node_self_matches = fnmatch.fnmatch(node.entry.name, pattern)

        # Rule 3: Traversal Logic for Directories / Inner Nodes
        if not node.children:
            if node_self_matches:
                return TreeNode(node.entry, [])
            return None

        filtered_children = []
        
        # Only descend if we are strictly below the max depth ceiling
        if max_depth is None or _current_relative_depth < max_depth:
            for child in node.children:
                filtered_child = self.filter_tree(
                    child, pattern, max_depth, _current_relative_depth + 1
                )
                if filtered_child is not None:
                    filtered_children.append(filtered_child)
        if filtered_children:
            return TreeNode(node.entry, filtered_children)
        if node_self_matches:
            return TreeNode(node.entry, [])
        # Directory failed the pattern and has no matching sub-elements
        return None

    def render_tree(self, node: TreeNode, prefix: str = "", is_last: bool = True, is_root: bool = True) -> str:
        """
        Recursively renders a TreeNode structure using terminal-style graphics.
        
        Args:
            node: The current TreeNode to render.
            prefix: Accumulator string tracking vertical ancestry lines (internal use).
            is_last: Boolean indicating if this node is the last sibling in its layer.
            is_root: Boolean flag to suppress connector symbols on the very top node.
        """
        output = ""

        # 1. Determine the structural connectors
        if is_root:
            # The root node (base directory/project scope) has no prefix bars
            pointer = ""
            current_prefix = ""
        else:
            pointer = "└── " if is_last else "├── "
            current_prefix = prefix + pointer

        # 2. Format names with trailing slash for directories to match standard convention
        suffix = "/" if node.is_dir else ""
        
        # Inline code comment optimization for agent readability
        comment = f"  # {node.entry.description}" if node.entry.description else ""
        
        # Append the line for the current node
        if is_root:
            output += f"{node.entry.name}{suffix}{comment}\n"
        else:
            output += f"{current_prefix}{node.entry.name}{suffix}{comment}\n"

        # 3. Process and render children nodes
        if node.children:
            # Sort children: directories/folders always float to the top alphabetically, 
            # followed by files, classes, and functions.
            type_priority = {"directory": 0, "folder": 0, "dir": 0, "file": 1, "class": 2, "function": 3}
            
            sorted_children = sorted(
                node.children,
                key=lambda c: (type_priority.get(c.entry.type, 99), c.entry.name.lower())
            )

            # Calculate the next structural prefix layer for the children lines
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
