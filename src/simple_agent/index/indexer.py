"""AgentIndex — persistent tree-structured project index for agent memory."""

from __future__ import annotations

import fnmatch
import os
import time

from typing import Optional
from sqlmodel import SQLModel, Field, Session, create_engine, select


class IndexEntry(SQLModel, table=True):
    __tablename__ = "index_entries"

    path: str = Field(primary_key=True)
    parent_path: str = Field(index=True, default="")
    name: str = Field(index=True)
    type: str = Field(default="file")
    description: str = Field(default="")
    updated_at: int = Field(default_factory=lambda: int(time.time()))


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

    def _ensure_parents(self, path: str) -> None:
        """Create intermediate folder entries for all ancestors of the given path."""
        if not path:
            return
        parts = path.split("/")
        session = self._get_session()
        try:
            for i in range(1, len(parts) + 1):
                ancestor = "/".join(parts[:i])
                existing = session.exec(
                    select(IndexEntry).where(IndexEntry.path == ancestor)
                ).first()
                if not existing:
                    parent_path, name = self._derive_parent_and_name(ancestor)
                    session.add(IndexEntry(
                        path=ancestor, parent_path=parent_path, name=name, type="directory", description="",
                    ))
            session.commit()
        finally:
            session.close()

    def update(self, path: str, type: str = "file", description: str = "") -> None:
        """Upsert an index entry."""
        path = path.rstrip("/")
        parent_path, name = self._derive_parent_and_name(path)
        self._ensure_parents(parent_path)
        with self._get_session() as session:
            existing = session.exec(
                select(IndexEntry).where(IndexEntry.path == path)
            ).first()
            if existing:
                existing.type = type
                existing.description = description
                existing.updated_at = int(time.time())
            else:
                session.add(IndexEntry(
                    path=path,
                    parent_path=parent_path,
                    name=name,
                    type=type,
                    description=description,
                ))
            session.commit()

    def remove(self, path: str) -> None:
        """Remove an entry and all its descendants."""
        clean = path.rstrip("/")
        with self._get_session() as session:
            entries = session.exec(
                select(IndexEntry).where(
                    (IndexEntry.path == clean) |
                    (IndexEntry.path.startswith(clean + "/")) |
                    (IndexEntry.path.startswith(clean + ":"))
                )
            ).all()
            for entry in entries:
                session.delete(entry)
            session.commit()

    def _load_tree(self, root_path: str) -> TreeNode | None:
        """Load all descendants of *root_path* in one query, returning the root node."""
        with self._get_session() as session:
            if root_path:
                entries = session.exec(
                    select(IndexEntry).where(
                        (IndexEntry.path == root_path) |
                        (IndexEntry.path.startswith(root_path + "/")) |
                        (IndexEntry.path.startswith(root_path + ":"))
                    ).order_by(IndexEntry.type.desc(), IndexEntry.name)
                ).all()
            else:
                entries = session.exec(
                    select(IndexEntry).order_by(IndexEntry.type.desc(), IndexEntry.name)
                ).all()

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
