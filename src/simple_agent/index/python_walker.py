"""Python file walker — extracts symbols using tree-sitter."""

from __future__ import annotations

from pathlib import Path

from simple_agent.index.tree import TreeNode

# ---------------------------------------------------------------------------
# tree-sitter Python parser (initialised once at module load)
# ---------------------------------------------------------------------------

try:
    import tree_sitter_python as tspython
    from tree_sitter import Language, Parser

    _PY_LANG = Language(tspython.language())
    _PY_PARSER = Parser(_PY_LANG)
except ImportError:
    _PY_LANG = None
    _PY_PARSER = None


def _walk_python_file(
    file_path: Path,
    *,
    depth: int | None = None,
    current_depth: int = 0,
) -> TreeNode:
    """Walk a Python file and return a TreeNode with symbol children.

    Uses tree-sitter to parse the file and extract ``class_definition``
    and ``function_definition`` nodes as child TreeNodes.  Respects
    *depth* — when ``current_depth >= depth``, returns a leaf node.
    """
    node = TreeNode(name=file_path.name, is_dir=False,
                    metadata={"abs_path": str(file_path)})

    if depth is not None and current_depth >= depth:
        return node

    if _PY_PARSER is None:
        return node

    try:
        source = file_path.read_bytes()
    except OSError:
        return node

    tree = _PY_PARSER.parse(source)
    root = tree.root_node

    def _walk_ast(ast_node, sym_depth: int) -> None:
        for child in ast_node.children:
            symbol = None

            if child.type == "class_definition":
                cls_name = child.child_by_field_name("name")
                name = cls_name.text.decode() if cls_name else "class"
                body = child.child_by_field_name("body")
                symbol = TreeNode(name=name, is_dir=False)
                if body is not None and (depth is None or sym_depth < depth):
                    _walk_body(body, sym_depth + 1, parent=symbol)

            elif child.type == "function_definition":
                func_name = child.child_by_field_name("name")
                name = (func_name.text.decode() + "()") if func_name else "func()"
                symbol = TreeNode(name=name, is_dir=False)

            elif child.type == "decorated_definition":
                for inner in child.children:
                    if inner.type in ("class_definition", "function_definition"):
                        _walk_ast(inner, sym_depth)
                        continue

            if symbol is not None:
                node.children.append(symbol)

    def _walk_body(body_node, sym_depth: int, parent: TreeNode) -> None:
        for child in body_node.children:
            if child.type == "function_definition":
                func_name = child.child_by_field_name("name")
                name = (func_name.text.decode() + "()") if func_name else "func()"
                s = TreeNode(name=name, is_dir=False)
                parent.children.append(s)
                if depth is None or sym_depth < depth:
                    nested_body = child.child_by_field_name("body")
                    if nested_body is not None:
                        _walk_body(nested_body, sym_depth + 1, parent=s)
            elif child.type == "class_definition":
                cls_name = child.child_by_field_name("name")
                name = cls_name.text.decode() if cls_name else "class"
                body = child.child_by_field_name("body")
                cls_node = TreeNode(name=name, is_dir=False)
                parent.children.append(cls_node)
                if body is not None and (depth is None or sym_depth < depth):
                    _walk_body(body, sym_depth + 1, parent=cls_node)

    _walk_ast(root, 0)
    return node
