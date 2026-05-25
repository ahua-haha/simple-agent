"""Python file walker — extracts symbols using tree-sitter."""

from __future__ import annotations

from pathlib import Path

from simple_agent.index.tree import TreeNode
import tree_sitter_python as tspython
from tree_sitter import Language, Node, Parser

# ---------------------------------------------------------------------------
# tree-sitter Python parser (initialised once at module load)
# ---------------------------------------------------------------------------

try:
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
    """Walk a Python file and return a TreeNode with symbol children."""
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
    node.children = _walk_module(tree.root_node, depth)
    return node


def _walk_module(module_node: Node, depth: int | None) -> list[TreeNode]:
    """Walk module-level children and return a list of symbol TreeNodes."""
    result: list[TreeNode] = []

    for child in module_node.children:
        symbol = None

        if child.type == "class_definition":
            symbol = _walk_class_definition(child, depth, 0)

        elif child.type == "function_definition":
            symbol = _walk_function_definition(child)

        elif child.type == "decorated_definition":
            for inner in child.children:
                if inner.type == "class_definition":
                    symbol = _walk_class_definition(inner, depth, 0)
                elif inner.type == "function_definition":
                    symbol = _walk_function_definition(inner)

        if symbol is not None:
            result.append(symbol)

    return result


def _walk_class_definition(class_node: Node, depth: int | None, sym_depth: int) -> TreeNode:
    """Walk a class_definition and return a TreeNode with methods as children."""
    cls_name = class_node.child_by_field_name("name")
    name = cls_name.text.decode() if cls_name else "class"
    symbol = TreeNode(name=name, is_dir=False)

    if depth is not None and sym_depth >= depth:
        return symbol

    body = class_node.child_by_field_name("body")
    if body is not None:
        symbol.children = _walk_block(body, depth, sym_depth + 1)

    return symbol


def _walk_function_definition(func_node: Node) -> TreeNode:
    """Walk a function_definition and return a leaf TreeNode."""
    func_name = func_node.child_by_field_name("name")
    name = (func_name.text.decode() + "()") if func_name else "func()"
    return TreeNode(name=name, is_dir=False)


def _walk_block(block_node: Node, depth: int | None, sym_depth: int) -> list[TreeNode]:
    """Walk a block node and return a list of child TreeNodes."""
    result: list[TreeNode] = []

    for child in block_node.children:
        symbol = None

        if child.type == "function_definition":
            func_name = child.child_by_field_name("name")
            name = (func_name.text.decode() + "()") if func_name else "func()"
            symbol = TreeNode(name=name, is_dir=False)

            if depth is None or sym_depth < depth:
                nested_body = child.child_by_field_name("body")
                if nested_body is not None:
                    symbol.children = _walk_block(nested_body, depth, sym_depth + 1)

        elif child.type == "class_definition":
            symbol = _walk_class_definition(child, depth, sym_depth)

        if symbol is not None:
            result.append(symbol)

    return result
