"""Tests for task tree rendering."""

from __future__ import annotations

from simple_agent.task_manager.models import ToolCallTask, CommonTask
from simple_agent.task_manager.review import TaskTreeRenderer, build_task_tree


def test_task_tree_renderer_uses_tool_call_metadata_from_task():
    root = CommonTask(id=1, title="Build feature")
    root.children.append(
        ToolCallTask(
            id=2,
            parent_id=1,
            status="done",
            tool_call_log_id=7,
            tool_call_name="index_upsert",
            tool_call_args={"path_id": "src/app.py"},
        )
    )

    output = TaskTreeRenderer(
        format="tree",
        depth=None,
    ).render(root)

    assert "- user_task [active] Build feature" in output
    assert '- tool_call 1. index_upsert args: {"path_id":"src/app.py"}' in output


def test_task_tree_renderer_flat_mode_does_not_mutate_tree():
    root = CommonTask(id=1, title="Build feature")
    task = CommonTask(id=2, parent_id=1, title="Inspect files")
    tool_call = ToolCallTask(
        id=3,
        parent_id=2,
        status="done",
        tool_call_log_id=1,
        tool_call_name="ls",
        tool_call_args={"path": "."},
    )
    task.children.append(tool_call)
    root.children.append(task)
    original_children = list(root.children)
    original_task_children = list(task.children)

    output = TaskTreeRenderer(format="flat", depth=None).render(root)

    assert "- user_task [active] Build feature" in output
    assert '- tool_call 1. ls args: {"path":"."}' in output
    assert root.children == original_children
    assert task.children == original_task_children


def test_build_task_tree_reconstructs_children_from_parent_id():
    root = CommonTask(id=1, title="Build feature")
    task = CommonTask(id=2, parent_id=1, title="Inspect files")
    tool_call = ToolCallTask(id=3, parent_id=2, tool_call_log_id=7, tool_call_name="ls")

    roots = build_task_tree([root, task, tool_call])

    assert roots == [root]
    assert root.children == [task]
    assert task.children == [tool_call]
