"""Compatibility tests for the retired task tree runtime."""

from simple_agent.task_manager.models import ManagedTask, TaskItem


def test_replacement_task_model_uses_ordered_items():
    task = ManagedTask(
        kind="user_task",
        title="Build feature",
        items=[TaskItem(kind="tool_call", ref_id=1), TaskItem(kind="task", ref_id=2)],
    )

    assert [(item.kind, item.ref_id) for item in task.items] == [
        ("tool_call", 1),
        ("task", 2),
    ]
