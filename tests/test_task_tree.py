"""Compatibility tests for the retired task tree runtime."""

from simple_agent.task_manager.models import ManagedTask


def test_replacement_task_model_uses_parent_and_seq():
    task = ManagedTask(
        kind="user_task",
        title="Build feature",
        parent_id=1,
        seq="U",
    )

    assert task.parent_id == 1
    assert task.seq == "U"
