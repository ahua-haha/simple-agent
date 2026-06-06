"""Compatibility tests for the retired task tree runtime."""

from simple_agent.task_manager.models import UserTask


def test_replacement_task_model_uses_parent_without_seq():
    task = UserTask(
        title="Build feature",
        parent_id=1,
    )

    assert task.parent_id == 1
    assert not hasattr(task, "seq")
