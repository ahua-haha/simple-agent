"""Compatibility exports for task lifecycle modules."""

from simple_agent.task_manager.base_lifecycle import (
    USER_TASK_SYSTEM_PROMPT,
    BaseTaskLifecycle,
    SessionState,
    TaskLifecycleError,
)
from simple_agent.task_manager.task_lifecycle import CommonTaskLifecycle, USER_TASK_COMPACT_SYSTEM_PROMPT, USER_TASK_INDEX_MEMORY_SYSTEM_PROMPT

__all__ = [
    "BaseTaskLifecycle",
    "SessionState",
    "TaskLifecycleError",
    "USER_TASK_COMPACT_SYSTEM_PROMPT",
    "USER_TASK_INDEX_MEMORY_SYSTEM_PROMPT",
    "USER_TASK_SYSTEM_PROMPT",
    "CommonTaskLifecycle",
]
