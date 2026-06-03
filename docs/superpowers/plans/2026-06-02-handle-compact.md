# Handle Compact Implementation Plan

> Status: updated after simplifying todo/message boundary logic.

**Goal:** Implement `SessionRunner.handle_compact()` so it replaces finished
todo work with one compacted todo and matching compacted runner messages.

**Architecture:** Todos store `create_tool_call_id` from the assistant
`create_todo` call that created them and `end_tool_call_id` from the
`finish_todo` or `error_todo` call that completed them. `TaskManager` owns
in-memory todo mutation and compact task replacement. Every task row, including
tool-call tasks, uses `parent_id` for tree structure and a string fractional
`seq` key for sibling ordering. `SessionRunner` owns runner messages with string
fractional `seq` keys, finds compact starts by scanning assistant `ToolCall.id`
values, and finds compact ends by scanning tool-result messages for the saved
end tool-call ID.

## Tasks

- [x] Persist `ManagedTask.create_tool_call_id`.
- [x] Store `create_tool_call_id` when `create_todo` runs.
- [x] Persist `ManagedTask.end_tool_call_id`.
- [x] Store `end_tool_call_id` when `finish_todo` or `error_todo` runs.
- [x] Remove pending boundary tracking and message-range fields from
  `TaskManager`.
- [x] Keep `compact_scope()` focused on selecting compacted and preserved
  todos.
- [x] Add `SessionRunner.find_assistant_message_seq_for_tool_call()`.
- [x] Add `SessionRunner.find_tool_result_message_seq_for_tool_call()`.
- [x] Make `handle_compact()` derive message replacement ranges from
  `create_tool_call_id` and `end_tool_call_id` lookup.
- [x] Make `TaskManager.replace_compact_scope()` delete the persisted task tree
  and save the rebuilt tree in one function.
- [x] Make `TaskManager.replace_compact_scope()` compute scope and consume the
  compact buffer itself instead of receiving arguments.
- [x] Store tool calls as `ManagedTask(kind="tool_call")` children that point to
  runner tool-call log rows.
- [x] Use string fractional `seq` keys for runner messages and managed-task
  sibling ordering.
- [x] Update focused task-manager and session-runner tests.

## Verification

Run:

```bash
uv run pytest tests/test_task_manager.py tests/test_session_runner.py -q
```

Expected: all tests pass.
