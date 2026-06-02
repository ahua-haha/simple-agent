# Handle Compact Design

## Goal

Implement `SessionRunner.handle_compact()` so it compacts finished todo work
inside the current user task while preserving any active or unfinished todo
context after the compact scope.

The compact operation uses an agent with compact-only tools. The agent produces
one compacted todo task with a description and selected useful tool-call
references. The runner then replaces the scoped todo tasks and corresponding
runner messages in one transaction.

## Message Boundary Lookup

Each todo task stores the tool-call IDs that open and close its message range:

- `create_tool_call_id: str | None`
- `end_tool_call_id: str | None`

`create_tool_call_id` comes from the assistant `create_todo` call that created
the todo. `end_tool_call_id` comes from the `finish_todo` or `error_todo` call
that completed the todo.

The task manager does not store runner-message sequence numbers and does not
resolve message boundaries at `turn_end`. It only mutates in-memory task data.

When compaction starts, `SessionRunner` owns the message list and derives the
message range by scanning assistant messages for `ToolCall` content:

- compact start: assistant message containing the first compacted todo's
  `create_tool_call_id`
- compact end: tool-result message with the latest compacted todo's
  `end_tool_call_id`
- preserved tail: messages after the compact end

This keeps message sequencing in the runner, where messages are stored, and
keeps the task manager focused on task state.

## Compact Scope

`handle_compact()` selects the compact scope from the current user task:

- start todo: first todo task in the current user task
- end todo: latest finished todo task in the current user task

The compact scope is inclusive from start todo through latest finished todo.

If there is no finished todo, compact returns to `running` without replacing
messages or tasks.

If there is an active or unfinished todo after the latest finished todo, that
todo and its effective messages are preserved. Preservation happens by
rebuilding the replacement suffix, not by keeping old DB rows after the compact
start sequence.

## Compact Agent

`handle_compact()` runs a compact-only agent call. The compacting agent does not
receive coding or filesystem tools.

The compacting prompt includes:

- user task title/request
- todos in the compact scope
- runner messages covered by those todos
- tool-call records referenced by those todos
- active or unfinished todo context that must be preserved

The compacting agent receives only compact task tools, such as:

- `create_compacted_todo(description)`
- `record_compacted_tool_call(tool_call_log_id)`
- `finish_compacted_todo()`

These tools mutate an in-memory compact buffer. They do not write to the DB
directly and they do not decide message replacement.

The compact result must be exactly one compacted todo task that contains:

- a description/result summary
- selected useful runner tool-call references

If the compacting agent does not create and finish one compacted todo, compact
fails clearly.

## Replacement Transaction

After the compact agent succeeds, `SessionRunner` applies the replacement in one
database transaction.

Messages use tail replacement:

1. Compute `start_seq` by finding the assistant message with the first compact
   todo's `create_tool_call_id`.
2. Compute `end_seq` by finding the tool-result message with the latest compact
   todo's `end_tool_call_id`.
3. Delete all runner messages where `seq >= start_seq`.
4. Insert compact message or messages starting at `start_seq`.
5. Insert messages after `end_seq` after the compact messages.

Tasks use scoped replacement under the existing user task:

1. Delete todo tasks in the compact scope.
2. Insert the compacted todo task.
3. Preserve active or unfinished todos after the compact scope.
4. Update `user_task.items` to contain the compacted todo plus preserved todos
   in order.

Runner metadata is saved in the same transaction:

- `phase = "running"`
- `status = "running"`
- `active_user_task_id` unchanged

After commit succeeds, the runner updates in-memory state:

- `_messages = messages_before_start + compact_messages + preserved_messages`
- `_phase = "running"`
- `_cancel_event.clear()`

The runner must not update in-memory data before the DB transaction commits.

## Error Handling

If the first compact todo lacks `create_tool_call_id`, if the latest compact
todo lacks `end_tool_call_id`, or if the runner cannot find the corresponding
assistant/tool-result message, compact fails clearly and the runner error path
persists the failure.

If DB replacement fails, the transaction rolls back and in-memory runner/task
state remains unchanged.

If the compacting agent creates no compacted todo or more than one compacted
todo, compact fails clearly.

## Testing

Focused tests should cover:

- task tool calls store todo `create_tool_call_id` and `end_tool_call_id`
- compact scope selects first todo through latest finished todo
- active todo after latest finished todo is preserved
- compact agent receives only compact task tools
- message tail replacement deletes from `start_seq` and inserts compact plus
  preserved messages
- task replacement deletes scoped todos and inserts one compacted todo
- failed DB replacement leaves old messages/tasks intact

## Out Of Scope

This design does not add coding tools to the compacting agent.

This design does not store message sequence boundaries on task records.
