# Handle Compact Design

## Goal

Implement `SessionRunner.handle_compact()` so it compacts finished todo work
inside the current user task while preserving any active or unfinished todo
context after the compact scope.

The compact operation uses an agent with compact-only tools. The agent produces
one compacted todo task with a description and selected useful tool-call
references. The runner then replaces the scoped todo tasks and corresponding
runner messages in one transaction.

## Message Boundary Tracking

Each todo task owns an inclusive message range:

- `message_start_seq: int | None`
- `message_end_seq: int | None`

Task tool calls cannot assign these values immediately because runner messages
are persisted at `turn_end`. Instead, task tools record pending task-boundary
events in memory:

- `create_todo` records a pending `start` boundary for the created todo and its
  tool call ID.
- `finish_todo` records a pending `end` boundary for the finished todo and its
  tool call ID.
- `error_todo` records a pending `end` boundary for the errored todo and its
  tool call ID.

At `turn_end`, the runner appends the assistant message and tool-result
messages. After those messages receive DB sequence numbers, the runner resolves
pending boundaries:

- start boundary: set `message_start_seq` to the assistant message seq that
  contains the task tool call
- end boundary: set `message_end_seq` to the tool-result message seq for the
  finishing or erroring task tool call

This makes task/message ownership explicit and avoids later inference from
message content.

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

1. Compute `start_seq = start_todo.message_start_seq`.
2. Delete all runner messages where `seq >= start_seq`.
3. Insert compact message or messages starting at `start_seq`.
4. Insert preserved active/uncompacted messages after the compact messages.

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

If a todo in the compact scope lacks `message_start_seq` or `message_end_seq`,
compact fails clearly and the runner error path persists the failure.

If DB replacement fails, the transaction rolls back and in-memory runner/task
state remains unchanged.

If the compacting agent creates no compacted todo or more than one compacted
todo, compact fails clearly.

## Testing

Focused tests should cover:

- task tool calls record todo `message_start_seq` and `message_end_seq`
- compact scope selects first todo through latest finished todo
- active todo after latest finished todo is preserved
- compact agent receives only compact task tools
- message tail replacement deletes from `start_seq` and inserts compact plus
  preserved messages
- task replacement deletes scoped todos and inserts one compacted todo
- failed DB replacement leaves old messages/tasks intact

## Out Of Scope

This design does not add coding tools to the compacting agent.

This design does not use inference from message content to discover task
boundaries.
