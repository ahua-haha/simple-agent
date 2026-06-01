# Task Manager Replacement Design

## Goal

Replace the current task-tree-oriented agent control flow with a simpler split:

- `AgentRuntime` owns only the message queue, model loop, generic tool execution, and event emission.
- `TaskManager` owns user tasks, agent-defined todos, task/tool-call ordering, and explicit compaction operations.

The runtime must not import task-manager models or make task lifecycle decisions. The agent manages todos by calling task tools, and ordinary tool calls are recorded into the currently active task context by the tool layer.

## Current Context

The existing implementation couples task state and runtime flow through `Task`, `CentralControl`, `PlanRunner`, and `ExploreRunner`. Tasks are persisted as a tree with `parent_id`, `running_task_id`, and `finished_task_ids`. Tool calls are already persisted separately as `ToolCallRecord` rows and can be referenced by ID.

The replacement should preserve persisted tool-call records, but remove the task tree as the runtime driver. Task management becomes an explicit service used by tools and orchestration code.

## Architecture

### AgentRuntime

`AgentRuntime` is intentionally generic. It:

- accepts a message queue and tool list
- calls the model
- executes tools
- appends assistant and tool-result messages
- emits runtime events

It does not know about user tasks, todos, compaction, active tasks, or task storage.

### TaskManager

`TaskManager` owns task state. It stores one active user task for a session/run, an ordered task timeline, and the currently active todo if one exists. It exposes Python methods and backs the agent-callable task tools.

`TaskManager` does not decide when compaction happens. It only performs compaction when a caller explicitly invokes the compaction API with the replacement aggregate task.

### Tool Layer

The tool layer prepares tools for `AgentRuntime`.

- Task tools call `TaskManager` methods.
- Normal tools execute normally and persist tool-call records.
- After a normal tool call is persisted, the tool layer calls `TaskManager.record_tool_call(tool_call_id)`.
- `record_tool_call` appends the tool call under the active todo, or under the user task when no todo is active.

This keeps task behavior outside the runtime while still letting the agent manage todos through normal tool calls.

## Data Model

Use one unified task model.

```python
class Task:
    id: int | None
    parent_id: int | None
    kind: Literal["user_task", "todo", "aggregate"]
    title: str
    status: Literal["active", "done", "error"]
    items: list[TaskItem]
    result: str | None
    error: str | None
    created_at: float
    updated_at: float


class TaskItem:
    kind: Literal["task", "tool_call"]
    ref_id: int
```

`items` is the source of truth for visible order. A task can contain both child tasks and direct tool calls in one ordered list.

Example:

```text
UserTask.items:
  - tool_call 1
  - task 10
  - tool_call 2
  - task 11

Todo 10.items:
  - tool_call 3
  - tool_call 4
```

The user task may have direct tool calls. This covers work done before a todo exists, after a todo is finished, or intentionally at the top level.

## Task Lifecycle

There is only one active todo at a time.

- `create_user_task(input)` creates the top-level task for a session/run.
- `create_todo(title)` creates a todo task, appends it to the active user task's `items`, and marks it active.
- `finish_todo(result=None)` marks the active todo as done and clears the active todo pointer.
- `error_todo(error)` marks the active todo as error and clears the active todo pointer.
- `record_tool_call(tool_call_id)` appends the tool-call item to the active todo, or to the user task if there is no active todo.
- `finish_user_task(result=None)` marks the user task done.

Invalid lifecycle transitions should fail with clear errors:

- creating a todo while another todo is active
- finishing or erroring a todo when no todo is active
- creating a second user task while one is active
- recording tool calls before a user task exists

## Agent-Callable Tools

Expose these task tools to the agent:

```text
create_todo(title)
finish_todo(result?)
error_todo(error)
```

These are normal tools from the runtime's point of view. The runtime executes them like any other tool and does not inspect their semantics.

The agent should use `create_todo` when it starts a coherent unit of work, then call `finish_todo` or `error_todo` when that unit is complete.

## Compaction

Compaction is separate from task completion. `finish_todo` only marks lifecycle state.

`compact_items(parent_task_id, item_refs, aggregate_task)` replaces selected visible items under a parent task with one aggregate task.

Rules:

- The caller decides when compaction happens.
- The caller supplies the compacted result.
- The caller supplies the aggregate task's selected tool-call items.
- Selected items must all belong to the same parent task's visible `items` list.
- Active todos cannot be compacted.
- Missing refs and duplicate refs are rejected.
- Source tasks remain persisted for audit/debugging, but are no longer in the parent's visible `items` list.

Example:

```text
Before:
UserTask.items:
  - task 1
  - task 2
  - task 3

After compact:
UserTask.items:
  - task 100

Task 100:
  kind: aggregate
  result: "Aggregated result for task 1, task 2, and task 3."
  items:
    - tool_call 2
    - tool_call 3
```

The design intentionally does not specify what triggers compaction.

## Context Rendering

The task manager should provide a context-rendering method that traverses the visible `items` timeline and returns compact context for the caller. Rendering is separate from runtime execution.

The renderer should preserve item order:

- direct user-task tool calls appear where they occurred
- visible todo tasks appear where they were created
- aggregate tasks replace the source items
- hidden source tasks are omitted from normal context rendering

Detailed source tasks remain available through explicit inspection/debug APIs.

## Persistence

Persist tasks in SQLite alongside existing tool-call records.

Recommended storage:

- add a task table for unified `Task` rows
- store `items` as JSON initially to keep the migration small
- keep using existing tool-call records by ID

If later query requirements grow, `TaskItem` can be normalized into a separate table without changing the runtime boundary.

## Testing

Add focused tests for:

- creating a user task
- creating and finishing one todo
- rejecting a second active todo
- attaching tool calls to the user task when no todo is active
- attaching tool calls to the active todo
- preserving mixed task/tool-call order in `items`
- compacting selected visible items into one aggregate task
- rejecting compaction of active todos, missing refs, and duplicate refs
- running the generic runtime with task tools without importing task manager internals

## Out Of Scope

- deciding when compaction should run
- generating compacted summaries
- nested todos
- multiple active todos
- replacing the existing tool-call persistence format
- UI changes for task visualization
