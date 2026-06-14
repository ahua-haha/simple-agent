# Task Lifecycle Refactor Design

## Goal

Refactor task management so task lifecycle behavior lives with the task type, while `TaskManager` remains responsible for owning the task tree and syncing it with the database.

The task object is the single source of truth for lifecycle data. The lifecycle behavior may use runtime context, but it should not query the runner or database directly.

## Current Problem

`TaskManager` currently owns several responsibilities:

- task tree mutation
- database load/save/sync
- active user task and active todo tracking
- runtime instruction generation
- lifecycle decisions such as when to prompt the agent to create or finish a todo
- agent tool construction
- compaction setup and sync

This makes the manager hard to reason about because task data, task lifecycle behavior, and persistence concerns are mixed together.

## Chosen Approach

Use lifecycle methods directly on concrete task classes.

`UserTask` and `TodoTask` will own task-kind-specific lifecycle behavior. `ToolCallTask` will stay as a plain data record for now and will not have lifecycle behavior.

This is intentionally simpler than a lifecycle registry or separate lifecycle class hierarchy. The project currently has only two active task kinds, and keeping the lifecycle methods on those task classes makes the runtime path easy to follow.

## Responsibilities

### Task Classes

`UserTask` stores user-task data and provides user-task lifecycle behavior:

- generate instruction text when no todo is active
- decide whether the agent should define a todo before more work
- expose small helper decisions based on runtime context

`TodoTask` stores todo data and provides todo lifecycle behavior:

- generate instruction text for focusing on the active todo
- decide whether the agent should check whether the todo is complete
- expose small helper decisions based on runtime context

`ToolCallTask` stores tool-call task data only:

- no lifecycle behavior in this refactor
- continues to reference the persisted runner tool-call log row

### TaskManager

`TaskManager` remains responsible for:

- loading the task tree from the database
- saving the task tree to the database
- owning active user task and active todo references
- allocating task IDs
- mutating the task tree through operations such as create todo, finish todo, error todo, finish user task, and record tool call
- creating agent tools that mutate the task tree
- routing lifecycle queries to the active task

`TaskManager` should no longer contain the detailed instruction rules itself. It should delegate those rules to `UserTask` or `TodoTask`.

### SessionRunner

`SessionRunner` remains responsible for:

- constructing runtime context from session state
- passing that runtime context to `TaskManager`
- choosing runner actions such as `normal_run`, `compact`, `handle_error`, and `wait_user_input`
- syncing messages, tool-call records, metadata, and task data

The runner should not reach into task internals for lifecycle decisions. It should ask the manager for task instructions and use manager operations for task mutations.

## Runtime Context

Add a small runtime context object for task lifecycle methods.

```python
class TaskRuntimeContext(BaseModel):
    session_id: str
    context_tokens: int
    total_tool_calls: int
    active_task_tool_calls: int
    current_assistant_message_id: int | None = None
    run_done: bool = False
```

The context contains transient runner state that affects lifecycle behavior. It is not persisted inside task metadata.

## Instruction Routing

`TaskManager.user_instruction_text(context)` should become routing logic:

```python
if active_todo is not None:
    return active_todo.instruction_text(context)
if active_user_task is not None:
    return active_user_task.instruction_text(context)
return wait_for_user_instruction
```

The existing instruction policy should be preserved:

- if no active todo, the agent should determine whether the user task is complex and create a small atomic todo before complex work
- if more than 5 tool calls have happened after the previous todo, the agent should define a todo before continuing
- if an active todo exists, the agent should focus on completing it
- if more than 10 tool calls have happened for the active todo, the agent should determine whether it is done and finish it if so

## Tool Creation

Keep `AgentTool` construction in `TaskManager` for now.

The tools mutate the task tree and need access to manager-owned state such as task ID allocation, active task references, and current assistant message ID. Moving the tool construction into task classes would make task data responsible for manager mutation details.

Task classes provide lifecycle text and simple decisions only. The actual tool objects stay in the manager in this refactor.

## Compaction

Compaction remains in the existing compaction classes and manager methods for now.

This refactor is focused on normal task lifecycle behavior by task kind. Compaction is a runner phase over existing tasks rather than a task kind, so it should not be moved into `UserTask` or `TodoTask` in this pass.

## Testing

Focused tests should cover:

- `UserTask.instruction_text` with no active todo context
- `UserTask.instruction_text` when more than 5 tool calls happened after the previous todo
- `TodoTask.instruction_text` for normal active todo focus
- `TodoTask.instruction_text` when more than 10 tool calls happened for the active todo
- `TaskManager.user_instruction_text(context)` routes to active todo before user task
- existing task manager tool tests still pass
- existing session runner tests still pass

## Non-Goals

- Do not add a separate lifecycle class hierarchy.
- Do not give `ToolCallTask` lifecycle behavior.
- Do not change task database schema.
- Do not move compaction logic in this pass.
- Do not change session runner action names or persistence behavior.
