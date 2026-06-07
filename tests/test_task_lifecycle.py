import pytest

from pi.ai.types import AssistantMessage, TextContent, ToolCall, ToolResultMessage, UserMessage

from simple_agent.db.db import Database
from simple_agent.message_store import MessageEntry
from simple_agent.task_manager.lifecycle import (
    USER_TASK_COMPACT_SYSTEM_PROMPT,
    USER_TASK_SYSTEM_PROMPT,
    TodoTaskLifecycle,
    UserTaskLifecycle,
)
from simple_agent.task_manager.models import TodoTask, ToolCallTask, UserTask


def _make_db(tmp_path):
    return Database(str(tmp_path / "session.db"))


class FakeAgentProcess:
    def __init__(self, assistant_message: AssistantMessage, tool_results: list[ToolResultMessage] | None = None):
        self.assistant_message = assistant_message
        self.tool_results = tool_results or []
        self.llm_calls = []
        self.tool_calls = []

    async def call_llm_step(self, system_prompt, messages, tools, cancel_event=None):
        self.llm_calls.append(
            {
                "system_prompt": system_prompt,
                "messages": list(messages),
                "tools": list(tools),
                "cancel_event": cancel_event,
            }
        )
        return self.assistant_message

    async def run_tool_calls_step(self, tools, assistant_message, cancel_event=None):
        self.tool_calls.append(
            {
                "tools": list(tools),
                "assistant_message": assistant_message,
                "cancel_event": cancel_event,
            }
        )
        return self.tool_results


class FakeCompactAgentProcess:
    def __init__(self):
        self.calls = []

    async def call_llm_step(self, system_prompt, messages, tools, cancel_event=None):
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "messages": list(messages),
                "tools": [tool.name for tool in tools],
                "cancel_event": cancel_event,
            }
        )
        if len(self.calls) == 1:
            return AssistantMessage(
                role="assistant",
                content=[
                    ToolCall(
                        id="compact_create",
                        name="create_compacted_user_task",
                        arguments={"description": "Summarized work"},
                    ),
                    ToolCall(
                        id="compact_record",
                        name="record_compacted_tool_call",
                        arguments={"tool_call_log_id": 7},
                    ),
                    ToolCall(id="compact_finish", name="finish_compacted_user_task", arguments={}),
                ],
            )
        return AssistantMessage(role="assistant", content=[TextContent(text="compact done")])

    async def run_tool_calls_step(self, tools, assistant_message, cancel_event=None):
        tools_by_name = {tool.name: tool for tool in tools}
        results = []
        for content in assistant_message.content:
            if not isinstance(content, ToolCall):
                continue
            result = await tools_by_name[content.name].execute(content.id, content.arguments)
            results.append(
                ToolResultMessage(
                    toolCallId=content.id,
                    toolName=content.name,
                    content=result.content,
                )
            )
        return results


def test_user_task_instruction_asks_for_complexity_check_when_tool_count_is_small():
    task = UserTask(title="Build feature")
    lifecycle = UserTaskLifecycle(task)

    instruction = lifecycle.instruction_text()

    assert "Runtime instruction for this turn" in instruction
    assert "Determine whether the user task is complex" in instruction
    assert "create the next small atomic todo first" in instruction


def test_user_task_instruction_requires_todo_after_many_tool_calls():
    task = UserTask(title="Build feature")
    task.children = [
        ToolCallTask(title=f"Tool call {index}", tool_call_log_id=index)
        for index in range(6)
    ]
    lifecycle = UserTaskLifecycle(task)

    instruction = lifecycle.instruction_text()

    assert "More than 5 tool calls have run since the previous todo" in instruction
    assert "create a small atomic todo before doing more work" in instruction


def test_todo_task_instruction_focuses_active_todo_when_tool_count_is_small():
    task = TodoTask(title="Inspect files")
    lifecycle = TodoTaskLifecycle(task)

    instruction = lifecycle.instruction_text()

    assert "Focus on the active todo: Inspect files" in instruction
    assert "Call finish_todo immediately when it is complete" in instruction


def test_todo_task_instruction_prompts_finish_check_after_many_tool_calls():
    task = TodoTask(title="Inspect files")
    task.children = [
        ToolCallTask(title=f"Tool call {index}", tool_call_log_id=index)
        for index in range(11)
    ]
    lifecycle = TodoTaskLifecycle(task)

    instruction = lifecycle.instruction_text()

    assert "More than 10 tool calls have run for the active todo" in instruction
    assert "call finish_todo now with a concise result" in instruction


def test_tool_call_task_remains_data_only():
    task = ToolCallTask(title="Tool call 1", tool_call_log_id=1)

    assert not hasattr(task, "instruction_text")


def test_task_data_objects_do_not_expose_lifecycle_methods():
    user_task = UserTask(title="Build feature")
    todo = TodoTask(title="Inspect files")

    for task in [user_task, todo]:
        assert not hasattr(task, "create_tools")
        assert not hasattr(task, "sync")
        assert not hasattr(task, "append_tool_call_task")


def test_user_task_lifecycle_uses_owned_allocator():
    next_id = 10

    def allocate_task_id():
        nonlocal next_id
        task_id = next_id
        next_id += 1
        return task_id

    user_task = UserTask(id=1, title="Build feature")
    lifecycle = UserTaskLifecycle(user_task, allocate_task_id=allocate_task_id)
    lifecycle.current_assistant_message_id = 22
    lifecycle.load_tool_call_log_id(7)
    assistant_message = AssistantMessage(
        role="assistant",
        content=[ToolCall(id="call_1", name="ls", arguments={"path": "."})],
    )
    tool_result = ToolResultMessage(
        toolCallId="call_1",
        toolName="ls",
        content=[TextContent(text="files")],
    )

    todo = lifecycle.create_todo_task(title="Inspect files")
    _tool_call_records, tool_call_tasks = lifecycle.create_tool_call_record_task_entries(
        assistant_message=assistant_message,
        tool_result_messages=[tool_result],
    )

    assert todo.id == 10
    assert todo.start_message_id == 22
    assert tool_call_tasks[0].id == 11
    assert tool_call_tasks[0].parent_id == user_task.id


def test_user_task_lifecycle_creates_tool_call_record_task_entries_without_appending():
    next_task_id = 20

    def allocate_task_id():
        nonlocal next_task_id
        task_id = next_task_id
        next_task_id += 1
        return task_id

    user_task = UserTask(id=1, title="Build feature")
    lifecycle = UserTaskLifecycle(user_task, allocate_task_id=allocate_task_id)
    lifecycle.load_tool_call_log_id(7)
    assistant_message = AssistantMessage(
        role="assistant",
        content=[ToolCall(id="call_1", name="ls", arguments={"path": "."})],
    )
    tool_result = ToolResultMessage(
        toolCallId="call_1",
        toolName="ls",
        content=[TextContent(text="files")],
    )

    tool_call_records, tool_call_tasks = lifecycle.create_tool_call_record_task_entries(
        assistant_message=assistant_message,
        tool_result_messages=[tool_result],
    )

    assert tool_call_records == [(7, assistant_message.content[0], tool_result)]
    assert len(tool_call_tasks) == 1
    tool_call_task = tool_call_tasks[0]
    assert tool_call_task.id == 20
    assert tool_call_task.parent_id == user_task.id
    assert tool_call_task.tool_call_log_id == 7
    assert user_task.children == []
    assert lifecycle.next_tool_call_log_id == 8


def test_todo_task_lifecycle_uses_owned_message_id_for_finish():
    todo = TodoTask(id=2, parent_id=1, title="Inspect files")
    lifecycle = TodoTaskLifecycle(todo, allocate_task_id=lambda: 3)
    lifecycle.current_assistant_message_id = 44

    lifecycle.finish_task(result="Inspected files")

    assert todo.status == "done"
    assert todo.end_message_id == 44


def test_lifecycle_tracks_next_task_transition():
    user_task = UserTask(id=1, title="Build feature")
    user_lifecycle = UserTaskLifecycle(user_task, allocate_task_id=lambda: 2)

    todo = user_lifecycle.create_todo_task(title="Inspect files")

    assert user_lifecycle.consume_next_task() is todo
    assert user_lifecycle.consume_next_task() is None

    todo_lifecycle = TodoTaskLifecycle(todo, user_task=user_task)
    todo_lifecycle.finish_task(result="Done")

    assert todo_lifecycle.consume_next_task() is user_task


def test_lifecycle_appends_messages_in_memory_until_explicit_sync(tmp_path):
    db = _make_db(tmp_path)
    lifecycle = UserTaskLifecycle(UserTask(id=1, title="Build feature"))
    seed = MessageEntry(id=1, message=UserMessage(content=[TextContent(text="hello")], timestamp=1))
    lifecycle.load_messages([seed], next_message_id=2)

    entry = lifecycle.append_message(AssistantMessage(role="assistant", content=[TextContent(text="hi")]))

    assert entry.id == 2
    assert lifecycle.messages == [seed, entry]
    assert lifecycle.next_message_id == 3
    assert db.list_runner_messages("session_a") == []

    with db.create_session() as session:
        lifecycle.sync_messages("session_a", db, [entry], session=session)
        session.commit()

    persisted = db.list_runner_messages("session_a")
    assert len(persisted) == 1
    assert persisted[0].content[0].text == "hi"


def test_lifecycle_replaces_message_range_and_syncs_explicit_message_list(tmp_path):
    db = _make_db(tmp_path)
    lifecycle = UserTaskLifecycle(UserTask(id=1, title="Build feature"))
    first = MessageEntry(id=1, message=UserMessage(content=[TextContent(text="one")], timestamp=1))
    second = MessageEntry(id=2, message=AssistantMessage(role="assistant", content=[TextContent(text="two")]))
    third = MessageEntry(id=3, message=AssistantMessage(role="assistant", content=[TextContent(text="three")]))
    lifecycle.load_messages([first, second, third], next_message_id=4)
    with db.create_session() as session:
        lifecycle.sync_replaced_messages("session_a", db, lifecycle.messages, session=session)
        session.commit()

    replacement = lifecycle.replace_message_range(
        start_message_id=2,
        end_message_id=3,
        replacement_messages=[
            AssistantMessage(role="assistant", content=[TextContent(text="compact")]),
        ],
    )

    assert [entry.id for entry in replacement] == [4]
    assert [entry.message.content[0].text for entry in lifecycle.messages] == ["one", "compact"]

    with db.create_session() as session:
        lifecycle.sync_replaced_messages("session_a", db, lifecycle.messages, session=session)
        session.commit()

    assert [message.content[0].text for message in db.list_runner_messages("session_a")] == ["one", "compact"]


def test_lifecycle_syncs_explicit_tool_call_records_without_buffer(tmp_path):
    db = _make_db(tmp_path)
    lifecycle = UserTaskLifecycle(UserTask(id=1, title="Build feature"))
    tool_call = ToolCall(id="call_1", name="ls", arguments={"path": "."})
    tool_result = ToolResultMessage(
        toolCallId="call_1",
        toolName="ls",
        content=[TextContent(text="files")],
    )

    with db.create_session() as session:
        lifecycle.sync_tool_calls("session_a", db, [(3, tool_call, tool_result)], session=session)
        session.commit()

    records = db.list_runner_tool_calls("session_a")
    assert len(records) == 1
    assert records[0].id == 3
    assert records[0].tool_call_id == "call_1"
    assert records[0].tool_name == "ls"
    assert '"path":"."' in records[0].tool_call_json
    assert "files" in records[0].tool_result_json


@pytest.mark.asyncio
async def test_user_task_lifecycle_run_calls_llm_appends_message_and_returns_next_action(tmp_path):
    db = _make_db(tmp_path)
    user_task = UserTask(id=1, title="Build feature")
    lifecycle = UserTaskLifecycle(user_task)
    seed = MessageEntry(id=1, message=UserMessage(content=[TextContent(text="Build feature")], timestamp=1))
    lifecycle.load_messages([seed], next_message_id=2)
    assistant_message = AssistantMessage(role="assistant", content=[TextContent(text="Done")])
    agent_process = FakeAgentProcess(assistant_message)

    result = await lifecycle.run(
        agent_process=agent_process,
        context_token_threshold=1000,
        tool_call_threshold=100,
    )

    assert agent_process.llm_calls[0]["system_prompt"] == USER_TASK_SYSTEM_PROMPT
    tool_names = [tool.name for tool in agent_process.llm_calls[0]["tools"]]
    assert tool_names[:2] == ["create_todo", "finish_user_task"]
    assert "read" in tool_names
    assert agent_process.llm_calls[0]["messages"][:-1] == [seed.message]
    assert "Runtime instruction for this turn" in agent_process.llm_calls[0]["messages"][-1].content[0].text
    assert agent_process.tool_calls == []
    assert result.new_messages == [MessageEntry(id=2, message=assistant_message)]
    assert lifecycle.messages == [seed, *result.new_messages]
    assert lifecycle.next_message_id == 3
    assert result.tool_call_records == []
    assert result.next_action == "compact"
    assert result.next_task is user_task
    assert user_task.status == "done"
    assert db.list_runner_messages("session_a") == []


@pytest.mark.asyncio
async def test_user_task_lifecycle_run_executes_tools_and_returns_current_task(tmp_path):
    db = _make_db(tmp_path)
    user_task = UserTask(id=1, title="Build feature")
    next_task_id = 10

    def allocate_task_id():
        nonlocal next_task_id
        task_id = next_task_id
        next_task_id += 1
        return task_id

    lifecycle = UserTaskLifecycle(user_task, allocate_task_id=allocate_task_id)
    lifecycle.load_messages([], next_message_id=1)
    lifecycle.load_tool_call_log_id(7)
    assistant_message = AssistantMessage(
        role="assistant",
        content=[ToolCall(id="call_1", name="ls", arguments={"path": "."})],
    )
    tool_result = ToolResultMessage(
        toolCallId="call_1",
        toolName="ls",
        content=[TextContent(text="files")],
    )
    agent_process = FakeAgentProcess(assistant_message, [tool_result])
    result = await lifecycle.run(
        agent_process=agent_process,
        context_token_threshold=1000,
        tool_call_threshold=100,
    )

    assert agent_process.llm_calls[0]["system_prompt"] == USER_TASK_SYSTEM_PROMPT
    assert agent_process.tool_calls[0]["assistant_message"] is assistant_message
    tool_names = [tool.name for tool in agent_process.tool_calls[0]["tools"]]
    assert tool_names[:2] == ["create_todo", "finish_user_task"]
    assert "read" in tool_names
    assert result.new_messages == [
        MessageEntry(id=1, message=assistant_message),
        MessageEntry(id=2, message=tool_result),
    ]
    assert lifecycle.messages == result.new_messages
    assert lifecycle.next_message_id == 3
    assert lifecycle.next_tool_call_log_id == 8
    assert result.tool_call_records == [(7, assistant_message.content[0], tool_result)]
    assert [child.tool_call_log_id for child in user_task.children] == [7]
    assert user_task.children[0].parent_id == user_task.id
    assert lifecycle.current_assistant_message_id is None
    assert result.next_action == "normal_run"
    assert result.next_task is user_task
    assert db.list_runner_messages("session_a") == []
    assert db.list_runner_tool_calls("session_a") == []


def test_user_task_lifecycle_compact_tools_do_not_require_begin_step():
    user_task = UserTask(id=1, title="Build feature", status="active")
    user_task.children.append(TodoTask(id=2, parent_id=1, title="Inspect files", status="done"))
    lifecycle = UserTaskLifecycle(user_task, allocate_task_id=lambda: 3)

    result = lifecycle.create_compacted_user_task(description="Summary")

    assert result is user_task
    assert user_task.result == "Summary"


def test_user_task_lifecycle_compaction_result_uses_user_task_boundaries():
    next_id = 10

    def allocate_task_id():
        nonlocal next_id
        task_id = next_id
        next_id += 1
        return task_id

    user_task = UserTask(
        id=1,
        title="Build feature",
        status="done",
        start_message_id=4,
        end_message_id=9,
        children=[ToolCallTask(id=2, parent_id=1, title="Tool call 7", status="done", tool_call_log_id=7)],
    )
    lifecycle = UserTaskLifecycle(user_task, allocate_task_id=allocate_task_id)

    result = lifecycle.create_compacted_user_task(description="Summarized work")
    lifecycle.record_compacted_tool_call(tool_call_log_id=7)
    lifecycle.finish_compacted_user_task()

    start_message_id, end_message_id, messages = lifecycle.compaction_result()

    assert result is user_task
    assert user_task.result == "Summarized work"
    assert start_message_id == 4
    assert end_message_id == 9
    assert messages == [
        AssistantMessage(
            role="assistant",
            content=[TextContent(text="Compacted user task: Summarized work\nUseful tool calls: [7]")],
        )
    ]


def test_user_task_lifecycle_compaction_sync_replaces_user_task_children(tmp_path):
    db = _make_db(tmp_path)
    user_task = UserTask(id=1, title="Build feature", status="done", start_message_id=1, end_message_id=5)
    first_tool = ToolCallTask(id=2, parent_id=1, title="Tool call 10", status="done", tool_call_log_id=10)
    todo = TodoTask(id=3, parent_id=1, title="Inspect files", status="done", result="Done")
    user_task.children = [first_tool, todo]
    with db.create_session() as session:
        db.upsert_managed_task(user_task, session=session)
        db.upsert_managed_task(first_tool, session=session)
        db.upsert_managed_task(todo, session=session)
        session.commit()

    next_id = 20

    def allocate_task_id():
        nonlocal next_id
        task_id = next_id
        next_id += 1
        return task_id

    lifecycle = UserTaskLifecycle(user_task, allocate_task_id=allocate_task_id)
    lifecycle.create_compacted_user_task(description="Whole task summary")
    lifecycle.record_compacted_tool_call(tool_call_log_id=10)
    lifecycle.finish_compacted_user_task()

    with db.create_session() as session:
        compacted = lifecycle.sync_compaction(db, session)
        session.commit()

    assert compacted is user_task
    assert db.get_managed_task(20) is not None
    assert db.get_managed_task(todo.id) is None
    loaded_children = db.list_managed_task_children(user_task.id)
    assert [child.tool_call_log_id for child in loaded_children] == [10]
    assert db.get_managed_task(user_task.id).result == "Whole task summary"


def test_user_task_lifecycle_compaction_requires_finished_compacted_user_task():
    user_task = UserTask(
        id=1,
        title="Build feature",
        status="done",
        start_message_id=1,
        end_message_id=2,
        children=[ToolCallTask(id=2, parent_id=1, title="Tool call 1", status="done", tool_call_log_id=1)],
    )
    lifecycle = UserTaskLifecycle(user_task, allocate_task_id=lambda: 3)

    with pytest.raises(RuntimeError, match="No compacted user task result"):
        lifecycle.compaction_result()


@pytest.mark.asyncio
async def test_user_task_lifecycle_handle_compact_runs_loop_and_returns_compaction_result():
    user_task = UserTask(
        id=1,
        title="Build feature",
        status="done",
        start_message_id=1,
        end_message_id=3,
        children=[ToolCallTask(id=2, parent_id=1, title="Tool call 7", status="done", tool_call_log_id=7)],
    )
    lifecycle = UserTaskLifecycle(user_task, allocate_task_id=lambda: 10)
    agent_process = FakeCompactAgentProcess()
    original_messages = [
        UserMessage(content=[TextContent(text="Build feature")], timestamp=1),
        AssistantMessage(role="assistant", content=[TextContent(text="work")]),
        AssistantMessage(role="assistant", content=[TextContent(text="done")]),
    ]
    lifecycle.load_messages([
        MessageEntry(id=index + 1, message=message)
        for index, message in enumerate(original_messages)
    ], next_message_id=4)

    result = await lifecycle.handle_compact(
        agent_process=agent_process,
    )

    assert [call["system_prompt"] for call in agent_process.calls] == [
        USER_TASK_COMPACT_SYSTEM_PROMPT,
        USER_TASK_COMPACT_SYSTEM_PROMPT,
    ]
    assert agent_process.calls[0]["tools"] == [
        "create_compacted_user_task",
        "record_compacted_tool_call",
        "finish_compacted_user_task",
    ]
    assert agent_process.calls[0]["messages"][:-1] == original_messages
    assert "Runtime instruction for compacting phase" in agent_process.calls[0]["messages"][-1].content[0].text
    assert result.next_action == "wait_user_input"
    assert result.next_task is None
    assert [entry.id for entry in lifecycle.messages] == [4]
    assert lifecycle.messages[0].message == AssistantMessage(
        role="assistant",
        content=[TextContent(text="Compacted user task: Summarized work\nUseful tool calls: [7]")],
    )
