import pytest

from pi.ai.types import AssistantMessage, TextContent, ToolCall, ToolResultMessage, UserMessage

from simple_agent.db.db import Database
from simple_agent.message_store import MessageEntry
from simple_agent.task_manager.lifecycle import (
    USER_TASK_COMPACT_SYSTEM_PROMPT,
    USER_TASK_SYSTEM_PROMPT,
    SessionState,
    TaskLifecycleError,
    TodoTaskLifecycle,
    UserTaskLifecycle,
)
from simple_agent.task_manager.models import TodoTask, ToolCallTask, UserTask


def _make_db(tmp_path):
    return Database(str(tmp_path / "session.db"))


def _user_lifecycle(
    task: UserTask,
    *,
    allocate_task_id=None,
    session_state: SessionState | None = None,
) -> UserTaskLifecycle:
    session_state = session_state or SessionState(messages=[])
    session_state.next_task = task
    session_state.next_task_id_to_run = task.id
    if allocate_task_id is not None and session_state.next_task_id_to_allocate is None:
        session_state.next_task_id_to_allocate = allocate_task_id()
    lifecycle = UserTaskLifecycle()
    lifecycle.set_data(session_state)
    return lifecycle


def _todo_lifecycle(
    task: TodoTask,
    *,
    allocate_task_id=None,
    session_state: SessionState | None = None,
) -> TodoTaskLifecycle:
    session_state = session_state or SessionState(messages=[])
    session_state.next_task = task
    session_state.next_task_id_to_run = task.id
    if allocate_task_id is not None and session_state.next_task_id_to_allocate is None:
        session_state.next_task_id_to_allocate = allocate_task_id()
    lifecycle = TodoTaskLifecycle()
    lifecycle.set_data(session_state)
    return lifecycle


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


class ExecutingFakeAgentProcess(FakeAgentProcess):
    async def run_tool_calls_step(self, tools, assistant_message, cancel_event=None):
        self.tool_calls.append(
            {
                "tools": list(tools),
                "assistant_message": assistant_message,
                "cancel_event": cancel_event,
            }
        )
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
    lifecycle = _user_lifecycle(task)

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
    lifecycle = _user_lifecycle(task)

    instruction = lifecycle.instruction_text()

    assert "More than 5 tool calls have run since the previous todo" in instruction
    assert "create a small atomic todo before doing more work" in instruction


def test_todo_task_instruction_focuses_active_todo_when_tool_count_is_small():
    task = TodoTask(title="Inspect files")
    lifecycle = _todo_lifecycle(task)

    instruction = lifecycle.instruction_text()

    assert "Focus on the active todo: Inspect files" in instruction
    assert "Call finish_todo immediately when it is complete" in instruction


def test_todo_task_instruction_prompts_finish_check_after_many_tool_calls():
    task = TodoTask(title="Inspect files")
    task.children = [
        ToolCallTask(title=f"Tool call {index}", tool_call_log_id=index)
        for index in range(11)
    ]
    lifecycle = _todo_lifecycle(task)

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
    lifecycle = _user_lifecycle(user_task, allocate_task_id=allocate_task_id)
    lifecycle.current_assistant_message_id = 22
    lifecycle._session_state.next_tool_call_log_id = 7
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
    _tool_call_records, tool_call_tasks = lifecycle._session_state.create_tool_call_record_task_entries(
        assistant_message=assistant_message,
        tool_result_messages=[tool_result],
        parent_task=user_task,
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
    lifecycle = _user_lifecycle(user_task, allocate_task_id=allocate_task_id)
    lifecycle._session_state.next_tool_call_log_id = 7
    assistant_message = AssistantMessage(
        role="assistant",
        content=[ToolCall(id="call_1", name="ls", arguments={"path": "."})],
    )
    tool_result = ToolResultMessage(
        toolCallId="call_1",
        toolName="ls",
        content=[TextContent(text="files")],
    )

    tool_call_records, tool_call_tasks = lifecycle._session_state.create_tool_call_record_task_entries(
        assistant_message=assistant_message,
        tool_result_messages=[tool_result],
        parent_task=user_task,
    )

    assert tool_call_records == [(7, assistant_message.content[0], tool_result)]
    assert len(tool_call_tasks) == 1
    tool_call_task = tool_call_tasks[0]
    assert tool_call_task.id == 20
    assert tool_call_task.parent_id == user_task.id
    assert tool_call_task.tool_call_log_id == 7
    assert user_task.children == []
    assert lifecycle._session_state.next_tool_call_log_id == 8


def test_todo_task_lifecycle_uses_owned_message_id_for_finish():
    todo = TodoTask(id=2, parent_id=1, title="Inspect files")
    lifecycle = _todo_lifecycle(todo, allocate_task_id=lambda: 3)
    lifecycle.current_assistant_message_id = 44

    lifecycle.finish_task(result="Inspected files")

    assert todo.status == "done"
    assert todo.end_message_id == 44


def test_lifecycle_tracks_next_task_transition():
    user_task = UserTask(id=1, title="Build feature")
    user_lifecycle = _user_lifecycle(user_task, allocate_task_id=lambda: 2)

    todo = user_lifecycle.create_todo_task(title="Inspect files")

    assert user_lifecycle._session_state.next_task_id_to_run == todo.id
    assert user_lifecycle._session_state.next_task is todo

    todo_lifecycle = _todo_lifecycle(todo)
    todo_lifecycle.finish_task(result="Done")

    assert todo_lifecycle._session_state.next_task_id_to_run == user_task.id
    assert todo_lifecycle._session_state.next_task is None


def test_lifecycle_allocates_task_id_from_session_state_context():
    session_state = SessionState(messages=[], next_task_id_to_allocate=7)
    user_task = UserTask(id=1, title="Build feature")
    lifecycle = _user_lifecycle(user_task, session_state=session_state)

    todo = lifecycle.create_todo_task(title="Inspect files")

    assert todo.id == 7
    assert session_state.next_task_id_to_allocate == 8


def test_session_state_creates_tool_call_records_and_tasks():
    session_state = SessionState(
        messages=[],
        next_task_id_to_allocate=10,
        next_tool_call_log_id=7,
    )
    parent_task = UserTask(id=1, title="Build feature")
    tool_call = ToolCall(id="call_1", name="ls", arguments={"path": "."})
    assistant_message = AssistantMessage(role="assistant", content=[tool_call])
    tool_result = ToolResultMessage(
        toolCallId="call_1",
        toolName="ls",
        content=[TextContent(text="files")],
    )

    records, tasks = session_state.create_tool_call_record_task_entries(
        assistant_message=assistant_message,
        tool_result_messages=[tool_result],
        parent_task=parent_task,
    )

    assert records == [(7, tool_call, tool_result)]
    assert len(tasks) == 1
    assert tasks[0].id == 10
    assert tasks[0].title == "Tool call 7"
    assert tasks[0].status == "done"
    assert tasks[0].parent_id == 1
    assert tasks[0].tool_call_log_id == 7
    assert session_state.next_tool_call_log_id == 8
    assert session_state.next_task_id_to_allocate == 11


def test_session_state_appends_messages_to_database(tmp_path):
    db = _make_db(tmp_path)
    session_state = SessionState(messages=[], database=db, session_id="session_a")
    entry = MessageEntry(
        id=5,
        message=AssistantMessage(role="assistant", content=[TextContent(text="hello")]),
    )

    with session_state.create_database_session() as session:
        session_state.append_messages_to_database(
            messages=[entry],
            session=session,
        )
        session.commit()

    messages = db.list_runner_messages("session_a")
    assert len(messages) == 1
    assert messages[0].content[0].text == "hello"
    entries = db.list_runner_message_entries("session_a")
    assert entries[0][0] == 5


def test_session_state_appends_tool_calls_to_database(tmp_path):
    db = _make_db(tmp_path)
    session_state = SessionState(messages=[], database=db, session_id="session_a")
    tool_call = ToolCall(id="call_1", name="ls", arguments={"path": "."})
    tool_result = ToolResultMessage(
        toolCallId="call_1",
        toolName="ls",
        content=[TextContent(text="files")],
    )

    with session_state.create_database_session() as session:
        session_state.append_tool_calls_to_database(
            tool_calls=[(7, tool_call, tool_result)],
            session=session,
        )
        session.commit()

    records = db.list_runner_tool_calls("session_a")
    assert len(records) == 1
    assert records[0].id == 7
    assert records[0].tool_call_id == "call_1"
    assert records[0].tool_name == "ls"
    assert '"path":"."' in records[0].tool_call_json
    assert "files" in records[0].tool_result_json


def test_session_state_appends_tasks_to_database(tmp_path):
    db = _make_db(tmp_path)
    session_state = SessionState(messages=[], database=db, session_id="session_a")
    user_task = UserTask(id=1, title="Build feature")
    todo = TodoTask(id=2, parent_id=1, title="Inspect files")

    with session_state.create_database_session() as session:
        session_state.append_tasks_to_database(
            tasks=[user_task, todo],
            session=session,
        )
        session.commit()

    assert db.get_managed_task(1) == user_task
    assert db.get_managed_task(2) == todo


def test_lifecycle_appends_messages_in_memory_until_explicit_sync(tmp_path):
    db = _make_db(tmp_path)
    session_state = SessionState(messages=[], database=db, session_id="session_a")
    _user_lifecycle(UserTask(id=1, title="Build feature"), session_state=session_state)
    seed = MessageEntry(id=1, message=UserMessage(content=[TextContent(text="hello")], timestamp=1))
    session_state.messages = [seed]
    session_state.next_message_id = 2

    entry = session_state.append_message(AssistantMessage(role="assistant", content=[TextContent(text="hi")]))

    assert entry.id == 2
    assert session_state.messages == [seed, entry]
    assert session_state.next_message_id == 3
    assert db.list_runner_messages("session_a") == []

    with session_state.create_database_session() as session:
        session_state.append_messages_to_database(messages=[entry], session=session)
        session.commit()

    persisted = db.list_runner_messages("session_a")
    assert len(persisted) == 1
    assert persisted[0].content[0].text == "hi"


def test_lifecycle_replaces_message_range_and_syncs_explicit_message_list(tmp_path):
    db = _make_db(tmp_path)
    session_state = SessionState(messages=[], database=db, session_id="session_a")
    _user_lifecycle(UserTask(id=1, title="Build feature"), session_state=session_state)
    first = MessageEntry(id=1, message=UserMessage(content=[TextContent(text="one")], timestamp=1))
    second = MessageEntry(id=2, message=AssistantMessage(role="assistant", content=[TextContent(text="two")]))
    third = MessageEntry(id=3, message=AssistantMessage(role="assistant", content=[TextContent(text="three")]))
    session_state.messages = [first, second, third]
    session_state.next_message_id = 4
    with session_state.create_database_session() as session:
        db.replace_runner_messages(
            "session_a",
            [entry.message for entry in session_state.messages],
            ids=[entry.id for entry in session_state.messages],
            session=session,
        )
        session.commit()

    replacement = session_state.replace_message_range(
        start_message_id=2,
        end_message_id=3,
        replacement_messages=[
            AssistantMessage(role="assistant", content=[TextContent(text="compact")]),
        ],
    )

    assert [entry.id for entry in replacement] == [4]
    assert [entry.message.content[0].text for entry in session_state.messages] == ["one", "compact"]

    with session_state.create_database_session() as session:
        db.replace_runner_messages(
            "session_a",
            [entry.message for entry in session_state.messages],
            ids=[entry.id for entry in session_state.messages],
            session=session,
        )
        session.commit()

    assert [message.content[0].text for message in db.list_runner_messages("session_a")] == ["one", "compact"]


def test_session_state_replaces_messages_in_database(tmp_path):
    db = _make_db(tmp_path)
    first = MessageEntry(id=1, message=AssistantMessage(role="assistant", content=[TextContent(text="one")]))
    second = MessageEntry(id=2, message=AssistantMessage(role="assistant", content=[TextContent(text="two")]))
    db.insert_runner_message("session_a", first.message, id=first.id)
    db.insert_runner_message("session_a", second.message, id=second.id)
    session_state = SessionState(
        messages=[
            MessageEntry(id=3, message=AssistantMessage(role="assistant", content=[TextContent(text="compact")])),
        ],
        database=db,
        session_id="session_a",
    )

    with session_state.create_database_session() as session:
        session_state.replace_messages_in_database(session=session)
        session.commit()

    entries = db.list_runner_message_entries("session_a")
    assert [entry_id for entry_id, _message in entries] == [3]
    assert [message.content[0].text for _entry_id, message in entries] == ["compact"]


def test_session_state_replaces_task_tree_in_database(tmp_path):
    db = _make_db(tmp_path)
    user_task = UserTask(id=1, title="Build feature")
    stale_todo = TodoTask(id=2, parent_id=1, title="Old todo")
    replacement_tool = ToolCallTask(id=3, parent_id=1, title="Tool call 7", status="done", tool_call_log_id=7)
    user_task.children = [replacement_tool]
    db.upsert_managed_task(user_task)
    db.upsert_managed_task(stale_todo)
    session_state = SessionState(messages=[], database=db, session_id="session_a")

    with session_state.create_database_session() as session:
        session_state.replace_task_tree_in_database(task=user_task, session=session)
        session.commit()

    assert db.get_managed_task(stale_todo.id) is None
    assert [child.tool_call_log_id for child in db.list_managed_task_children(user_task.id)] == [7]


def test_lifecycle_syncs_explicit_tool_call_records_without_buffer(tmp_path):
    db = _make_db(tmp_path)
    session_state = SessionState(messages=[], database=db, session_id="session_a")
    _user_lifecycle(UserTask(id=1, title="Build feature"), session_state=session_state)
    tool_call = ToolCall(id="call_1", name="ls", arguments={"path": "."})
    tool_result = ToolResultMessage(
        toolCallId="call_1",
        toolName="ls",
        content=[TextContent(text="files")],
    )

    with session_state.create_database_session() as session:
        session_state.append_tool_calls_to_database(tool_calls=[(3, tool_call, tool_result)], session=session)
        session.commit()

    records = db.list_runner_tool_calls("session_a")
    assert len(records) == 1
    assert records[0].id == 3
    assert records[0].tool_call_id == "call_1"
    assert records[0].tool_name == "ls"
    assert '"path":"."' in records[0].tool_call_json
    assert "files" in records[0].tool_result_json


@pytest.mark.asyncio
async def test_user_task_lifecycle_run_calls_llm_appends_message_and_returns_state(tmp_path):
    db = _make_db(tmp_path)
    user_task = UserTask(id=1, title="Build feature")
    session_state = SessionState(
        messages=[],
        database=db,
        session_id="session_a",
        next_task_id_to_allocate=2,
    )
    lifecycle = _user_lifecycle(user_task, session_state=session_state)
    seed = MessageEntry(id=1, message=UserMessage(content=[TextContent(text="Build feature")], timestamp=1))
    assistant_message = AssistantMessage(role="assistant", content=[TextContent(text="Done")])
    agent_process = FakeAgentProcess(assistant_message)
    lifecycle._session_state.messages = [seed]
    lifecycle._session_state.next_message_id = 2
    result = await lifecycle.run(agent_process=agent_process)

    assert agent_process.llm_calls[0]["system_prompt"] == USER_TASK_SYSTEM_PROMPT
    tool_names = [tool.name for tool in agent_process.llm_calls[0]["tools"]]
    assert tool_names[:2] == ["create_todo", "finish_user_task"]
    assert "read" in tool_names
    assert agent_process.llm_calls[0]["messages"][:-1] == [seed.message]
    assert "Runtime instruction for this turn" in agent_process.llm_calls[0]["messages"][-1].content[0].text
    assert agent_process.tool_calls == []
    assert result is lifecycle._session_state
    assert lifecycle._session_state.messages == [seed, MessageEntry(id=2, message=assistant_message)]
    assert lifecycle._session_state.next_message_id == 3
    assert lifecycle._session_state.next_task is None
    assert lifecycle._session_state.next_task_id_to_run is None
    assert user_task.status == "done"
    assert [message.content[0].text for message in db.list_runner_messages("session_a")] == ["Done"]
    assert db.get_managed_task(user_task.id).status == "done"


@pytest.mark.asyncio
async def test_user_task_lifecycle_run_keeps_done_task_for_compaction_when_needed(tmp_path):
    db = _make_db(tmp_path)
    user_task = UserTask(id=1, title="Build feature")
    session_state = SessionState(
        messages=[],
        database=db,
        session_id="session_a",
        next_task_id_to_allocate=2,
    )
    lifecycle = _user_lifecycle(user_task, session_state=session_state)
    lifecycle.should_compact_after_turn = lambda: True
    assistant_message = AssistantMessage(role="assistant", content=[TextContent(text="Done")])
    agent_process = FakeAgentProcess(assistant_message)

    result = await lifecycle.run(agent_process=agent_process)

    assert result is lifecycle._session_state
    assert user_task.status == "done"
    assert lifecycle._session_state.next_task is user_task
    assert lifecycle._session_state.next_task_id_to_run == user_task.id
    assert db.get_managed_task(user_task.id).status == "done"


@pytest.mark.asyncio
async def test_user_task_lifecycle_run_raises_on_assistant_error(tmp_path):
    _db = _make_db(tmp_path)
    user_task = UserTask(id=1, title="Build feature")
    lifecycle = _user_lifecycle(user_task)
    assistant_message = AssistantMessage(
        role="assistant",
        content=[TextContent(text="failed")],
        stopReason="error",
        errorMessage="model failed",
    )
    agent_process = FakeAgentProcess(assistant_message)

    with pytest.raises(TaskLifecycleError, match="model failed"):
        await lifecycle.run(agent_process=agent_process)


@pytest.mark.asyncio
async def test_user_task_lifecycle_run_executes_tools_and_returns_current_task(tmp_path):
    db = _make_db(tmp_path)
    user_task = UserTask(id=1, title="Build feature")
    session_state = SessionState(
        messages=[],
        database=db,
        session_id="session_a",
        next_task_id_to_allocate=10,
    )
    lifecycle = _user_lifecycle(user_task, session_state=session_state)
    lifecycle._session_state.messages = []
    lifecycle._session_state.next_message_id = 1
    lifecycle._session_state.next_tool_call_log_id = 7
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
    )

    assert agent_process.llm_calls[0]["system_prompt"] == USER_TASK_SYSTEM_PROMPT
    assert agent_process.tool_calls[0]["assistant_message"] is assistant_message
    tool_names = [tool.name for tool in agent_process.tool_calls[0]["tools"]]
    assert tool_names[:2] == ["create_todo", "finish_user_task"]
    assert "read" in tool_names
    assert result is lifecycle._session_state
    assert lifecycle._session_state.messages == [
        MessageEntry(id=1, message=assistant_message),
        MessageEntry(id=2, message=tool_result),
    ]
    assert lifecycle._session_state.next_message_id == 3
    assert lifecycle._session_state.next_tool_call_log_id == 8
    assert [child.tool_call_log_id for child in user_task.children] == [7]
    assert user_task.children[0].parent_id == user_task.id
    assert lifecycle.current_assistant_message_id is None
    assert lifecycle._session_state.next_task is user_task
    assert [type(message).__name__ for message in db.list_runner_messages("session_a")] == [
        "AssistantMessage",
        "ToolResultMessage",
    ]
    assert [record.id for record in db.list_runner_tool_calls("session_a")] == [7]
    assert [child.tool_call_log_id for child in db.list_managed_task_children(user_task.id)] == [7]


@pytest.mark.asyncio
async def test_user_task_lifecycle_run_syncs_created_todo_task(tmp_path):
    db = _make_db(tmp_path)
    user_task = UserTask(id=1, title="Build feature")
    session_state = SessionState(
        messages=[],
        database=db,
        session_id="session_a",
        next_task_id_to_allocate=2,
    )
    lifecycle = _user_lifecycle(user_task, session_state=session_state)
    assistant_message = AssistantMessage(
        role="assistant",
        content=[ToolCall(id="call_1", name="create_todo", arguments={"title": "Inspect files"})],
    )
    agent_process = ExecutingFakeAgentProcess(assistant_message)

    await lifecycle.run(agent_process=agent_process)

    children = db.list_managed_task_children(user_task.id)
    todos = [child for child in children if child.kind == "todo"]
    tool_calls = [child for child in children if child.kind == "tool_call"]
    assert len(todos) == 1
    assert todos[0].title == "Inspect files"
    assert todos[0].start_message_id == 1
    assert len(tool_calls) == 1
    assert lifecycle._session_state.next_task_id_to_run == todos[0].id
    assert lifecycle._session_state.next_task is user_task.children[0]


def test_user_task_lifecycle_compact_tools_do_not_require_begin_step():
    user_task = UserTask(id=1, title="Build feature", status="active")
    user_task.children.append(TodoTask(id=2, parent_id=1, title="Inspect files", status="done"))
    lifecycle = _user_lifecycle(user_task, allocate_task_id=lambda: 3)

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
    lifecycle = _user_lifecycle(user_task, allocate_task_id=allocate_task_id)

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


def test_user_task_lifecycle_compaction_requires_finished_compacted_user_task():
    user_task = UserTask(
        id=1,
        title="Build feature",
        status="done",
        start_message_id=1,
        end_message_id=2,
        children=[ToolCallTask(id=2, parent_id=1, title="Tool call 1", status="done", tool_call_log_id=1)],
    )
    lifecycle = _user_lifecycle(user_task, allocate_task_id=lambda: 3)

    with pytest.raises(RuntimeError, match="No compacted user task result"):
        lifecycle.compaction_result()


@pytest.mark.asyncio
async def test_user_task_lifecycle_handle_compact_runs_loop_and_returns_state(tmp_path):
    db = _make_db(tmp_path)
    user_task = UserTask(
        id=1,
        parent_id=99,
        title="Build feature",
        status="done",
        start_message_id=1,
        end_message_id=3,
        children=[
            ToolCallTask(id=2, parent_id=1, title="Tool call 7", status="done", tool_call_log_id=7),
            TodoTask(id=3, parent_id=1, title="Old todo", status="done"),
        ],
    )
    session_state = SessionState(
        messages=[],
        database=db,
        session_id="session_a",
        next_task_id_to_allocate=10,
    )
    lifecycle = _user_lifecycle(user_task, session_state=session_state)
    agent_process = FakeCompactAgentProcess()
    original_messages = [
        UserMessage(content=[TextContent(text="Build feature")], timestamp=1),
        AssistantMessage(role="assistant", content=[TextContent(text="work")]),
        AssistantMessage(role="assistant", content=[TextContent(text="done")]),
    ]
    with db.create_session() as session:
        db.upsert_managed_task(user_task, session=session)
        for child in user_task.children:
            db.upsert_managed_task(child, session=session)
        db.replace_runner_messages("session_a", original_messages, ids=[1, 2, 3], session=session)
        session.commit()
    lifecycle._session_state.messages = [
        MessageEntry(id=index + 1, message=message)
        for index, message in enumerate(original_messages)
    ]
    lifecycle._session_state.next_message_id = 4

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
    assert result is lifecycle._session_state
    assert lifecycle._session_state.next_task is None
    assert lifecycle._session_state.next_task_id_to_run == 99
    assert [entry.id for entry in lifecycle._session_state.messages] == [4]
    assert lifecycle._session_state.messages[0].message == AssistantMessage(
        role="assistant",
        content=[TextContent(text="Compacted user task: Summarized work\nUseful tool calls: [7]")],
    )
    persisted_messages = db.list_runner_messages("session_a")
    assert len(persisted_messages) == 1
    assert persisted_messages[0] == lifecycle._session_state.messages[0].message
    persisted_user_task = db.get_managed_task(user_task.id)
    assert persisted_user_task.result == "Summarized work"
    persisted_children = db.list_managed_task_children(user_task.id)
    assert [child.tool_call_log_id for child in persisted_children] == [7]
    assert db.get_managed_task(3) is None


@pytest.mark.asyncio
async def test_user_task_lifecycle_handle_compact_without_children_routes_to_parent():
    user_task = UserTask(
        id=1,
        parent_id=99,
        title="Build feature",
        status="done",
        start_message_id=1,
        end_message_id=2,
    )
    lifecycle = _user_lifecycle(user_task)
    agent_process = FakeCompactAgentProcess()

    result = await lifecycle.handle_compact(agent_process=agent_process)

    assert result is lifecycle._session_state
    assert lifecycle._session_state.next_task is None
    assert lifecycle._session_state.next_task_id_to_run == 99
    assert agent_process.calls == []
