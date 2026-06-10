import json

import pytest

from pi.agent import AgentTool, AgentToolResult
from pi.ai.types import AssistantMessage, TextContent, ToolCall, ToolResultMessage, UserMessage

from simple_agent.db.db import Database
from simple_agent.message_store import MessageEntry
from simple_agent.run_log import runtime_logger
from simple_agent.index.indexer import AgentIndex
from simple_agent.task_manager.lifecycle import (
    USER_TASK_COMPACT_SYSTEM_PROMPT,
    USER_TASK_SYSTEM_PROMPT,
    SessionState,
    TaskLifecycleError,
    CommonTaskLifecycle,
)
from simple_agent.task_manager.repo_memory_lifecycle import RepoMemoryLifecycle
from simple_agent.task_manager.models import RepoMemoryTask, ToolCallTask, CommonTask, task_from_metadata


def _make_db(tmp_path):
    return Database(str(tmp_path / "session.db"))


@pytest.fixture(autouse=True)
def _runtime_log_dir(tmp_path):
    runtime_logger.set_log_dir(tmp_path / "logs")


def _read_log_records(tmp_path, session_id="session_a"):
    log_file = tmp_path / "logs" / f"{session_id}.jsonl"
    return [json.loads(line) for line in log_file.read_text(encoding="utf-8").splitlines()]


def _user_lifecycle(
    task: CommonTask,
    *,
    allocate_task_id=None,
    session_state: SessionState | None = None,
) -> CommonTaskLifecycle:
    session_state = session_state or SessionState(messages=[])
    session_state.next_task = task
    session_state.next_task_id_to_run = task.id
    if allocate_task_id is not None and session_state.next_task_id_to_allocate is None:
        session_state.next_task_id_to_allocate = allocate_task_id()
    lifecycle = CommonTaskLifecycle()
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
    task = CommonTask(title="Build feature")
    lifecycle = _user_lifecycle(task)

    instruction = lifecycle.instruction_text()

    assert instruction.startswith("<system-instruction>")
    assert instruction.endswith("</system-instruction>")
    assert "## Current task process information" not in instruction
    assert "## What can be done next" in instruction
    assert "## Reminder instruction" in instruction
    assert "## Common Task" in instruction
    assert "## Repo Memory Task" in instruction
    assert "Early in the task, create a sub task" in instruction
    assert "Always finish the current task as soon as the requested work is complete" in instruction


def test_user_task_instruction_recommends_sub_task_mid_run():
    task = CommonTask(title="Build feature")
    task.children = [ToolCallTask(tool_call_log_id=index) for index in range(2)]
    lifecycle = _user_lifecycle(task)

    instruction = lifecycle.instruction_text()

    assert "Mid run, prefer creating a sub task" in instruction
    assert "Always finish the current task as soon as the requested work is complete" in instruction


def test_user_task_instruction_requires_next_task_after_six_tool_calls():
    task = CommonTask(title="Build feature")
    task.children = [ToolCallTask(tool_call_log_id=index) for index in range(6)]
    lifecycle = _user_lifecycle(task)

    instruction = lifecycle.instruction_text()

    assert "<system-instruction>" in instruction
    assert "## Current task process information" not in instruction
    assert "## What can be done next" in instruction
    assert "## Reminder instruction" in instruction
    assert "## Common Task" in instruction
    assert "Create the next sub task now before continuing more work" in instruction
    assert "Always finish the current task as soon as the requested work is complete" in instruction


def test_user_task_instruction_renders_task_tree_after_ten_tool_calls():
    task = CommonTask(title="Build feature")
    task.children = [
        CommonTask(title="Inspect files"),
        *[
            ToolCallTask(tool_call_log_id=index, tool_call_name="read")
            for index in range(11)
        ],
    ]
    task.children[0].children = [ToolCallTask(tool_call_log_id=99, tool_call_name="nested")]
    lifecycle = _user_lifecycle(task)

    instruction = lifecycle.instruction_text()

    assert "## Current task process information" in instruction
    assert "Task tree:" in instruction
    assert "- user_task [active] Build feature" in instruction
    assert "  - user_task [active] Inspect files" in instruction
    assert "  - tool_call 1. read" in instruction
    assert "nested" not in instruction


def test_tool_call_task_remains_data_only():
    task = ToolCallTask(tool_call_log_id=1)

    assert not hasattr(task, "instruction_text")


def test_task_data_objects_do_not_expose_lifecycle_methods():
    user_task = CommonTask(title="Build feature")
    repo_memory = RepoMemoryTask(
        title="Write repo memory",
        index_db_path="./index.db",
    )

    for task in [user_task, repo_memory]:
        assert not hasattr(task, "create_tools")
        assert not hasattr(task, "sync")
        assert not hasattr(task, "append_tool_call_task")


def test_repo_memory_task_roundtrips_metadata():
    task = RepoMemoryTask(
        id=3,
        title="Write repo memory",
        repo_path="/repo",
        index_db_path="/repo/.agent-index.db",
        result="Updated index",
    )

    restored = task_from_metadata(
        id=task.id,
        parent_id=task.parent_id,
        kind=task.kind,
        status=task.status,
        metadata=task.metadata_json(),
    )

    assert restored == task


def test_repo_memory_task_does_not_own_runtime_agent_index(tmp_path):
    task = RepoMemoryTask(
        id=3,
        title="Write repo memory",
        repo_path=str(tmp_path),
        index_db_path=str(tmp_path / "index.db"),
    )
    restored = task_from_metadata(
        id=task.id,
        parent_id=task.parent_id,
        kind=task.kind,
        status=task.status,
        metadata=task.metadata_json(),
    )

    assert not hasattr(task, "agent_index")
    assert not hasattr(restored, "agent_index")
    metadata = json.loads(task.metadata_json())
    assert "_agent_index" not in metadata
    assert "_current_assistant_message_id" not in metadata


def test_repo_memory_lifecycle_instruction_and_tools(tmp_path):
    task = RepoMemoryTask(
        id=3,
        title="Write repo memory",
        repo_path=str(tmp_path),
        index_db_path=str(tmp_path / "index.db"),
    )
    session_state = SessionState(messages=[])
    session_state.next_task = task
    session_state.next_task_id_to_run = task.id
    lifecycle = RepoMemoryLifecycle()
    lifecycle.set_data(session_state)

    instruction = lifecycle.instruction_text()
    tools = lifecycle.create_tools()

    assert "Write durable repo memory" in instruction
    assert "short and concise description" in instruction
    assert "what each entry does" in instruction
    assert task.repo_path in instruction
    assert task.index_db_path in instruction
    assert "index_tree" in [tool.name for tool in tools]
    assert "index_upsert" in [tool.name for tool in tools]


def test_repo_memory_lifecycle_owns_runtime_agent_index(tmp_path):
    task = RepoMemoryTask(
        id=3,
        title="Write repo memory",
        repo_path=str(tmp_path),
        index_db_path=str(tmp_path / "index.db"),
    )
    session_state = SessionState(messages=[])
    session_state.next_task = task
    session_state.next_task_id_to_run = task.id
    lifecycle = RepoMemoryLifecycle()
    lifecycle.set_data(session_state)

    first_index = lifecycle._agent_index
    tools = lifecycle.create_tools()

    assert isinstance(first_index, AgentIndex)
    assert "index_tree" in [tool.name for tool in tools]
    assert "index_upsert" in [tool.name for tool in tools]


def test_user_task_persists_compacted_tool_call_log_ids():
    task = CommonTask(id=1, title="Build feature", compacted_tool_call_log_ids=[2, 5])

    restored = task_from_metadata(
        id=task.id,
        parent_id=task.parent_id,
        kind=task.kind,
        status=task.status,
        metadata=task.metadata_json(),
    )

    assert restored.compacted_tool_call_log_ids == [2, 5]


def test_base_lifecycle_provides_next_task_instruction_and_tool():
    task = CommonTask(id=1, title="Build feature")
    session_state = SessionState(
        messages=[],
        next_task=task,
        next_task_id_to_run=task.id,
        next_task_id_to_allocate=2,
    )
    lifecycle = _user_lifecycle(task, session_state=session_state)

    next_task_tools = lifecycle.create_next_task_tools(enabled_task_kinds=["common"])

    assert [tool.name for tool in next_task_tools] == ["create_next_task"]
    assert next_task_tools[0].parameters["properties"]["kind"]["enum"] == ["common"]


def test_base_lifecycle_create_next_task_supports_common_task():
    parent = CommonTask(id=1, title="Build feature")
    session_state = SessionState(
        messages=[],
        next_task=parent,
        next_task_id_to_run=parent.id,
        next_task_id_to_allocate=2,
    )
    lifecycle = _user_lifecycle(parent, session_state=session_state)

    task = lifecycle.create_next_task(
        kind="common",
        title="Inspect lifecycle flow",
        enabled_task_kinds=["common"],
    )

    assert isinstance(task, CommonTask)
    assert task.id is None
    assert task.parent_id == parent.id
    assert task.start_message_id is None
    assert lifecycle.created_task is task


@pytest.mark.asyncio
async def test_base_lifecycle_create_next_task_tool_mutates_session_state():
    task = CommonTask(id=1, title="Build feature")
    session_state = SessionState(
        messages=[],
        next_task=task,
        next_task_id_to_run=task.id,
        next_task_id_to_allocate=2,
    )
    lifecycle = _user_lifecycle(task, session_state=session_state)
    tool = lifecycle.create_next_task_tools(enabled_task_kinds=["common"])[0]

    result = await tool.execute("call_1", {"kind": "common", "title": "Inspect files"})

    next_task = lifecycle.created_task
    assert isinstance(next_task, CommonTask)
    assert next_task.id is None
    assert next_task.parent_id == task.id
    assert next_task.start_message_id is None
    assert task.children == []
    assert session_state.next_task is task
    assert session_state.next_task_id_to_run == task.id
    assert result.content[0].text == "Created next task: user_task Inspect files"


def test_task_models_do_not_own_runtime_message_id():
    user_task = CommonTask(id=1, title="Build feature")

    restored_user = task_from_metadata(
        id=user_task.id,
        parent_id=user_task.parent_id,
        kind=user_task.kind,
        status=user_task.status,
        metadata=user_task.metadata_json(),
    )

    assert "_current_assistant_message_id" not in json.loads(user_task.metadata_json())
    assert not hasattr(user_task, "current_assistant_message_id")
    assert not hasattr(restored_user, "current_assistant_message_id")


def test_todo_task_kind_is_not_supported():
    with pytest.raises(ValueError, match="Unknown task kind: todo"):
        task_from_metadata(
            id=2,
            parent_id=1,
            kind="todo",
            status="active",
            metadata='{"title":"Inspect files"}',
        )


def test_user_task_lifecycle_uses_owned_allocator():
    next_id = 10

    def allocate_task_id():
        nonlocal next_id
        task_id = next_id
        next_id += 1
        return task_id

    user_task = CommonTask(id=1, title="Build feature")
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

    next_task = lifecycle.create_next_task(kind="common", title="Inspect files", enabled_task_kinds=["common"])
    _tool_call_records, tool_call_tasks = lifecycle._session_state.create_tool_call_record_task_entries(
        assistant_message=assistant_message,
        tool_result_messages=[tool_result],
        parent_task=user_task,
    )

    assert next_task.id is None
    assert next_task.start_message_id is None
    assert tool_call_tasks[0].id == 10
    assert tool_call_tasks[0].parent_id == user_task.id


def test_user_task_lifecycle_creates_tool_call_record_task_entries_without_appending():
    next_task_id = 20

    def allocate_task_id():
        nonlocal next_task_id
        task_id = next_task_id
        next_task_id += 1
        return task_id

    user_task = CommonTask(id=1, title="Build feature")
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


def test_lifecycle_tracks_next_task_transition():
    user_task = CommonTask(id=1, title="Build feature")
    user_lifecycle = _user_lifecycle(user_task, allocate_task_id=lambda: 2)

    next_task = user_lifecycle.create_next_task(kind="common", title="Inspect files", enabled_task_kinds=["common"])

    assert user_lifecycle._session_state.next_task_id_to_run == user_task.id
    assert user_lifecycle._session_state.next_task is user_task
    assert next_task.id is None
    assert next_task.parent_id == user_task.id


def test_lifecycle_allocates_task_id_from_session_state_context():
    session_state = SessionState(messages=[], next_task_id_to_allocate=7)
    user_task = CommonTask(id=1, title="Build feature")
    lifecycle = _user_lifecycle(user_task, session_state=session_state)

    next_task = lifecycle.create_next_task(kind="common", title="Inspect files", enabled_task_kinds=["common"])

    assert next_task.id is None
    assert session_state.next_task_id_to_allocate == 7


def test_session_state_creates_tool_call_records_and_tasks():
    session_state = SessionState(
        messages=[],
        next_task_id_to_allocate=10,
        next_tool_call_log_id=7,
    )
    parent_task = CommonTask(id=1, title="Build feature")
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
    assert tasks[0].status == "done"
    assert tasks[0].parent_id == 1
    assert tasks[0].tool_call_log_id == 7
    assert tasks[0].tool_call_name == "ls"
    assert tasks[0].tool_call_args == {"path": "."}
    assert session_state.next_tool_call_log_id == 8
    assert session_state.next_task_id_to_allocate == 11


def test_session_state_records_tool_call_args_for_rendering():
    session_state = SessionState(
        messages=[],
        next_tool_call_log_id=7,
        next_task_id_to_allocate=10,
    )
    parent_task = CommonTask(id=1, title="Build feature")
    long_value = "x" * 160
    tool_call = ToolCall(id="call_1", name="bash", arguments={"command": long_value})
    assistant_message = AssistantMessage(role="assistant", content=[tool_call])
    tool_result = ToolResultMessage(
        toolCallId="call_1",
        toolName="bash",
        content=[TextContent(text="done")],
    )

    _records, tasks = session_state.create_tool_call_record_task_entries(
        assistant_message=assistant_message,
        tool_result_messages=[tool_result],
        parent_task=parent_task,
    )

    assert tasks[0].tool_call_name == "bash"
    assert tasks[0].tool_call_args == {"command": long_value}
    rendered = tasks[0].format_for_render(sequence=1)
    assert rendered.startswith('tool_call 1. bash args: {"command":"')
    assert rendered.endswith("...")
    assert len(rendered) <= len("tool_call 1. bash args: ") + 120


def test_session_state_loads_compacted_tool_calls_by_log_id(tmp_path):
    db = _make_db(tmp_path)
    session_state = SessionState(messages=[], database=db, session_id="session_a")
    tool_call = ToolCall(id="call_7", name="ls", arguments={"path": "."})
    tool_result = ToolResultMessage(
        toolCallId="call_7",
        toolName="ls",
        content=[TextContent(text="files")],
    )
    with db.create_session() as session:
        db.insert_runner_tool_call(
            id=7,
            session_id="session_a",
            tool_call_id="call_7",
            tool_name="ls",
            tool_call_json=tool_call.model_dump_json(),
            tool_result_json=tool_result.model_dump_json(),
            session=session,
        )
        session.commit()

    assert session_state.compacted_tool_calls([7]) == [(tool_call, tool_result)]


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
    user_task = CommonTask(id=1, title="Build feature")
    child_task = CommonTask(id=2, parent_id=1, title="Inspect files")

    with session_state.create_database_session() as session:
        session_state.append_tasks_to_database(
            tasks=[user_task, child_task],
            session=session,
        )
        session.commit()

    assert db.get_managed_task(1) == user_task
    assert db.get_managed_task(2) == child_task


def test_lifecycle_appends_messages_in_memory_until_explicit_sync(tmp_path):
    db = _make_db(tmp_path)
    session_state = SessionState(messages=[], database=db, session_id="session_a")
    _user_lifecycle(CommonTask(id=1, title="Build feature"), session_state=session_state)
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
    _user_lifecycle(CommonTask(id=1, title="Build feature"), session_state=session_state)
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


def test_session_state_replaces_message_range_in_database(tmp_path):
    db = _make_db(tmp_path)
    messages = [
        MessageEntry(id=10, message=UserMessage(content=[TextContent(text="one")], timestamp=1)),
        MessageEntry(id=40, message=AssistantMessage(role="assistant", content=[TextContent(text="two")])),
        MessageEntry(id=20, message=AssistantMessage(role="assistant", content=[TextContent(text="three")])),
        MessageEntry(id=30, message=AssistantMessage(role="assistant", content=[TextContent(text="four")])),
    ]
    with db.create_session() as session:
        for entry in messages:
            db.insert_runner_message("session_a", entry.message, id=entry.id, session=session)
        session.commit()
    session_state = SessionState(messages=list(messages), database=db, session_id="session_a")
    replacement = [
        MessageEntry(id=50, message=AssistantMessage(role="assistant", content=[TextContent(text="compact")]))
    ]

    with session_state.create_database_session() as session:
        session_state.replace_message_range_in_database(
            start_message_id=40,
            end_message_id=20,
            replacement_messages=replacement,
            session=session,
        )
        session.commit()

    entries = db.list_runner_message_entries("session_a")
    assert [entry_id for entry_id, _message in entries] == [10, 30, 50]
    assert [message.content[0].text for _entry_id, message in entries] == ["one", "four", "compact"]


def test_lifecycle_syncs_explicit_tool_call_records_without_buffer(tmp_path):
    db = _make_db(tmp_path)
    session_state = SessionState(messages=[], database=db, session_id="session_a")
    _user_lifecycle(CommonTask(id=1, title="Build feature"), session_state=session_state)
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
    user_task = CommonTask(id=1, title="Build feature")
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
    assert tool_names[:2] == ["create_next_task", "finish_user_task"]
    assert "read" in tool_names
    assert agent_process.llm_calls[0]["messages"][:-1] == [seed.message]
    assert "<system-instruction>" in agent_process.llm_calls[0]["messages"][-1].content[0].text
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
    user_task = CommonTask(id=1, title="Build feature")
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
async def test_user_task_lifecycle_run_compacts_already_done_task(tmp_path):
    db = _make_db(tmp_path)
    user_task = CommonTask(id=1, parent_id=99, title="Build feature", status="done", start_message_id=1)
    user_message = UserMessage(content=[TextContent(text="Build feature")], timestamp=1)
    session_state = SessionState(
        messages=[MessageEntry(id=1, message=user_message)],
        database=db,
        session_id="session_a",
        next_message_id=2,
        next_task_id_to_allocate=10,
    )
    with db.create_session() as session:
        db.upsert_managed_task(user_task, session=session)
        db.insert_runner_message("session_a", user_message, id=1, session=session)
        session.commit()
    lifecycle = _user_lifecycle(user_task, session_state=session_state)
    agent_process = FakeAgentProcess(AssistantMessage(role="assistant", content=[TextContent(text="Done")]))

    result = await lifecycle.run(agent_process=agent_process)

    assert result is lifecycle._session_state
    assert lifecycle._session_state.next_task is None
    assert lifecycle._session_state.next_task_id_to_run == 99
    assert agent_process.llm_calls[0]["system_prompt"] == USER_TASK_COMPACT_SYSTEM_PROMPT
    persisted_messages = db.list_runner_messages("session_a")
    assert [message.role for message in persisted_messages] == ["user", "assistant"]
    assert persisted_messages[0].content[0].text == "Build feature"
    assert persisted_messages[1].content[0].text == (
        "Finished task: Build feature\n"
        "Result: Build feature\n"
        "Following tool calls preserve useful context: []"
    )


@pytest.mark.asyncio
async def test_user_task_lifecycle_run_raises_on_assistant_error(tmp_path):
    _db = _make_db(tmp_path)
    user_task = CommonTask(id=1, title="Build feature")
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
    runtime_logger.set_log_dir(tmp_path / "logs")
    user_task = CommonTask(id=1, title="Build feature")
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
    assert tool_names[:2] == ["create_next_task", "finish_user_task"]
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
    assert lifecycle._session_state.next_task is user_task
    assert [type(message).__name__ for message in db.list_runner_messages("session_a")] == [
        "AssistantMessage",
        "ToolResultMessage",
    ]
    assert [record.id for record in db.list_runner_tool_calls("session_a")] == [7]
    assert [child.tool_call_log_id for child in db.list_managed_task_children(user_task.id)] == [7]
    records = _read_log_records(tmp_path)
    assert len(records) == 1
    assert records[0]["event"] == "handle_running"
    assert records[0]["session_id"] == "session_a"
    assert records[0]["assistant_message_id"] == 1
    assert records[0]["assistant_message"]["content"][0]["name"] == "ls"
    assert records[0]["tool_results"] == [
        {"tool_call_id": "call_1", "tool_name": "ls", "message_id": 2}
    ]


@pytest.mark.asyncio
async def test_user_task_lifecycle_run_syncs_created_common_task(tmp_path):
    db = _make_db(tmp_path)
    user_task = CommonTask(id=1, title="Build feature")
    session_state = SessionState(
        messages=[],
        database=db,
        session_id="session_a",
        next_task_id_to_allocate=2,
    )
    lifecycle = _user_lifecycle(user_task, session_state=session_state)
    assistant_message = AssistantMessage(
        role="assistant",
        content=[
            ToolCall(
                id="call_1",
                name="create_next_task",
                arguments={"kind": "common", "title": "Inspect lifecycle flow"},
            )
        ],
    )
    agent_process = ExecutingFakeAgentProcess(assistant_message)

    await lifecycle.run(agent_process=agent_process)

    children = db.list_managed_task_children(user_task.id)
    common_tasks = [child for child in children if isinstance(child, CommonTask)]
    tool_calls = [child for child in children if child.kind == "tool_call"]
    assert len(common_tasks) == 1
    assert common_tasks[0].title == "Inspect lifecycle flow"
    assert common_tasks[0].start_message_id == 1
    assert len(tool_calls) == 1
    assert [child.kind for child in user_task.children] == ["tool_call", "user_task"]
    assert lifecycle._session_state.next_task_id_to_run == common_tasks[0].id
    assert lifecycle._session_state.next_task is user_task.children[1]


def test_user_task_lifecycle_records_compacted_tool_call_log_id():
    user_task = CommonTask(
        id=1,
        title="Build feature",
        status="done",
        children=[ToolCallTask(id=2, parent_id=1, status="done", tool_call_log_id=7)],
    )
    lifecycle = _user_lifecycle(user_task)

    lifecycle.record_compacted_tool_call_log(tool_call_log_id=7)
    lifecycle.record_compacted_tool_call_log(tool_call_log_id=7)

    assert user_task.compacted_tool_call_log_ids == [7]


@pytest.mark.asyncio
async def test_user_task_lifecycle_run_compact_one_turn_records_tool_call_log_id(tmp_path):
    class OneTurnCompactAgentProcess:
        def __init__(self):
            self.llm_calls = []
            self.tool_calls = []

        async def call_llm_step(self, system_prompt, messages, tools, cancel_event=None):
            self.llm_calls.append(
                {
                    "system_prompt": system_prompt,
                    "messages": list(messages),
                    "tools": [tool.name for tool in tools],
                    "cancel_event": cancel_event,
                }
            )
            return AssistantMessage(
                role="assistant",
                content=[
                    ToolCall(
                        id="compact_record",
                        name="record_compacted_tool_call_log",
                        arguments={"tool_call_log_id": 7},
                    )
                ],
            )

        async def run_tool_calls_step(self, tools, assistant_message, cancel_event=None):
            self.tool_calls.append(
                {
                    "tools": [tool.name for tool in tools],
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

    user_task = CommonTask(
        id=1,
        title="Build feature",
        status="done",
        children=[ToolCallTask(id=2, parent_id=1, status="done", tool_call_log_id=7)],
    )
    db = _make_db(tmp_path)
    session_state = SessionState(
        messages=[MessageEntry(id=1, message=UserMessage(content=[TextContent(text="go")], timestamp=1))],
        database=db,
        session_id="session_a",
        next_message_id=2,
        next_task_id_to_allocate=10,
    )
    lifecycle = _user_lifecycle(user_task, session_state=session_state)
    agent_process = OneTurnCompactAgentProcess()

    result = await lifecycle.run_compact_one_turn(agent_process=agent_process)

    assert result is lifecycle._session_state
    assert agent_process.llm_calls[0]["system_prompt"] == USER_TASK_COMPACT_SYSTEM_PROMPT
    assert agent_process.llm_calls[0]["tools"] == ["record_compacted_tool_call_log"]
    assert agent_process.llm_calls[0]["messages"][0] == session_state.messages[0].message
    assert "Runtime instruction for compacting phase" in agent_process.llm_calls[0]["messages"][-1].content[0].text
    assert agent_process.tool_calls[0]["tools"] == ["record_compacted_tool_call_log"]
    assert user_task.compacted_tool_call_log_ids == [7]
    assert [entry.id for entry in lifecycle._session_state.messages] == [1, 2, 3]
    assert [record.tool_name for record in db.list_runner_tool_calls("session_a")] == [
        "record_compacted_tool_call_log"
    ]
    persisted_user_task = db.get_managed_task(user_task.id)
    assert persisted_user_task.compacted_tool_call_log_ids == [7]
    persisted_children = db.list_managed_task_children(user_task.id)
    assert [child.tool_call_name for child in persisted_children] == [
        None,
        "record_compacted_tool_call_log",
    ]


@pytest.mark.asyncio
async def test_user_task_lifecycle_run_compact_one_turn_finishes_and_replaces_messages(tmp_path):
    db = _make_db(tmp_path)
    user_task = CommonTask(
        id=1,
        parent_id=99,
        title="Build feature",
        status="done",
        start_message_id=1,
        compacted_tool_call_log_ids=[7],
        children=[
            ToolCallTask(id=2, parent_id=1, status="done", tool_call_log_id=7),
            CommonTask(id=3, parent_id=1, title="Old subtask", status="done"),
        ],
    )
    original_messages = [
        UserMessage(content=[TextContent(text="Build feature")], timestamp=1),
        AssistantMessage(role="assistant", content=[TextContent(text="work")]),
    ]
    session_state = SessionState(
        messages=[
            MessageEntry(id=index + 1, message=message)
            for index, message in enumerate(original_messages)
        ],
        database=db,
        session_id="session_a",
        next_message_id=3,
        next_task_id_to_allocate=10,
    )
    with db.create_session() as session:
        db.upsert_managed_task(user_task, session=session)
        for child in user_task.children:
            db.upsert_managed_task(child, session=session)
        db.replace_runner_messages("session_a", original_messages, ids=[1, 2], session=session)
        db.insert_runner_tool_call(
            id=7,
            session_id="session_a",
            tool_call_id="call_7",
            tool_name="ls",
            tool_call_json='{"id":"call_7","name":"ls","arguments":{"path":"."}}',
            tool_result_json=ToolResultMessage(
                toolCallId="call_7",
                toolName="ls",
                content=[TextContent(text="files")],
            ).model_dump_json(),
            session=session,
        )
        session.commit()
    lifecycle = _user_lifecycle(user_task, session_state=session_state)
    agent_process = FakeAgentProcess(AssistantMessage(role="assistant", content=[TextContent(text="compact done")]))

    result = await lifecycle.run_compact_one_turn(agent_process=agent_process)

    assert result is lifecycle._session_state
    assert lifecycle._session_state.next_task is None
    assert lifecycle._session_state.next_task_id_to_run == 99
    assert [entry.id for entry in lifecycle._session_state.messages] == [4, 5, 6]
    assert [entry.message.role for entry in lifecycle._session_state.messages] == ["user", "assistant", "tool_result"]
    assert lifecycle._session_state.messages[0].message.content[0].text == "Build feature"
    assert lifecycle._session_state.messages[1].message.content[0].text == (
        "Finished task: Build feature\n"
        "Result: Build feature\n"
        "Following tool calls preserve useful context: ['call_7']"
    )
    compacted_tool_call = lifecycle._session_state.messages[1].message.content[1]
    assert compacted_tool_call == ToolCall(id="call_7", name="ls", arguments={"path": "."})
    assert lifecycle._session_state.messages[2].message.tool_call_id == "call_7"
    persisted_messages = db.list_runner_messages("session_a")
    assert [message.role for message in persisted_messages] == ["user", "assistant", "tool_result"]
    assert persisted_messages[1].content[1] == ToolCall(id="call_7", name="ls", arguments={"path": "."})
    assert persisted_messages[2].content[0].text == "files"
    persisted_children = db.list_managed_task_children(user_task.id)
    assert [child.id for child in persisted_children] == [2, 3]


def test_user_task_lifecycle_should_compact_after_more_than_ten_tool_calls():
    user_task = CommonTask(
        id=1,
        title="Build feature",
        children=[ToolCallTask(id=index + 2, parent_id=1, tool_call_log_id=index) for index in range(11)],
    )
    lifecycle = _user_lifecycle(user_task)

    assert lifecycle.should_compact_after_turn() is True


def test_user_task_lifecycle_should_compact_after_nested_tool_calls():
    subtask = CommonTask(
        id=2,
        parent_id=1,
        title="Inspect files",
        children=[
            ToolCallTask(id=index + 3, parent_id=2, tool_call_log_id=index)
            for index in range(11)
        ],
    )
    user_task = CommonTask(id=1, title="Build feature", children=[subtask])
    lifecycle = _user_lifecycle(user_task)

    assert lifecycle.should_compact_after_turn() is True
