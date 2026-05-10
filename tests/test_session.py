"""Tests for Session."""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from simple_agent.session import Session
from simple_agent.state.state import CommitData, RunRecord, SessionData, SingleRunTask

requires_api_key = pytest.mark.skipif(
    not os.environ.get("DEEPSEEK_API_KEY"),
    reason="DEEPSEEK_API_KEY not set",
)


class TestSessionInit:
    """Tests for Session.__init__."""

    def test_new_session_has_empty_state(self):
        """New session should have empty messages and runs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session = Session("test", base_dir=tmpdir)
            assert session.messages == []
            assert session.runs == []
            assert session.commit_data is None

    def test_create_new(self, tmpdir):
        """Calling Session() creates a new one when no JSON file exists."""
        session = Session("test-session", base_dir=str(tmpdir))
        assert session.runs == []
        assert session.messages == []
        assert session.commit_data is None

    def test_load_existing_session(self, tmpdir):
        """Session loads existing JSON file."""
        data = SessionData(
            name="test-session",
            messages=[],
            runs=[],
            created_at=1000.0,
            updated_at=2000.0,
        )
        filepath = os.path.join(str(tmpdir), "test-session.json")
        with open(filepath, "w") as f:
            f.write(data.model_dump_json(indent=2))

        session = Session("test-session", base_dir=str(tmpdir))
        assert session.messages == []
        assert session.runs == []
        assert session.commit_data is None
        assert session._created_at == 1000.0


class TestSessionRun:
    """Tests for Session.run()."""

    @requires_api_key
    @pytest.mark.asyncio
    async def test_run_returns_completed_task_with_results(self, tmpdir):
        """run() should return a SingleRunTask with populated results."""
        session = Session("test", base_dir=str(tmpdir))
        task = await session.run("list the files in src/")

        assert isinstance(task, SingleRunTask)
        assert task.input == "list the files in src/"
        assert task.result is not None

    @requires_api_key
    @pytest.mark.asyncio
    async def test_run_accumulates_messages(self, tmpdir):
        """Messages should grow after each run."""
        session = Session("test", base_dir=str(tmpdir))
        assert len(session.messages) == 0

        await session.run("what files are in src/")
        msg_count_after_first = len(session.messages)
        assert msg_count_after_first > 0

        await session.run("summarize the main entry point")
        assert len(session.messages) > msg_count_after_first

    @requires_api_key
    @pytest.mark.asyncio
    async def test_run_records_run_record(self, tmpdir):
        """Each run should append a RunRecord."""
        session = Session("test", base_dir=str(tmpdir))
        await session.run("list files")
        await session.run("explore main.py")

        assert len(session.runs) == 2
        r0 = session.runs[0]
        assert r0.input == "list files"
        assert r0.status == "finished"
        assert r0.new_message_count > 0
        assert r0.started_at <= r0.finished_at


class TestSessionCommit:
    """Tests for Session.commit()."""

    @requires_api_key
    @pytest.mark.asyncio
    async def test_commit_creates_json_file(self, tmpdir):
        """commit() should create a JSON file with correct content."""
        session = Session("test", base_dir=str(tmpdir))

        filepath = await session.commit()
        assert os.path.exists(filepath)
        assert filepath.endswith(".json")

        with open(filepath) as f:
            raw = json.load(f)
        assert raw["name"] == "test"
        assert raw["messages"] == []
        assert raw["runs"] == []
        assert raw["commit_data"]["extracted_instructions"] == []
        assert raw["commit_data"]["aggregated_results"] == []

    @requires_api_key
    @pytest.mark.asyncio
    async def test_commit_after_run_persists_data(self, tmpdir):
        """Data committed after a run should be loadable."""
        session = Session("test", base_dir=str(tmpdir))
        await session.run("list files")
        await session.commit()

        session2 = Session("test", base_dir=str(tmpdir))
        assert len(session2.messages) > 0
        assert len(session2.runs) == 1
        assert session2.runs[0].input == "list files"
        assert session2.commit_data is not None

    @requires_api_key
    @pytest.mark.asyncio
    async def test_commit_writes_to_temp_first(self, tmpdir):
        """commit() should write to temp file then rename for atomicity."""
        session = Session("test", base_dir=str(tmpdir))

        await session.commit()
        tmp_files = [f for f in os.listdir(str(tmpdir)) if f.endswith(".tmp")]
        assert len(tmp_files) == 0, f"Temp files left behind: {tmp_files}"

    @requires_api_key
    @pytest.mark.asyncio
    async def test_commit_creates_directory_if_missing(self, tmpdir):
        """commit() should create the sessions directory if it doesn't exist."""
        nested = os.path.join(str(tmpdir), "nested", "dirs")
        session = Session("test", base_dir=nested)
        assert not os.path.exists(nested)
        await session.commit()
        assert os.path.exists(nested)

    @requires_api_key
    @pytest.mark.asyncio
    async def test_commit_overwrites_existing(self, tmpdir):
        """Commit should overwrite the existing JSON file."""
        session = Session("test", base_dir=str(tmpdir))
        await session.commit()
        mtime1 = os.path.getmtime(session._filepath())

        import time
        time.sleep(0.01)
        await session.commit()
        mtime2 = os.path.getmtime(session._filepath())
        assert mtime2 > mtime1

    @requires_api_key
    @pytest.mark.asyncio
    async def test_commit_populates_commit_data(self, tmpdir):
        """Commit on empty session should populate commit_data."""
        session = Session("test", base_dir=str(tmpdir))
        await session.commit()

        assert session.commit_data is not None
        assert isinstance(session.commit_data, CommitData)
        assert session.commit_data.extracted_instructions == []
        assert session.commit_data.aggregated_results == []

    @requires_api_key
    @pytest.mark.asyncio
    async def test_commit_loads_commit_data_from_file(self, tmpdir):
        """Loading a committed session should restore commit_data."""
        session = Session("test", base_dir=str(tmpdir))
        await session.commit()

        session2 = Session("test", base_dir=str(tmpdir))
        assert session2.commit_data is not None
        assert session2.commit_data.extracted_instructions == []


class TestSessionListSessions:
    """Tests for Session.list_sessions()."""

    def test_list_sessions_returns_names(self, tmpdir):
        """Should return session names from the directory."""
        sessions_dir = os.path.join(str(tmpdir), "sessions")
        os.makedirs(sessions_dir, exist_ok=True)
        for name in ["alpha", "beta", "gamma"]:
            with open(os.path.join(sessions_dir, f"{name}.json"), "w") as f:
                f.write("{}")

        names = Session.list_sessions(base_dir=sessions_dir)
        assert "alpha" in names
        assert "beta" in names
        assert "gamma" in names

    def test_list_sessions_returns_empty_when_missing(self, tmpdir):
        """Should return empty list when directory does not exist."""
        nonexistent = os.path.join(str(tmpdir), "nope")
        names = Session.list_sessions(base_dir=nonexistent)
        assert names == []
