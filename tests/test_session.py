"""Tests for Session."""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from simple_agent.session import Session
from simple_agent.state.state import CommitData, RunRecord, SessionData, SingleRunTask


class TestSessionCommit:
    """Tests for Session.commit()."""

    @pytest.mark.asyncio
    async def test_commit_creates_json_file(self, tmpdir):
        """commit() should create a JSON file with correct content."""
        session = Session("test")

        await session.run("summarize what this project do, what the core module do")

        filepath = await session.commit()
        assert os.path.exists(filepath)
        assert filepath.endswith(".json")