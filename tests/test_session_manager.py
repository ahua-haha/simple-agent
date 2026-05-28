"""Tests for SessionManager."""

from __future__ import annotations

import asyncio
import os

import pytest

from simple_agent.session.session_manager import SessionBusyError, SessionManager
from simple_agent.session.session import Session


class TestSessionManager:
    """SessionManager create/get/list/remove."""

    def test_create_adds_to_registry(self, tmp_path):
        sm = SessionManager(sessions_dir=str(tmp_path))
        s = sm.create()
        assert s.id in sm._sessions
        assert sm.get(s.id) is s

    def test_get_missing_returns_none(self, tmp_path):
        sm = SessionManager(sessions_dir=str(tmp_path))
        assert sm.get("nonexistent") is None

    def test_list_returns_all(self, tmp_path):
        sm = SessionManager(sessions_dir=str(tmp_path))
        s1 = sm.create()
        s2 = sm.create()
        sessions = sm.list()
        assert len(sessions) == 2
        ids = [s["id"] for s in sessions]
        assert s1.id in ids
        assert s2.id in ids

    def test_remove_parks_and_removes(self, tmp_path):
        sm = SessionManager(sessions_dir=str(tmp_path))
        s = sm.create()
        sid = s.id
        sm.remove(sid)
        assert sm.get(sid) is None

    def test_persistence_round_trip(self, tmp_path):
        sm = SessionManager(sessions_dir=str(tmp_path))
        s = sm.create()
        s._cursor_id = 42
        s._checkpoint()

        # Reload
        sm2 = SessionManager(sessions_dir=str(tmp_path))
        restored = sm2.get(s.id)
        assert restored is not None
        assert restored._cursor_id == 42

    def test_reload_skips_bad_dir_entries(self, tmp_path):
        # Create a non-db file in the sessions dir — shouldn't crash reload
        sm = SessionManager(sessions_dir=str(tmp_path))
        s = sm.create()
        s._checkpoint()

        sm2 = SessionManager(sessions_dir=str(tmp_path))
        # Session is parked (not auto-loaded), get() reloads from disk
        restored = sm2.get(s.id)
        assert restored is not None
        assert restored._cursor_id == s._cursor_id


class TestSessionManagerRunPause:
    """SessionManager run, pause, and cooldown."""

    @pytest.mark.asyncio
    async def test_run_returns_event_queue(self, tmp_path, monkeypatch):
        sm = SessionManager(sessions_dir=str(tmp_path))
        s = sm.create()

        # Mock Session.run to avoid actual LLM calls
        async def mock_run(self, user_input):
            pass

        monkeypatch.setattr(
            "simple_agent.session.session.Session.run", mock_run
        )

        queue = sm.run(s.id, "test input")
        assert queue is not None
        assert isinstance(queue, asyncio.Queue)

    @pytest.mark.asyncio
    async def test_run_on_running_raises_busy_error(self, tmp_path):
        sm = SessionManager(sessions_dir=str(tmp_path))
        s = sm.create()
        sm._run_tasks[s.id] = asyncio.create_task(asyncio.sleep(1))

        with pytest.raises(SessionBusyError):
            sm.run(s.id, "another input")
        # Clean up
        sm._run_tasks[s.id].cancel()

    @pytest.mark.asyncio
    async def test_pause_signals_session(self, tmp_path):
        sm = SessionManager(sessions_dir=str(tmp_path))
        s = sm.create()
        sm.pause(s.id)
        assert s._cancel_event.is_set()

    @pytest.mark.asyncio
    async def test_pause_idle_session_is_noop(self, tmp_path):
        sm = SessionManager(sessions_dir=str(tmp_path))
        s = sm.create()
        # Should not raise
        sm.pause(s.id)

    @pytest.mark.asyncio
    async def test_cooldown_parks_session(self, tmp_path):
        sm = SessionManager(sessions_dir=str(tmp_path), cooldown_seconds=0)
        s = sm.create()
        s._running = False
        sm._start_cooldown(s.id, s)
        # With cooldown_seconds=0, the timer fires immediately
        await asyncio.sleep(0.05)
        assert s.id not in sm._sessions

    @pytest.mark.asyncio
    async def test_run_cancels_cooldown(self, tmp_path, monkeypatch):
        sm = SessionManager(sessions_dir=str(tmp_path), cooldown_seconds=10)
        s = sm.create()
        sm._start_cooldown(s.id, s)
        assert s.id in sm._cooldown_timers

        # Mock Session.run to avoid actual LLM calls
        async def mock_run(self, user_input):
            pass

        monkeypatch.setattr(
            "simple_agent.session.session.Session.run", mock_run
        )

        sm.run(s.id, "test")
        assert s.id not in sm._cooldown_timers


class TestAPIEndpoints:
    """Session API endpoints (via TestClient)."""

    @pytest.fixture
    def client(self, tmp_path):
        from fastapi.testclient import TestClient

        from simple_agent.web.app import create_app

        app = create_app(
            db_path=":memory:",
            sessions_dir=str(tmp_path),
            cooldown_seconds=60,
        )
        return TestClient(app)

    def test_create_session(self, client):
        resp = client.post("/api/sessions", json={})
        assert resp.status_code == 201
        data = resp.json()
        assert "id" in data
        assert "created_at" in data

    def test_list_sessions(self, client):
        client.post("/api/sessions", json={})
        client.post("/api/sessions", json={})
        resp = client.get("/api/sessions")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2

    def test_get_session(self, client):
        created = client.post("/api/sessions", json={})
        sid = created.json()["id"]
        resp = client.get(f"/api/sessions/{sid}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == sid

    def test_get_missing_session(self, client):
        resp = client.get("/api/sessions/nonexistent")
        assert resp.status_code == 404

    def test_delete_session(self, client):
        created = client.post("/api/sessions", json={})
        sid = created.json()["id"]
        resp = client.delete(f"/api/sessions/{sid}")
        assert resp.status_code == 204

        listing = client.get("/api/sessions").json()
        assert len(listing) == 0

    def test_delete_missing_session(self, client):
        resp = client.delete("/api/sessions/nonexistent")
        assert resp.status_code == 404

    def test_pause_session(self, client):
        created = client.post("/api/sessions", json={})
        sid = created.json()["id"]
        resp = client.post(f"/api/sessions/{sid}/pause")
        assert resp.status_code == 200
        assert resp.json()["status"] == "paused"

    def test_pause_missing_session(self, client):
        resp = client.post("/api/sessions/nonexistent/pause")
        assert resp.status_code == 404

    def test_run_session(self, client, monkeypatch):
        import asyncio as _asyncio

        created = client.post("/api/sessions", json={})
        sid = created.json()["id"]

        # Mock Session.run to avoid actual LLM calls
        async def mock_run(self, user_input):
            if self.event_queue is not None:
                self.event_queue.put_nowait(None)

        monkeypatch.setattr(
            "simple_agent.session.session.Session.run", mock_run
        )

        resp = client.post(f"/api/sessions/{sid}/run", json={"input": "hello"})
        assert resp.status_code == 200
        assert "text/plain" in resp.headers["content-type"]
        assert resp.headers["x-vercel-ai-data-stream"] == "v1"

    def test_run_missing_session(self, client):
        resp = client.post("/api/sessions/nonexistent/run", json={"input": "hello"})
        assert resp.status_code == 404
