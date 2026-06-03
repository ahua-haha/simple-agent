"""Session REST API — CRUD + run/pause with SSE streaming."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from typing import AsyncGenerator

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
from pi.agent import AgentEvent
from pydantic import BaseModel

from simple_agent.session.session_manager import SessionBusyError, SessionManager


class RunRequest(BaseModel):
    input: str | None = None


def _get_session_manager(request: Request) -> SessionManager:
    return request.app.state.session_manager


# ── router ─────────────────────────────────────────────────────────────

router = APIRouter()


# ── endpoint functions ────────────────────────────────────────────────


@router.post("/sessions", status_code=201)
async def create_session(request: Request):
    sm = _get_session_manager(request)
    session = sm.create()
    return {
        "id": session.id,
    }


@router.get("/sessions")
async def list_sessions(request: Request):
    sm = _get_session_manager(request)
    return sm.list()


@router.get("/sessions/{session_id}")
async def get_session(session_id: str, request: Request):
    sm = _get_session_manager(request)
    session = sm.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "id": session.id,
    }


@router.delete("/sessions/{session_id}", status_code=204)
async def delete_session(session_id: str, request: Request):
    sm = _get_session_manager(request)
    if sm.get(session_id) is None:
        raise HTTPException(status_code=404, detail="Session not found")
    sm.remove(session_id)


@router.post("/sessions/{session_id}/run")
async def run_session(session_id: str, body: RunRequest, request: Request):
    sm = _get_session_manager(request)
    print(body.input)
    try:
        queue = sm.run(session_id, body.input)
    except SessionBusyError:
        raise HTTPException(status_code=409, detail="Session is already running")
    except LookupError:
        raise HTTPException(status_code=404, detail="Session not found")

    response = StreamingResponse(
        _stream_session_events(queue),
        media_type="text/event-stream",
    )
    return response


@router.post("/sessions/{session_id}/pause")
async def pause_session(session_id: str, request: Request):
    sm = _get_session_manager(request)
    if sm.get(session_id) is None:
        raise HTTPException(status_code=404, detail="Session not found")
    sm.pause(session_id)
    return {"status": "paused"}


def create_session_router() -> APIRouter:
    return router


# ── event serialization ──────────────────────────────────────────────────

def _serialize(event: AgentEvent | dict) -> str:
    """Serialize a pi-agent event to a single-line JSON string."""
    if isinstance(event, dict):
        return json.dumps(event)
    return json.dumps(asdict(event), default=lambda o: o.model_dump())


# ── stream ──────────────────────────────────────────────────────────────


async def _stream_session_events(queue: asyncio.Queue) -> AsyncGenerator[str, None]:
    """Read pi-agent events from *queue*, yield SSE frames.

    Each event is serialized as:
        event: <AgentEvent.type>
        data: <JSON AgentEvent>
        <blank line>
    """
    import logging
    logger = logging.getLogger(__name__)

    while True:
        event = await queue.get()
        if event is None:
            logger.debug("stream: received sentinel, closing")
            break

        if isinstance(event, dict):
            event_type = event.get("type", "error")
        else:
            event_type = getattr(event, "type", type(event).__name__)

        payload = _serialize(event)
        logger.debug("stream: event=%s", event_type)
        yield f"event: {event_type}\ndata: {payload}\n\n"

    logger.debug("stream: sending [DONE]")
    yield "data: [DONE]\n\n"
