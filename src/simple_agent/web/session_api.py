"""Session REST API — CRUD + run/pause with SSE streaming."""

from __future__ import annotations

import asyncio
import json
from typing import AsyncGenerator

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from pi.agent import (
    AgentEndEvent,
    MessageUpdateEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
)

from simple_agent.session.session_manager import SessionBusyError


class RunRequest(BaseModel):
    input: str


def _get_session_manager(request: Request):
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
        "created_at": session._created_at,
        "updated_at": session._updated_at,
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
        "created_at": session._created_at,
        "updated_at": session._updated_at,
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
    try:
        queue = sm.run(session_id, body.input)
    except SessionBusyError:
        raise HTTPException(status_code=409, detail="Session is already running")
    except LookupError:
        raise HTTPException(status_code=404, detail="Session not found")

    response = StreamingResponse(
        _stream_session_events(queue),
        media_type="text/plain; charset=utf-8",
    )
    response.headers["x-vercel-ai-data-stream"] = "v1"
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


# ── stream helper ──────────────────────────────────────────────────────


async def _stream_session_events(queue: asyncio.Queue) -> AsyncGenerator[str, None]:
    """Convert agent events from *queue* to v6 data stream protocol frames."""
    message_id = None
    in_text_block = False
    text_block_id = None
    text_block_counter = 0

    def _emit(frame: dict) -> str:
        return f"data: {json.dumps(frame)}\n"

    while True:
        event = await queue.get()
        if event is None:
            break

        if isinstance(event, AgentEndEvent):
            if in_text_block:
                yield _emit({"type": "text-end", "id": text_block_id})
                in_text_block = False
            finish_reason = "stop"
            yield _emit({"type": "finish", "finishReason": finish_reason})

        elif isinstance(event, MessageUpdateEvent):
            ae = event.assistant_message_event
            if ae.type == "text_delta":
                if not in_text_block:
                    text_block_counter += 1
                    text_block_id = f"txt_{text_block_counter}"
                    yield _emit({"type": "text-start", "id": text_block_id})
                    if message_id is None:
                        message_id = f"msg_{id(event)}"
                        yield _emit({"type": "start", "messageId": message_id})
                    in_text_block = True
                yield _emit({"type": "text-delta", "id": text_block_id, "delta": ae.delta})

        elif isinstance(event, ToolExecutionStartEvent):
            if in_text_block:
                yield _emit({"type": "text-end", "id": text_block_id})
                in_text_block = False
            yield _emit({
                "type": "tool-input-start",
                "toolCallId": event.tool_call_id,
                "toolName": event.tool_name,
            })
            yield _emit({
                "type": "tool-input-available",
                "toolCallId": event.tool_call_id,
                "toolName": event.tool_name,
                "input": event.args,
            })

        elif isinstance(event, ToolExecutionEndEvent):
            result = event.result
            if not isinstance(result, dict):
                result = {"value": str(result)} if not hasattr(result, "__dict__") else result.__dict__
            yield _emit({
                "type": "tool-output-available",
                "toolCallId": event.tool_call_id,
                "output": result,
            })

    yield "data: [DONE]\n"
