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
    AgentEvent,
    AgentStartEvent,
    AgentToolResult,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
    TurnStartEvent,
    TurnEndEvent,
)
from pi.ai.types import (
    AssistantMessageEvent,
    DoneEvent,
    ErrorEvent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ThinkingStartEvent,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
)

from simple_agent.session.session_manager import SessionBusyError, SessionManager


class RunRequest(BaseModel):
    input: str


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
    print(body.input)
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


# ── pi-agent → Vercel AI SDK v1 data frame conversion ────────────────────

_REASON_MAP = {"stop": "stop", "length": "length", "tool_use": "tool-calls"}


def convert_to_v6(event: AgentEvent | dict) -> list[dict]:
    """Convert a pi-agent event to Vercel AI SDK data stream frame dicts.

    Returns 0-N frame dicts. Each event maps independently — no mutable state.
    """
    # ── raw error dicts ───────────────────────────────────────────────
    if isinstance(event, dict):
        msg = event.get("errorText", event.get("message", ""))
        return [{"type": "error", "errorText": str(msg)}]

    # ── agent lifecycle ───────────────────────────────────────────────
    if isinstance(event, AgentStartEvent):
        return [{"type": "start", "messageId": f"msg_{id(event)}"}]

    # ── message lifecycle ─────────────────────────────────────────────
    if isinstance(event, MessageUpdateEvent):
        return _convert_assistant_event(event.assistant_message_event)

    # ── tool execution ────────────────────────────────────────────────
    if isinstance(event, ToolExecutionEndEvent):
        frame: dict = {
            "type": "tool-output-available",
            "toolCallId": event.tool_call_id,
            "output": _extract_tool_output(event.result),
        }
        if event.is_error:
            frame["errorText"] = "Tool execution failed"
        return [frame]

    # ── agent end (fallback finish) ───────────────────────────────────
    if isinstance(event, AgentEndEvent):
        return [{"type": "finish", "finishReason": "stop"}]

    # ── passthrough (no frames emitted) ───────────────────────────────
    if isinstance(event, (
        MessageStartEvent,
        MessageEndEvent,
        ToolExecutionStartEvent,
        ToolExecutionUpdateEvent,
        TurnStartEvent,
        TurnEndEvent,
    )):
        return []

    return []


def _convert_assistant_event(ae: AssistantMessageEvent) -> list[dict]:
    """Map an AssistantMessageEvent subtype to Vercel AI SDK frames."""
    tp = ae.type

    if tp == "text_start":
        return [{"type": "text-start", "id": f"txt-{ae.content_index}"}]

    if tp == "text_delta":
        return [{"type": "text-delta", "id": f"txt-{ae.content_index}", "delta": ae.delta}]

    if tp == "text_end":
        return [{"type": "text-end", "id": f"txt-{ae.content_index}"}]

    if tp == "thinking_start":
        return [{"type": "reasoning-start", "id": f"rsn-{ae.content_index}"}]

    if tp == "thinking_delta":
        return [{"type": "reasoning-delta", "id": f"rsn-{ae.content_index}", "delta": ae.delta}]

    if tp == "thinking_end":
        return [{"type": "reasoning-end", "id": f"rsn-{ae.content_index}"}]

    if tp == "toolcall_start":
        return [{"type": "tool-input-start", "toolCallId": f"call_{ae.content_index}", "toolName": ""}]

    if tp == "toolcall_delta":
        return [{"type": "tool-input-delta", "toolCallId": f"call_{ae.content_index}", "delta": ae.delta}]

    if tp == "toolcall_end":
        tc = ae.tool_call
        return [{
            "type": "tool-input-available",
            "toolCallId": f"call_{ae.content_index}",
            "toolName": tc.name,
            "input": tc.arguments,
        }]

    if tp == "done":
        return [{"type": "finish", "finishReason": _REASON_MAP.get(ae.reason, "stop")}]

    if tp == "error":
        msg = getattr(ae.error, "error_message", None) or str(ae.error)
        return [{"type": "error", "errorText": msg}]

    # "start" event inside MessageUpdate — redundant with MessageStartEvent
    return []


def _extract_tool_output(result) -> str:
    """Extract text from an AgentToolResult's content items."""
    if result is None:
        return ""
    if isinstance(result, (str, int, float, bool)):
        return str(result)
    if isinstance(result, (dict, list)):
        return result
    if isinstance(result, AgentToolResult):
        parts = [c.text for c in result.content]
        return "\n".join(parts) if parts else ""
    return str(result)


# ── stream ──────────────────────────────────────────────────────────────


async def _stream_session_events(queue: asyncio.Queue) -> AsyncGenerator[str, None]:
    """Read pi-agent events from *queue*, yield Vercel AI SDK SSE lines."""
    import logging
    logger = logging.getLogger(__name__)

    while True:
        event = await queue.get()
        if event is None:
            logger.debug("stream: received sentinel, closing")
            break

        event_type = type(event).__name__ if not isinstance(event, dict) else "dict"
        logger.debug("stream: event=%s", event_type)

        for frame in convert_to_v6(event):
            logger.debug("stream: frame type=%s", frame.get("type"))
            yield f"data: {json.dumps(frame)}\n"

    logger.debug("stream: sending [DONE]")
    yield "data: [DONE]\n"
