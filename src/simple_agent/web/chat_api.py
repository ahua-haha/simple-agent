"""Chat API endpoint following the Vercel AI SDK data stream protocol (v6).

Wire format: SSE-like — each frame is a ``data: {json}\\n`` line,
terminated by ``data: [DONE]\\n``.

The ``x-vercel-ai-data-stream: v1`` header tells the client to parse
frames as the AI SDK data stream protocol.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import AsyncGenerator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from pi.agent import (
    AgentEndEvent,
    MessageUpdateEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
)
from pi.ai.types import TextContent, UserMessage

from simple_agent.process.agent_process import AgentProcess, AgentState


class ClientMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ClientMessage]


def convert_to_agent_messages(messages: list[ClientMessage]) -> list:
    """Convert Vercel AI SDK client messages to pi-agent UserMessage format.

    User messages become UserMessage with TextContent.
    Assistant messages pass through as-is.
    """
    agent_messages = []
    for msg in messages:
        if msg.role == "user":
            agent_messages.append(
                UserMessage(
                    content=[TextContent(text=msg.content)],
                    timestamp=int(time.time() * 1000),
                )
            )
    return agent_messages


class AgentEventHandler:
    """Subscribe to AgentProcess events, push v6 stream frames into a queue.

    Maps pi-agent events to AI SDK v6 frame types:

    =====================  ===========================================
    pi-agent event         v6 frame(s)
    =====================  ===========================================
    start()                ``start`` (lifecycle)
    text_delta             ``text-start`` + ``text-delta`` / ``text-delta``
    ToolExecutionStart     ``text-end`` + ``tool-input-start`` + ``tool-input-available``
    ToolExecutionEnd       ``tool-output-available``
    AgentEnd               ``text-end`` + ``finish`` + sentinel
    =====================  ===========================================
    """

    def __init__(self, queue: asyncio.Queue, state: AgentState):
        self._queue = queue
        self._state = state
        self._message_id = f"msg_{uuid.uuid4().hex[:12]}"
        self._in_text_block = False
        self._text_block_id: str | None = None
        self._text_block_counter = 0

    def _emit(self, frame: dict) -> None:
        self._queue.put_nowait(f"data: {json.dumps(frame)}\n")

    def _start_text_block(self) -> None:
        if not self._in_text_block:
            self._text_block_counter += 1
            self._text_block_id = f"txt_{self._text_block_counter}"
            self._emit({"type": "text-start", "id": self._text_block_id})
            self._in_text_block = True

    def _end_text_block(self) -> None:
        if self._in_text_block:
            self._emit({"type": "text-end", "id": self._text_block_id})
            self._in_text_block = False
            self._text_block_id = None

    def start(self) -> None:
        """Emit the lifecycle start frame. Called before agent.run()."""
        self._emit({"type": "start", "messageId": self._message_id})

    def __call__(self, event) -> None:
        if isinstance(event, MessageUpdateEvent):
            ae = event.assistant_message_event
            if ae.type == "text_delta":
                self._start_text_block()
                self._emit({"type": "text-delta", "id": self._text_block_id, "delta": ae.delta})

        elif isinstance(event, ToolExecutionStartEvent):
            self._end_text_block()
            self._emit({
                "type": "tool-input-start",
                "toolCallId": event.tool_call_id,
                "toolName": event.tool_name,
            })
            self._emit({
                "type": "tool-input-available",
                "toolCallId": event.tool_call_id,
                "toolName": event.tool_name,
                "input": event.args,
            })

        elif isinstance(event, ToolExecutionEndEvent):
            result = event.result
            if not isinstance(result, dict):
                if hasattr(result, "__dict__"):
                    result = result.__dict__
                else:
                    result = {"value": str(result)}
            self._emit({
                "type": "tool-output-available",
                "toolCallId": event.tool_call_id,
                "output": result,
            })

        elif isinstance(event, AgentEndEvent):
            self._end_text_block()
            finish_reason = "tool-calls" if self._state.finish_reason else "stop"
            self._emit({"type": "finish", "finishReason": finish_reason})
            self._queue.put_nowait(None)


async def stream_agent_response(
    messages: list[ClientMessage],
    model,
    system_prompt: str,
    tools: list,
) -> AsyncGenerator[str, None]:
    """Run the agent and yield AI SDK v6 data stream protocol lines."""
    queue: asyncio.Queue = asyncio.Queue()
    state = AgentState()
    handler = AgentEventHandler(queue, state)

    handler.start()

    agent = AgentProcess(model)
    agent.subscribe(handler)

    agent_messages = convert_to_agent_messages(messages)
    user_prompt = messages[-1].content if messages else ""

    task = asyncio.create_task(
        agent.run(
            system_prompt=system_prompt,
            messages=agent_messages,
            tools=tools,
            state=state,
            user_prompt=user_prompt,
        )
    )

    while True:
        item = await queue.get()
        if item is None:
            break
        yield item

    yield "data: [DONE]\n"

    await task


def create_chat_router(
    model,
    system_prompt: str = "You are a helpful assistant.",
    tools: list | None = None,
) -> APIRouter:
    """Factory returning an APIRouter with the POST /chat endpoint."""

    router = APIRouter()

    @router.post("/chat")
    async def chat(request: ChatRequest):
        response = StreamingResponse(
            stream_agent_response(
                messages=request.messages,
                model=model,
                system_prompt=system_prompt,
                tools=tools or [],
            ),
            media_type="text/plain; charset=utf-8",
        )
        response.headers["x-vercel-ai-data-stream"] = "v1"
        return response

    return router
