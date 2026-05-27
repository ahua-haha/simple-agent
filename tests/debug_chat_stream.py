"""Debug script: call the chat API endpoint and print raw streaming data.

Usage:
    python tests/debug_chat_stream.py [--url http://localhost:8080/api/chat]

Each raw chunk from the stream is printed with a prefix so you can see
exactly what the server emits over the wire.
"""

from __future__ import annotations

import argparse
import json
import sys

import httpx


async def debug_stream(url: str, prompt: str) -> None:
    """Send a chat request and print raw streaming chunks."""
    payload = {"messages": [{"role": "user", "content": prompt}]}

    print(f"POST {url}")
    print(f"Request: {json.dumps(payload)}")
    print("-" * 60)

    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
        async with client.stream("POST", url, json=payload) as response:
            print(f"Status: {response.status_code}")
            print(f"Headers:")
            for key, value in response.headers.items():
                print(f"  {key}: {value}")
            print("-" * 60)
            print("Stream body (raw chunks):\n")

            chunk_num = 0
            async for chunk in response.aiter_bytes():
                chunk_num += 1
                decoded = chunk.decode("utf-8", errors="replace")
                print(f"--- chunk {chunk_num} ({len(chunk)} bytes) ---")
                print(decoded, end="")

            print(f"\n{'=' * 60}")
            print(f"Total chunks: {chunk_num}")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Debug the chat API stream")
    parser.add_argument(
        "--url",
        default="http://localhost:8080/api/chat",
        help="Chat endpoint URL (default: http://localhost:8080/api/chat)",
    )
    parser.add_argument(
        "--prompt",
        default="Say hello in exactly 3 words.",
        help="User prompt to send",
    )
    args = parser.parse_args()

    try:
        await debug_stream(args.url, args.prompt)
    except httpx.ConnectError:
        print(f"ERROR: Could not connect to {args.url}", file=sys.stderr)
        print("Is the server running? Start it with:", file=sys.stderr)
        print(
            "  simple-agent-web --model-provider deepseek --model-name deepseek-v4-pro",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
