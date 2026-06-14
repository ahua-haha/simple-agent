"""pi-agent CLI - Interactive CLI for pi-agent runtime."""

from __future__ import annotations

import asyncio
import sys

from pi.agent import Agent
from pi.ai import get_model


async def run_interactive() -> None:
    """Run an interactive prompt loop with the agent."""
    model_name = sys.argv[1] if len(sys.argv) > 1 else "anthropic/claude-sonnet-4-5"

    agent = Agent()
    agent.set_model(get_model("anthropic", model_name))
    agent.set_system_prompt("You are a helpful CLI assistant. Respond to user queries.")

    def on_event(event):
        if event.type == "message_update":
            ae = event.assistant_message_event
            if ae.type == "text_delta":
                print(ae.delta, end="", flush=True)
        elif event.type == "agent_end":
            print("\n--- done ---")

    agent.subscribe(on_event)

    print(f"pi-agent CLI (model: {model_name})")
    print("Type 'exit' to quit")
    print("-" * 40)

    while True:
        try:
            user_input = await asyncio.to_thread(input, "You: ")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_input.strip().lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        if not user_input.strip():
            continue

        print("Agent: ", end="", flush=True)
        await agent.prompt(user_input)
        print()


def main() -> None:
    """Main entry point for the pi CLI."""
    asyncio.run(run_interactive())


if __name__ == "__main__":
    main()
