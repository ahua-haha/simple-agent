"""FastAPI app for browsing the task tree and serving the agent chat API."""

import os

from fastapi import FastAPI

from simple_agent.db.db import Database
from simple_agent.models import register_custom_models
from simple_agent.session.session_manager import DEFAULT_COOLDOWN_SECONDS, SessionManager
from simple_agent.web.chat_api import create_chat_router
from simple_agent.web.session_api import create_session_router
from simple_agent.web.task_api import router as task_router


_db: Database | None = None


def get_db() -> Database:
    if _db is None:
        raise RuntimeError("Database not initialized.")
    return _db


def create_app(
    db_path: str,
    model=None,
    system_prompt: str = "You are a helpful assistant.",
    tools: list | None = None,
    sessions_dir: str = "./sessions",
    cooldown_seconds: int = DEFAULT_COOLDOWN_SECONDS,
) -> FastAPI:
    global _db
    _db = Database(db_path)

    app = FastAPI(title="Simple Agent Web")

    # SessionManager — always available, even without a model
    session_manager = SessionManager(
        sessions_dir=sessions_dir,
        cooldown_seconds=cooldown_seconds,
    )
    app.state.session_manager = session_manager
    app.state.get_db = get_db

    # Session API — always registered
    session_router = create_session_router()
    app.include_router(session_router, prefix="/api")

    # Task tree HTML API — always registered
    app.include_router(task_router)

    # Legacy stateless chat — only when a model is explicitly provided
    if model is not None:
        chat_router = create_chat_router(
            model=model,
            system_prompt=system_prompt,
            tools=tools,
        )
        app.include_router(chat_router, prefix="/api")

    return app


def main() -> None:
    """Entry point for ``python -m simple_agent.web.app``."""
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Simple Agent web server")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on (default: 8080)")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
    parser.add_argument("--db", default="./data/tool_log.db", help="Path to SQLite database (default: ./data/tool_log.db)")
    parser.add_argument("--model-provider", default=None, help="Model provider for chat API (e.g. anthropic, deepseek)")
    parser.add_argument("--model-name", default=None, help="Model name for chat API (e.g. claude-sonnet-4-5)")
    parser.add_argument("--system-prompt", default="You are a helpful assistant.", help="System prompt for chat API")
    parser.add_argument(
        "--cooldown-seconds",
        type=int,
        default=int(os.environ.get("SESSION_COOLDOWN_SECONDS", str(DEFAULT_COOLDOWN_SECONDS))),
        help=f"Seconds before an idle session parks to disk (default: {DEFAULT_COOLDOWN_SECONDS})",
    )
    parser.add_argument(
        "--sessions-dir",
        default="./sessions",
        help="Directory for session persistence (default: ./sessions)",
    )
    args = parser.parse_args()

    register_custom_models()
    model = None
    if args.model_provider and args.model_name:
        from pi.ai import get_model
        from pi.ai.models import _model_registry

        model = get_model(args.model_provider, args.model_name)
        if model is None:
            available = ", ".join(sorted(_model_registry.keys()))
            print(
                f"Error: model '{args.model_name}' not found for provider "
                f"'{args.model_provider}'.\n"
                f"Available providers: {available}",
                flush=True,
            )
            import sys

            sys.exit(1)

    app = create_app(
        db_path=args.db,
        model=model,
        system_prompt=args.system_prompt,
        sessions_dir=args.sessions_dir,
        cooldown_seconds=args.cooldown_seconds,
    )

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
