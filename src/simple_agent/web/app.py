"""FastAPI app for browsing the task tree and serving the session API."""

import os

from fastapi import FastAPI

from simple_agent.db.db import Database
from simple_agent.session.session_manager import DEFAULT_COOLDOWN_SECONDS, SessionManager
from simple_agent.web.session_api import create_session_router
from simple_agent.web.task_api import router as task_router


_db: Database | None = None


def get_db() -> Database:
    if _db is None:
        raise RuntimeError("Database not initialized.")
    return _db


def create_app(
    db_path: str,
    sessions_dir: str = "./sessions",
    cooldown_seconds: int = DEFAULT_COOLDOWN_SECONDS,
) -> FastAPI:
    global _db
    _db = Database(db_path)

    app = FastAPI(title="Simple Agent Web")

    session_manager = SessionManager(
        sessions_dir=sessions_dir,
        cooldown_seconds=cooldown_seconds,
    )
    app.state.session_manager = session_manager
    app.state.get_db = get_db

    session_router = create_session_router()
    app.include_router(session_router, prefix="/api")

    app.include_router(task_router)

    return app


def main() -> None:
    """Entry point for ``python -m simple_agent.web.app``."""
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Simple Agent web server")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on (default: 8080)")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
    parser.add_argument("--db", default="./data/tool_log.db", help="Path to SQLite database (default: ./data/tool_log.db)")
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

    app = create_app(
        db_path=args.db,
        sessions_dir=args.sessions_dir,
        cooldown_seconds=args.cooldown_seconds,
    )

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
