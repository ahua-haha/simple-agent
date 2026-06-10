"""FastAPI app for browsing the task tree and serving the session API."""

import os

from fastapi import FastAPI

from simple_agent.session.session_manager import SessionManager
from simple_agent.web.session_api import create_session_router


def create_app(
    sessions_dir: str = "./sessions",
    workspace_dir: str | None = None,
) -> FastAPI:
    app = FastAPI(title="Simple Agent Web")

    app.state.workspace_dir = workspace_dir or os.getcwd()
    session_manager = SessionManager(sessions_dir=sessions_dir)
    app.state.session_manager = session_manager

    session_router = create_session_router()
    app.include_router(session_router, prefix="/api")

    return app


def main() -> None:
    """Entry point for ``python -m simple_agent.web.app``."""
    import argparse
    import logging

    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)-8s %(message)s",
    )

    parser = argparse.ArgumentParser(description="Simple Agent web server")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on (default: 8080)")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
    parser.add_argument(
        "--sessions-dir",
        default="./sessions",
        help="Directory for session persistence (default: ./sessions)",
    )
    parser.add_argument(
        "--workspace-dir",
        default=os.getcwd(),
        help="Directory where agent tools operate (default: current working directory)",
    )
    args = parser.parse_args()

    app = create_app(
        sessions_dir=args.sessions_dir,
        workspace_dir=args.workspace_dir,
    )

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
