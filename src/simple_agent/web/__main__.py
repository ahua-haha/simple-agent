"""Entry point for the web UI debug server."""

import argparse

import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Simple Agent web debug UI")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on (default: 8080)")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
    parser.add_argument("--db", default="./data/tool_log.db", help="Path to SQLite database (default: ./data/tool_log.db)")
    args = parser.parse_args()

    from simple_agent.web.app import create_app
    app = create_app(args.db)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
