"""Entry point for the web UI debug server."""

import argparse

import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Simple Agent web debug UI")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on (default: 8080)")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
    parser.add_argument("--db", default="./data/tool_log.db", help="Path to SQLite database (default: ./data/tool_log.db)")
    parser.add_argument("--model-provider", default=None, help="Model provider for chat API (e.g. anthropic, deepseek)")
    parser.add_argument("--model-name", default=None, help="Model name for chat API (e.g. claude-sonnet-4-5)")
    parser.add_argument("--system-prompt", default="You are a helpful assistant.", help="System prompt for chat API")
    args = parser.parse_args()

    from simple_agent.web.app import create_app

    model = None
    if args.model_provider and args.model_name:
        from pi.ai import get_model
        model = get_model(args.model_provider, args.model_name)

    app = create_app(
        db_path=args.db,
        model=model,
        system_prompt=args.system_prompt,
    )

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
