"""Interactive playback for a session JSONL run log.

Usage:

    uv run python tests/manual_session_log_playback.py logs/session_runs/session_abc.jsonl
"""

from simple_agent.cli.session_log_playback import main


if __name__ == "__main__":
    main()
