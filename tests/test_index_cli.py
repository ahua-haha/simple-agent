"""Tests for repo-index CLI."""

from __future__ import annotations

import subprocess

from simple_agent.index import AgentIndex


def test_repo_index_tree_prints_agent_index_tree(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    src_dir = repo / "src"
    src_dir.mkdir()
    app_file = src_dir / "app.py"
    app_file.write_text("print('hello')\n")
    db_path = tmp_path / "index.db"
    index = AgentIndex(str(db_path), base_dir=str(repo))
    index.upsert_entry(
        "src/app.py",
        {"kind": "file", "description": "Application entry point"},
    )

    result = subprocess.run(
        [
            "python",
            "-m",
            "simple_agent.cli.repo_index",
            "tree",
            "--db",
            str(db_path),
            "--repo",
            str(repo),
            "--path",
            "src",
            "--depth",
            "2",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "src/" in result.stdout
    assert "app.py  # Application entry point" in result.stdout

