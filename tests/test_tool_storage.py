"""Tests for Database storage module."""

from __future__ import annotations

import os
import tempfile

from simple_agent.db.db import Database


class TestDatabaseInit:
    """Tests for Database._init_db() method."""

    def test_init_db_creates_tables(self):
        """_init_db() should create tables in SQLite database."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            db = Database(db_path)

            import sqlite3
            conn = sqlite3.connect(db_path)
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = [row[0] for row in cursor.fetchall()]
            conn.close()

            assert "runnertoolcallrecord" in tables
            assert "taskrecord" in tables
            assert "managedtaskrecord" not in tables
        finally:
            os.unlink(db_path)

    def test_init_db_enables_wal_mode(self):
        """_init_db() should enable WAL mode for concurrent reads."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            db = Database(db_path)

            import sqlite3
            conn = sqlite3.connect(db_path)
            cursor = conn.execute("PRAGMA journal_mode")
            mode = cursor.fetchone()[0]
            conn.close()

            assert mode == "wal"
        finally:
            os.unlink(db_path)
