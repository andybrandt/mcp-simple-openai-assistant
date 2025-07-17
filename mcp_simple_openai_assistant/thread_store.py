"""A module for managing a local SQLite database of OpenAI Assistant threads.

This module provides a ThreadStore class that encapsulates all database
operations, including creating the database and table, as well as adding,
updating, listing, and deleting thread records. This allows the application
to manage conversation threads persistently.
"""

import sqlite3
import os
from datetime import datetime, timezone

class ThreadStore:
    """Manages the persistence of thread data in a local SQLite database."""

    def __init__(self, db_path: str):
        """Initializes the ThreadStore with a path to the SQLite database file.

        Args:
            db_path: The file path for the SQLite database.
        """
        if not db_path:
            raise ValueError("Database path cannot be empty.")
        self.db_path = db_path
        self._conn = None

    def _get_connection(self) -> sqlite3.Connection:
        """Returns a SQLite database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def initialize_database(self):
        """Creates the database and the threads table if they do not exist."""
        # Ensure the directory for the database file exists, but skip for in-memory DB
        if self.db_path != ':memory:':
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS threads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT NOT NULL UNIQUE,
                name TEXT NOT NULL,
                description TEXT,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                last_used_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()

    def add_thread(self, thread_id: str, name: str, description: str | None) -> int:
        """Adds a new thread record to the database.

        Args:
            thread_id: The unique identifier for the thread from OpenAI.
            name: A user-defined name for the thread.
            description: A user-defined description for the thread.

        Returns:
            The row ID of the newly inserted thread.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO threads (thread_id, name, description, last_used_at)
            VALUES (?, ?, ?, ?)
        """, (thread_id, name, description, datetime.now(timezone.utc)))
        conn.commit()
        return cursor.lastrowid

    def list_threads(self) -> list[sqlite3.Row]:
        """Retrieves all thread records from the database.

        Returns:
            A list of rows, where each row is a dictionary-like object
            representing a thread.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM threads ORDER BY last_used_at DESC")
        return cursor.fetchall()

    def update_thread_metadata(self, thread_id: str, name: str, description: str | None):
        """Updates the name and description of a specific thread.

        Args:
            thread_id: The ID of the thread to update.
            name: The new name for the thread.
            description: The new description for the thread.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE threads
            SET name = ?, description = ?, last_used_at = ?
            WHERE thread_id = ?
        """, (name, description, datetime.now(timezone.utc), thread_id))
        conn.commit()

    def update_thread_last_used(self, thread_id: str):
        """Updates the last_used_at timestamp for a specific thread.

        Args:
            thread_id: The ID of the thread to update.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE threads
            SET last_used_at = ?
            WHERE thread_id = ?
        """, (datetime.now(timezone.utc), thread_id))
        conn.commit()

    def delete_thread(self, thread_id: str):
        """Deletes a thread record from the database by its thread_id.

        Args:
            thread_id: The ID of the thread to delete.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM threads WHERE thread_id = ?", (thread_id,))
        conn.commit()

    def close(self):
        """Closes the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None 