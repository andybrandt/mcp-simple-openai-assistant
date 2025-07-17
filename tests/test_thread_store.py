"""Unit tests for the ThreadStore class.

These tests use an in-memory SQLite database to ensure that the
database logic for managing threads is correct, isolated, and fast.
"""

import unittest
import sqlite3
from mcp_simple_openai_assistant.thread_store import ThreadStore

class TestThreadStore(unittest.TestCase):
    """Test suite for the ThreadStore."""

    def setUp(self):
        """Set up an in-memory database and a ThreadStore instance for each test."""
        # Use ':memory:' for an in-memory SQLite database
        self.store = ThreadStore(':memory:')
        self.store.initialize_database()

    def tearDown(self):
        """Close the database connection after each test."""
        self.store.close()

    def test_add_and_list_threads(self):
        """Verify that a thread can be added and then listed."""
        thread_id = "thread_abc123"
        name = "Test Thread"
        description = "A thread for testing."

        self.store.add_thread(thread_id, name, description)

        threads = self.store.list_threads()
        self.assertEqual(len(threads), 1)
        self.assertEqual(threads[0]['thread_id'], thread_id)
        self.assertEqual(threads[0]['name'], name)
        self.assertEqual(threads[0]['description'], description)

    def test_add_thread_is_unique(self):
        """Verify that adding a thread with a duplicate thread_id raises an error."""
        thread_id = "thread_xyz789"
        self.store.add_thread(thread_id, "Unique 1", "First one")
        with self.assertRaises(sqlite3.IntegrityError):
            self.store.add_thread(thread_id, "Unique 2", "Second one")

    def test_update_thread_metadata(self):
        """Verify that a thread's metadata can be updated correctly."""
        thread_id = "thread_update_me"
        self.store.add_thread(thread_id, "Original Name", "Original Desc")

        new_name = "Updated Name"
        new_description = "Updated Description"
        self.store.update_thread_metadata(thread_id, new_name, new_description)

        threads = self.store.list_threads()
        self.assertEqual(threads[0]['name'], new_name)
        self.assertEqual(threads[0]['description'], new_description)

    def test_update_thread_last_used(self):
        """Verify that the last_used_at timestamp is updated."""
        thread_id = "thread_timestamp"
        self.store.add_thread(thread_id, "Timestamp Test", "Testing last used.")
        
        threads_before = self.store.list_threads()
        first_timestamp = threads_before[0]['last_used_at']

        # In a real scenario, there would be a delay here.
        # For testing, we can just call the update and check that it runs.
        self.store.update_thread_last_used(thread_id)
        
        threads_after = self.store.list_threads()
        second_timestamp = threads_after[0]['last_used_at']
        
        self.assertNotEqual(first_timestamp, second_timestamp)

    def test_delete_thread(self):
        """Verify that a thread can be deleted from the database."""
        thread_id_to_keep = "thread_keeper"
        thread_id_to_delete = "thread_goner"

        self.store.add_thread(thread_id_to_keep, "Keeper", "Keep this one.")
        self.store.add_thread(thread_id_to_delete, "Goner", "Delete this one.")

        self.store.delete_thread(thread_id_to_delete)

        threads = self.store.list_threads()
        self.assertEqual(len(threads), 1)
        self.assertEqual(threads[0]['thread_id'], thread_id_to_keep)

if __name__ == '__main__':
    unittest.main() 