"""Core business logic for interacting with the OpenAI Assistants API.

This module is responsible for all direct communication with the OpenAI API
and is designed to be independent of the MCP server implementation.
"""

import os
from typing import Optional, Literal
import openai
from openai.types.beta import Assistant, Thread
from openai.types.beta.threads import Run
from .thread_store import ThreadStore
import sqlite3

RunStatus = Literal["completed", "in_progress", "failed", "cancelled", "expired"]


class AssistantManager:
    """Handles interactions with OpenAI's Assistant API."""

    def __init__(self, api_key: str, db_path: str):
        """Initialize the OpenAI client with an explicit API key."""
        if not api_key:
            raise ValueError("OpenAI API key cannot be empty.")
        self.client = openai.OpenAI(api_key=api_key)
        self.thread_store = ThreadStore(db_path)
        self.thread_store.initialize_database()

    async def create_assistant(
        self,
        name: str,
        instructions: str,
        model: str = "gpt-4o"
    ) -> Assistant:
        """Create a new OpenAI assistant."""
        return self.client.beta.assistants.create(
            name=name,
            instructions=instructions,
            model=model
        )

    async def create_new_assistant_thread(
        self, name: str, description: Optional[str] = None
    ) -> Thread:
        """Creates a new, persistent conversation thread."""
        metadata = {
            "name": name,
            "description": description or ""
        }
        thread = self.client.beta.threads.create(metadata=metadata)
        self.thread_store.add_thread(thread.id, name, description)
        return thread

    async def list_assistants(self, limit: int = 20) -> list[Assistant]:
        """List available OpenAI assistants."""
        response = self.client.beta.assistants.list(limit=limit)
        return response.data

    def list_threads(self) -> list[sqlite3.Row]:
        """List all managed threads from the local database."""
        return self.thread_store.list_threads()

    async def retrieve_assistant(self, assistant_id: str) -> Assistant:
        """Get details about a specific assistant."""
        return self.client.beta.assistants.retrieve(assistant_id)

    async def update_assistant(
        self,
        assistant_id: str,
        name: Optional[str] = None,
        instructions: Optional[str] = None,
        model: Optional[str] = None
    ) -> Assistant:
        """Update an existing assistant's configuration."""
        update_params = {}
        if name is not None:
            update_params["name"] = name
        if instructions is not None:
            update_params["instructions"] = instructions
        if model is not None:
            update_params["model"] = model

        return self.client.beta.assistants.update(
            assistant_id=assistant_id,
            **update_params
        )

    async def update_thread(
        self, thread_id: str, name: Optional[str], description: Optional[str]
    ):
        """Update the metadata of a thread on OpenAI and in the local DB."""
        metadata = {
            "name": name,
            "description": description or ""
        }
        # First, update the thread on OpenAI's servers
        updated_thread = self.client.beta.threads.update(
            thread_id=thread_id,
            metadata=metadata
        )
        # Then, update the local database
        self.thread_store.update_thread_metadata(thread_id, name, description)
        return updated_thread

    async def delete_thread(self, thread_id: str):
        """Delete a thread from OpenAI and the local database."""
        # First, delete the thread from OpenAI's servers
        result = self.client.beta.threads.delete(thread_id)
        # If successful, delete from the local database
        if result.deleted:
            self.thread_store.delete_thread(thread_id)
        return result

    async def run_thread(
        self,
        thread_id: str,
        assistant_id: str,
        message: str
    ):
        """
        Sends a message to a thread and streams the assistant's response.
        This is an async generator that yields events from the run.
        """
        # Update the last used timestamp
        self.thread_store.update_thread_last_used(thread_id)

        # Add the user's message to the thread
        self.client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=message
        )

        # Stream the assistant's response
        stream = self.client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=assistant_id,
            stream=True
        )
        for event in stream:
            yield event 