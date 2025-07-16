"""Core business logic for interacting with the OpenAI Assistants API.

This module is responsible for all direct communication with the OpenAI API
and is designed to be independent of the MCP server implementation.
"""

import os
from typing import Optional, Literal
import openai
from openai.types.beta import Assistant, Thread
from openai.types.beta.threads import Run

RunStatus = Literal["completed", "in_progress", "failed", "cancelled", "expired"]


class AssistantManager:
    """Handles interactions with OpenAI's Assistant API."""

    def __init__(self, api_key: str):
        """Initialize the OpenAI client with an explicit API key."""
        if not api_key:
            raise ValueError("OpenAI API key cannot be empty.")
        self.client = openai.OpenAI(api_key=api_key)

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

    async def new_thread(self) -> Thread:
        """Create a new conversation thread."""
        return self.client.beta.threads.create()

    async def list_assistants(self, limit: int = 20) -> list[Assistant]:
        """List available OpenAI assistants."""
        response = self.client.beta.assistants.list(limit=limit)
        return response.data

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