"""Pytest test suite for the MCP OpenAI Assistant server."""

import os
import asyncio
import pytest
from unittest.mock import AsyncMock, patch
from dotenv import load_dotenv
from fastmcp import Client, FastMCP, Context
from mcp_simple_openai_assistant.assistant_manager import AssistantManager
from textwrap import dedent
from typing import Optional

# Load environment variables for the test session
load_dotenv()

# --- Fixtures ---

@pytest.fixture(scope="session")
def api_key() -> str:
    """Fixture to provide the OpenAI API key and skip tests if not found."""
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        pytest.skip("OPENAI_API_KEY not found in environment, skipping integration tests.")
    return key

@pytest.fixture(scope="session")
def test_assistant_name() -> str:
    """Provides a unique name for the assistant created during the test run."""
    return "Test Assistant - Pytest"

@pytest.fixture
def client(api_key: str) -> Client:
    """
    Provides a FastMCP client configured to talk to a fresh, in-memory server
    instance for each test.
    """
    # Create a completely new FastMCP app for each test to ensure isolation
    test_app = FastMCP(name="openai-assistant-test")
    
    # Initialize a new manager with the API key for this test
    # The user has specified to use a memory DB for these tests.
    test_manager = AssistantManager(api_key=api_key, db_path=":memory:")

    # This is a bit of a workaround to register tools on a new app instance
    # within a test. We define the tools inside the fixture.
    @test_app.tool(annotations={"title": "List OpenAI Assistants", "readOnlyHint": True})
    async def list_assistants(limit: int = 20) -> str:
        assistants = await test_manager.list_assistants(limit)
        if not assistants: return "No assistants found."
        assistant_list = [
            dedent(f"""
            ID: {a.id}
            Name: {a.name}
            Model: {a.model}""")
            for a in assistants
        ]
        return "Available Assistants:\\n\\n" + "\\n---\\n".join(assistant_list)

    @test_app.tool(annotations={"title": "Create OpenAI Assistant", "readOnlyHint": False})
    async def create_assistant(name: str, instructions: str, model: str = "gpt-4o") -> str:
        result = await test_manager.create_assistant(name, instructions, model)
        return f"Created assistant '{result.name}' with ID: {result.id}"

    @test_app.tool(annotations={"title": "Retrieve OpenAI Assistant", "readOnlyHint": True})
    async def retrieve_assistant(assistant_id: str) -> str:
        result = await test_manager.retrieve_assistant(assistant_id)
        return dedent(f"""
        Assistant Details:
        ID: {result.id}
        Name: {result.name}
        Model: {result.model}
        Instructions: {result.instructions}
        """)
    
    @test_app.tool(annotations={"title": "Create New Assistant Thread", "readOnlyHint": False})
    async def create_new_assistant_thread(
        name: str, description: Optional[str] = None
    ) -> str:
        thread = await test_manager.create_new_assistant_thread(name, description)
        return f"Created new thread '{name}' with ID: {thread.id}"

    @test_app.tool(annotations={"title": "List Managed Threads", "readOnlyHint": True})
    async def list_threads() -> str:
        threads = test_manager.list_threads()
        if not threads:
            return "No managed threads found."
        thread_list = [
            dedent(f"""
            Thread ID: {t['thread_id']}
            Name: {t['name']}
            Description: {t['description']}
            Last Used: {t['last_used_at']}
            """)
            for t in threads
        ]
        return "Managed Threads:\\n\\n" + "\\n---\\n".join(thread_list)

    @test_app.tool(annotations={"title": "Delete Managed Thread", "readOnlyHint": False})
    async def delete_thread(thread_id: str) -> str:
        result = await test_manager.delete_thread(thread_id)
        if result.deleted:
            return f"Successfully deleted thread {thread_id}."
        else:
            return f"Failed to delete thread {thread_id} on the server."

    @test_app.tool(annotations={"title": "Ask Assistant in Thread and Stream Response", "readOnlyHint": False})
    async def ask_assistant_in_thread(thread_id: str, assistant_id: str, message: str, ctx: Context) -> str:
        final_message = ""
        await ctx.report_progress(progress=0, message="Starting assistant run...")
        async for event in test_manager.run_thread(thread_id, assistant_id, message):
            if event.event == 'thread.message.delta':
                text_delta = event.data.delta.content[0].text
                final_message += text_delta.value
                await ctx.report_progress(progress=50, message=f"Assistant writing: {final_message}")
            elif event.event == 'thread.run.step.created':
                await ctx.report_progress(progress=25, message="Assistant is performing a step...")
        
        await ctx.report_progress(progress=100, message="Run complete.")
        return final_message

    return Client(test_app)


# --- Test Cases ---

@pytest.mark.asyncio
async def test_list_assistants(client: Client):
    """Test the list_assistants tool."""
    async with client:
        result = await client.call_tool("list_assistants")
        assert "Available Assistants" in result.data or "No assistants found" in result.data

@pytest.mark.asyncio
async def test_create_and_retrieve_assistant(client: Client, test_assistant_name: str):
    """Test creating a new assistant and then retrieving it."""
    async with client:
        # Create
        create_result = await client.call_tool(
            "create_assistant",
            {
                "name": test_assistant_name,
                "instructions": "A test assistant for pytest.",
                "model": "gpt-4o"
            }
        )
        assert f"Created assistant '{test_assistant_name}'" in create_result.data
        
        # Extract the ID from the response text
        assistant_id = create_result.data.split("ID: ")[-1]
        assert assistant_id is not None

        # Retrieve
        retrieve_result = await client.call_tool(
            "retrieve_assistant",
            {"assistant_id": assistant_id}
        )
        assert f"ID: {assistant_id}" in retrieve_result.data
        assert f"Name: {test_assistant_name}" in retrieve_result.data

@pytest.mark.asyncio
async def test_streaming_conversation_flow(client: Client, test_assistant_name: str):
    """
    Tests the new streaming conversation flow:
    1. Find or create the test assistant.
    2. Create a new thread.
    3. Call `run_thread` and verify progress messages are received.
    """
    # Use a list to capture progress messages from the handler
    progress_updates = []
    async def progress_handler(progress: int, total: int | None, message: str | None):
        if message:
            progress_updates.append(message)

    async with client:
        # 1. Find or create the test assistant
        list_res = await client.call_tool("list_assistants", {"limit": 100})
        
        assistant_id = None
        for block in list_res.data.split('---'):
            if f"Name: {test_assistant_name}" in block:
                lines = block.strip().split('\\n')
                for line in lines:
                    if line.startswith("ID: "):
                        assistant_id = line.split("ID: ")[1].strip()
                        break
            if assistant_id:
                break
        
        if not assistant_id:
            create_res = await client.call_tool("create_assistant", {"name": test_assistant_name, "instructions": "Test bot."})
            assistant_id = create_res.data.split("ID: ")[-1]

        assert assistant_id

        # 2. Create a thread
        thread_res = await client.call_tool("create_new_assistant_thread", {"name": "Test Thread", "description": "A test thread"})
        thread_id = thread_res.data.split("ID: ")[-1]
        assert thread_id

        # 3. Call run_thread and stream the response
        final_result = await client.call_tool(
            "ask_assistant_in_thread",
            {
                "thread_id": thread_id,
                "assistant_id": assistant_id,
                "message": "Hello! What is 2 + 2? Please explain your steps."
            },
            progress_handler=progress_handler
        )

        # 4. Assertions
        assert "4" in final_result.data or "four" in final_result.data.lower()
        assert len(progress_updates) > 2
        assert "Starting assistant run..." in progress_updates[0]
        assert "Run complete." in progress_updates[-1]

@pytest.mark.asyncio
async def test_thread_management_lifecycle(client: Client):
    """
    Tests the full lifecycle of a managed thread by making real API calls:
    1. Create a new thread via the tool.
    2. Verify it's in the local database.
    3. Delete the thread via the tool.
    4. Verify it has been removed from the local database.
    """
    thread_id = None
    thread_name = "Test Full Lifecycle"
    try:
        async with client:
            # 1. Verify no threads with this name exist initially
            initial_list = await client.call_tool("list_threads")
            assert thread_name not in initial_list.data

            # 2. Create a new thread (real API call)
            create_result = await client.call_tool(
                "create_new_assistant_thread",
                {"name": thread_name, "description": "Testing the full cycle."}
            )
            assert f"Created new thread '{thread_name}'" in create_result.data
            thread_id = create_result.data.split("ID: ")[-1]
            assert thread_id.startswith("thread_")

            # 3. List threads and verify the new thread is present in the DB
            list_after_create = await client.call_tool("list_threads")
            assert thread_id in list_after_create.data
            assert thread_name in list_after_create.data

    finally:
        # 4. Cleanup: Delete the thread (real API call)
        if thread_id:
            async with client:
                delete_result = await client.call_tool("delete_thread", {"thread_id": thread_id})
                assert "Successfully deleted" in delete_result.data

                # 5. Verify it's gone from the local DB
                list_after_delete = await client.call_tool("list_threads")
                assert thread_id not in list_after_delete.data 