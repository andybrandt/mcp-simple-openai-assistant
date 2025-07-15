"""Pytest test suite for the MCP OpenAI Assistant server."""

import os
import asyncio
import pytest
from dotenv import load_dotenv
from fastmcp import Client, FastMCP
from mcp_simple_openai_assistant.app import app as mcp_app, manager
from mcp_simple_openai_assistant.assistant_manager import AssistantManager

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
    test_manager = AssistantManager(api_key=api_key)

    # This is a bit of a workaround to register tools on a new app instance
    # within a test. We define the tools inside the fixture.
    @test_app.tool(annotations={"title": "List OpenAI Assistants", "readOnlyHint": True})
    async def list_assistants(limit: int = 20) -> str:
        assistants = await test_manager.list_assistants(limit)
        if not assistants: return "No assistants found."
        assistant_list = [f"ID: {a.id}, Name: {a.name}" for a in assistants]
        return "Available Assistants:\n" + "\n".join(assistant_list)

    @test_app.tool(annotations={"title": "Create OpenAI Assistant", "readOnlyHint": False})
    async def create_assistant(name: str, instructions: str, model: str = "gpt-4o") -> str:
        result = await test_manager.create_assistant(name, instructions, model)
        return f"Created assistant '{result.name}' with ID: {result.id}"

    @test_app.tool(annotations={"title": "Retrieve OpenAI Assistant", "readOnlyHint": True})
    async def retrieve_assistant(assistant_id: str) -> str:
        result = await test_manager.retrieve_assistant(assistant_id)
        return f"ID: {result.id}\nName: {result.name}\nInstructions: {result.instructions}"
    
    @test_app.tool(annotations={"title": "Create New Thread", "readOnlyHint": False})
    async def new_thread() -> str:
        result = await test_manager.new_thread()
        return f"Created new thread with ID: {result.id}"

    @test_app.tool(annotations={"title": "Send Message and Start Run", "readOnlyHint": False})
    async def send_message(thread_id: str, assistant_id: str, message: str) -> str:
        run = await test_manager.send_message(thread_id, assistant_id, message)
        return f"Message sent. Run {run.id} started."

    @test_app.tool(annotations={"title": "Check Assistant Response", "readOnlyHint": True})
    async def check_response(thread_id: str) -> str:
        status, response = await test_manager.check_response(thread_id)
        if status == "completed": return response
        return f"Run status is: {status}"

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
async def test_full_conversation_flow(client: Client, test_assistant_name: str):
    """
    Tests the full, legacy conversation flow:
    1. Find or create the test assistant.
    2. Create a new thread.
    3. Send a message.
    4. Poll with check_response until the response is complete.
    """
    async with client:
        # 1. Find or create the test assistant
        list_res = await client.call_tool("list_assistants", {"limit": 100})
        
        assistant_id = None
        # Find the assistant in the list and parse its ID
        for block in list_res.data.split('---'):
            if f"Name: {test_assistant_name}" in block:
                lines = block.strip().split('\\n')
                for line in lines:
                    if line.startswith("ID: "):
                        assistant_id = line.split("ID: ")[1]
                        break
            if assistant_id:
                break
        
        if not assistant_id:
            create_res = await client.call_tool("create_assistant", {"name": test_assistant_name, "instructions": "Test bot."})
            assistant_id = create_res.data.split("ID: ")[-1]

        assert assistant_id

        # 2. Create a thread
        thread_res = await client.call_tool("new_thread")
        thread_id = thread_res.data.split("ID: ")[-1]
        assert thread_id

        # 3. Send a message
        await client.call_tool(
            "send_message",
            {
                "thread_id": thread_id,
                "assistant_id": assistant_id,
                "message": "Hello! What is 2 + 2?"
            }
        )
        
        # 4. Poll for response
        for _ in range(10): # Poll up to 10 times
            check_res = await client.call_tool("check_response", {"thread_id": thread_id})
            if not str(check_res.data).startswith("Run status is:"):
                # We got a final answer
                assert "4" in str(check_res.data) or "four" in str(check_res.data).lower()
                return
            await asyncio.sleep(2) # Wait before polling again

        pytest.fail("Assistant response timed out after multiple checks.") 