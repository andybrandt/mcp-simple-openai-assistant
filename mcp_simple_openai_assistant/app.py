"""FastMCP server application definition.

This module initializes the FastMCP application and uses decorators
to expose the business logic from the AssistantManager as MCP tools.
"""

from textwrap import dedent
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from .assistant_manager import AssistantManager

# Initialize the FastMCP application
app = FastMCP(name="openai-assistant")

# This will be initialized in the main entry point after the env is loaded
manager: AssistantManager | None = None


@app.tool(
    annotations={
        "title": "Create OpenAI Assistant",
        "readOnlyHint": False
    }
)
async def create_assistant(name: str, instructions: str, model: str = "gpt-4o") -> str:
    """
    Create a new OpenAI assistant.

    You can provide instructions that this assistant will follow and specify a model.
    NOTE: It is recommended to check existing assistants with list_assistants before creating a new one.
    """
    if not manager:
        raise ToolError("AssistantManager not initialized.")
    try:
        result = await manager.create_assistant(name, instructions, model)
        return f"Created assistant '{result.name}' with ID: {result.id}"
    except Exception as e:
        raise ToolError(f"Failed to create assistant: {e}")

@app.tool(
    annotations={
        "title": "Create New Thread",
        "readOnlyHint": False
    }
)
async def new_thread() -> str:
    """Creates a new conversation thread for interacting with an assistant."""
    if not manager:
        raise ToolError("AssistantManager not initialized.")
    try:
        result = await manager.new_thread()
        return f"Created new thread with ID: {result.id}"
    except Exception as e:
        raise ToolError(f"Failed to create thread: {e}")

@app.tool(
    annotations={
        "title": "Send Message and Start Run",
        "readOnlyHint": False
    }
)
async def send_message(thread_id: str, assistant_id: str, message: str) -> str:
    """
    Send a message to an assistant and start processing.
    The response will not be immediately available - use check_response to get it when ready.
    """
    if not manager:
        raise ToolError("AssistantManager not initialized.")
    try:
        run = await manager.send_message(thread_id, assistant_id, message)
        return f"Message sent. Run {run.id} started in thread {thread_id}. Use check_response to get the result."
    except Exception as e:
        raise ToolError(f"Failed to send message: {e}")


@app.tool(
    annotations={
        "title": "Check Assistant Response",
        "readOnlyHint": True
    }
)
async def check_response(thread_id: str) -> str:
    """
    Check if an assistant's response is ready in the thread.
    Returns 'in_progress' status or the actual response if ready.
    """
    if not manager:
        raise ToolError("AssistantManager not initialized.")
    try:
        status, response = await manager.check_response(thread_id)
        if status == "completed":
            return response
        else:
            return f"Run status is: {status}. Please try again shortly."
    except Exception as e:
        raise ToolError(f"Failed to check response: {e}")

@app.tool(
    annotations={
        "title": "List OpenAI Assistants",
        "readOnlyHint": True
    }
)
async def list_assistants(limit: int = 20) -> str:
    """
    List all available OpenAI assistants associated with the API key.
    Returns a list of assistants with their IDs, names, and configurations.
    """
    if not manager:
        raise ToolError("AssistantManager not initialized.")
    try:
        assistants = await manager.list_assistants(limit)
        if not assistants:
            return "No assistants found."

        assistant_list = [
            dedent(f"""
            ID: {a.id}
            Name: {a.name}
            Model: {a.model}""")
            for a in assistants
        ]
        return "Available Assistants:\\n\\n" + "\\n---\\n".join(assistant_list)
    except Exception as e:
        raise ToolError(f"Failed to list assistants: {e}")

@app.tool(
    annotations={
        "title": "Retrieve OpenAI Assistant",
        "readOnlyHint": True
    }
)
async def retrieve_assistant(assistant_id: str) -> str:
    """Get detailed information about a specific assistant."""
    if not manager:
        raise ToolError("AssistantManager not initialized.")
    try:
        result = await manager.retrieve_assistant(assistant_id)
        return dedent(f"""
        Assistant Details:
        ID: {result.id}
        Name: {result.name}
        Model: {result.model}
        Instructions: {result.instructions}
        """)
    except Exception as e:
        raise ToolError(f"Failed to retrieve assistant {assistant_id}: {e}")

@app.tool(
    annotations={
        "title": "Update OpenAI Assistant",
        "readOnlyHint": False
    }
)
async def update_assistant(
    assistant_id: str,
    name: str = None,
    instructions: str = None,
    model: str = None
) -> str:
    """
    Modify an existing assistant's name, instructions, or model.
    At least one optional parameter must be provided.
    """
    if not manager:
        raise ToolError("AssistantManager not initialized.")
    if not any([name, instructions, model]):
        raise ToolError("You must provide at least one field to update (name, instructions, or model).")
    try:
        result = await manager.update_assistant(assistant_id, name, instructions, model)
        return f"Successfully updated assistant '{result.name}' (ID: {result.id})."
    except Exception as e:
        raise ToolError(f"Failed to update assistant {assistant_id}: {e}") 