"""FastMCP server application definition.

This module initializes the FastMCP application and uses decorators
to expose the business logic from the AssistantManager as MCP tools.
"""

from textwrap import dedent
from fastmcp import FastMCP, Context
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
        "title": "Ask Assistant in Thread and Stream Response",
        "readOnlyHint": False
    }
)
async def ask_assistant_in_thread(thread_id: str, assistant_id: str, message: str, ctx: Context) -> str:
    """
    Sends a message to an assistant within a specific thread and streams the response.
    This provides progress updates and the final message in a single call.
    """
    if not manager:
        raise ToolError("AssistantManager not initialized.")

    final_message = ""
    try:
        await ctx.report_progress(progress=0, message="Starting assistant run...")
        async for event in manager.run_thread(thread_id, assistant_id, message):
            if event.event == 'thread.message.delta':
                text_delta = event.data.delta.content[0].text
                final_message += text_delta.value
                await ctx.report_progress(progress=50, message=f"Assistant writing: {final_message}")
            elif event.event == 'thread.run.step.created':
                await ctx.report_progress(progress=25, message="Assistant is performing a step...")
        
        await ctx.report_progress(progress=100, message="Run complete.")
        return final_message

    except Exception as e:
        raise ToolError(f"An error occurred during the run: {e}")


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