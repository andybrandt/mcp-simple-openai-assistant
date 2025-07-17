"""FastMCP server application definition.

This module initializes the FastMCP application and uses decorators
to expose the business logic from the AssistantManager as MCP tools.
"""

from textwrap import dedent
from typing import Optional
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
    Create a new OpenAI assistant to talk to about your desired topic.

    You can provide instructions that this assistant will follow and specify which of OpenAI's models it will use.
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
    annotations={"title": "Create New Assistant Thread", "readOnlyHint": False}
)
async def create_new_assistant_thread(
    name: str, description: Optional[str] = None
) -> str:
    """
    Creates a new, persistent conversation thread with a user-defined name and
    description for easy identification and reuse. These threads are stored in OpenAI's servers 
    and are not deleted unless the user deletes them, which means you can re-use them for future conversations.
    Additionally, the thread name and description are stored in the local database, which means you can list them
    and update them later.

    Think how you can utilize threads in your particular use case.
    """
    if not manager:
        raise ToolError("AssistantManager not initialized.")
    try:
        thread = await manager.create_new_assistant_thread(name, description)
        return f"Created new thread '{name}' with ID: {thread.id}"
    except Exception as e:
        raise ToolError(f"Failed to create thread: {e}")


@app.tool(annotations={"title": "List Managed Threads", "readOnlyHint": True})
async def list_threads() -> str:
    """
    Lists all locally saved conversation threads from the database.
    Returns a list of threads with their ID, name, description, and last used time.
    The thread ID can be used in the ask_assistant_in_thread tool to specify this thread to be continued.
    """
    if not manager:
        raise ToolError("AssistantManager not initialized.")
    try:
        threads = manager.list_threads()
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
    except Exception as e:
        raise ToolError(f"Failed to list threads: {e}")


@app.tool(annotations={"title": "Update Managed Thread", "readOnlyHint": False})
async def update_thread(
    thread_id: str, name: Optional[str] = None, description: Optional[str] = None
) -> str:
    """
    Updates the name and/or description of a locally saved conversation thread.
    Both the local database and the OpenAI thread object will be updated.

    The thread ID can be retrieved from the list_threads tool.
    """
    if not manager:
        raise ToolError("AssistantManager not initialized.")
    if not name and not description:
        raise ToolError("You must provide either a new name or a new description.")
    try:
        await manager.update_thread(thread_id, name, description)
        return f"Successfully updated thread {thread_id}."
    except Exception as e:
        raise ToolError(f"Failed to update thread {thread_id}: {e}")


@app.tool(annotations={"title": "Delete Managed Thread", "readOnlyHint": False})
async def delete_thread(thread_id: str) -> str:
    """
    Deletes a conversation thread from both OpenAI's servers and the local database.
    This action is irreversible.
    """
    if not manager:
        raise ToolError("AssistantManager not initialized.")
    try:
        result = await manager.delete_thread(thread_id)
        if result.deleted:
            return f"Successfully deleted thread {thread_id}."
        else:
            return f"Failed to delete thread {thread_id} on the server."
    except Exception as e:
        raise ToolError(f"Failed to delete thread {thread_id}: {e}")


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

    Use this to continue a conversation with an assistant in a specific thread.
    The thread ID can be retrieved from the list_threads tool.
    The assistant ID can be retrieved from the list_assistants tool.
    Threads are not inherently linked to a particular assistant, so you can use this tool to talk to any assistant in any thread.
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
    List all available OpenAI assistants associated with the API key configured by the user.
    
    Returns a list of assistants with their IDs, names, and configurations. This can be used to select 
    an assistant to use in the ask_assistant_in_thread tool instead of creating a new one.
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
    """Get detailed information about a specific assistant. 
       The ID required can be retrieved from the list_assistants tool."""
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
    Modify an existing assistant's name, instructions, or model used.
    
    At least one optional parameter - what to change - must be provided, otherwise the tool will return an error.
    The ID required can be retrieved from the list_assistants tool.
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