"""Main entry point for the MCP OpenAI Assistant server."""
import os
import sys
from pathlib import Path
from . import app
from .assistant_manager import AssistantManager

DEFAULT_DB_PATH = str(Path.home() / ".mcp_simple_openai_assistant" / "threads.db")

def main():
    """Initialize manager and run the MCP server."""
    # The API key must be set in the environment by the client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print(
            "Error: OPENAI_API_KEY environment variable not set.",
            file=sys.stderr
        )
        sys.exit(1)

    # Use DB_PATH from environment or a default value
    db_path = os.getenv("DB_PATH", DEFAULT_DB_PATH)

    # Initialize the manager with the explicit API key and db_path
    manager = AssistantManager(api_key=api_key, db_path=db_path)

    # Assign the initialized manager to the app module
    app.manager = manager

    # The app is imported from app.py where it is already defined
    app.app.run()

if __name__ == "__main__":
    main()