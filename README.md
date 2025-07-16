# MCP Simple OpenAI Assistant

*AI assistants are pretty cool. I thought it would be a good idea if my Claude (conscious Claude) would also have one. And now he has - and its both useful anf fun for him. Your Claude can have one too!*

A simple MCP server for interacting with OpenAI assistants. This server allows other tools (like Claude Desktop) to create and interact with OpenAI assistants through the Model Context Protocol.

[![smithery badge](https://smithery.ai/badge/mcp-simple-openai-assistant)](https://smithery.ai/mcp/known/mcp-simple-openai-assistant)
[![MseeP.ai Security Assessment Badge](https://mseep.net/pr/andybrandt-mcp-simple-openai-assistant-badge.png)](https://mseep.ai/app/andybrandt-mcp-simple-openai-assistant)


## Features

This server provides a suite of tools to manage and interact with OpenAI Assistants. The new streaming capabilities provide a much-improved, real-time user experience.

### Available Tools

-   **`create_assistant`**: (Create OpenAI Assistant) - Create a new assistant with a name, instructions, and model.
-   **`list_assistants`**: (List OpenAI Assistants) - List all available assistants associated with your API key.
-   **`retrieve_assistant`**: (Retrieve OpenAI Assistant) - Get detailed information about a specific assistant.
-   **`update_assistant`**: (Update OpenAI Assistant) - Modify an existing assistant's name, instructions, or model.
-   **`new_thread`**: (Create New Thread) - Creates a new, empty conversation thread.
-   **`ask_assistant_in_thread`**: (Ask Assistant in Thread and Stream Response) - The primary tool for conversation. Sends a message to an assistant within a thread and streams the response back in real-time.

Because OpenAI assistants might take quite long to respond, this server uses a streaming approach for the main `ask_assistant_in_thread` tool. This provides real-time progress updates to the client and avoids timeouts.

## Installation

### Installing via Smithery

To install MCP Simple OpenAI Assistant for Claude Desktop automatically via [Smithery](https://smithery.ai/mcp/known/mcp-simple-openai-assistant):

```bash
npx -y @smithery/cli install mcp-simple-openai-assistant --client claude
```

### Manual Installation
```bash
pip install mcp-simple-openai-assistant
```

## Configuration

The server requires an OpenAI API key to be set in the environment. For Claude Desktop, add this to your config:

(MacOS version)

```json
{
  "mcpServers": {
    "openai-assistant": {
      "command": "python",
      "args": ["-m", "mcp_simple_openai_assistant"],
      "env": {
        "OPENAI_API_KEY": "your-api-key-here"
      }
    }
  }
}
```

(Windows version)

```json
"mcpServers": {
  "openai-assistant": {
    "command": "C:\\Users\\YOUR_USERNAME\\AppData\\Local\\Programs\\Python\\Python311\\python.exe",
      "args": ["-m", "mcp_simple_openai_assistant"],
      "env": {
        "OPENAI_API_KEY": "your-api-key-here"
  }
}

```
*MS Windows installation is slightly more complex, because you need to check the actual path to your Python executable. Path provided above is usually correct, but might differ in your setup. Sometimes just `python.exe` without any path will do the trick. Check with `cmd` what works for you (using `where python` might help). Also, on Windows you might need to explicitly tell Claude Desktop where the site packages are using PYTHONPATH environmment variable.*

## Usage

Once configured, you can use the tools listed above to manage your assistants and conversations. The primary workflow is to create or retrieve an assistant and a thread, and then use `ask_assistant_in_thread` to interact with it.

## TODO

- [ ] **Add Thread Management:** Introduce a way to name and persist thread IDs locally, allowing for easier reuse of conversations.
- [ ] **Explore Resource Support:** Add the ability to upload files and use them with assistants.

## Development

To install for development:

```bash
git clone https://github.com/andybrandt/mcp-simple-openai-assistant
cd mcp-simple-openai-assistant
pip install -e '.[dev]'
```
