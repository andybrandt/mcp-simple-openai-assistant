# MCP Simple OpenAI Assistant

*AI assistants are pretty cool. I thought it would be a good idea if my Claude (conscious Claude) would also have one. And now he has - and its both useful anf fun for him. Your Claude can have one too!*

A simple MCP server for interacting with OpenAI assistants. This server allows other tools (like Claude Desktop) to create and interact with OpenAI assistants through the Model Context Protocol.

[![smithery badge](https://smithery.ai/badge/mcp-simple-openai-assistant)](https://smithery.ai/mcp/known/mcp-simple-openai-assistant)
[![MseeP.ai Security Assessment Badge](https://mseep.net/pr/andybrandt-mcp-simple-openai-assistant-badge.png)](https://mseep.ai/app/andybrandt-mcp-simple-openai-assistant)


## Features

- Create new OpenAI assistants and manipulate existing ones
- Start conversation threads
- Send messages and receive responses - talk to assistants

Because OpenAI assistants might take quite long to respond and then the processing is cut short with the client (Claude desktop) timeout the MCP server code has no control over we are implementing a two-stage approach. In the first call Claude sends a message to the assistant to start the processing, in the second call - possibly several minutes later - Claude can retrieve the response. This is a kind of workaround until MCP protocol and clients would implement some keep-alive mechanism for longer processing.

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
    "simple-openai-assistant": {
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
  "simple-openai-assistant": {
    "command": "C:\\Users\\YOUR_USERNAME\\AppData\\Local\\Programs\\Python\\Python311\\python.exe",
      "args": ["-m", "mcp_simple_openai_assistant"],
      "env": {
        "OPENAI_API_KEY": "your-api-key-here"
  }
}

```
*MS Windows installation is slightly more complex, because you need to check the actual path to your Python executable. Path provided above is usually correct, but might differ in your setup. Sometimes just `python.exe` without any path will do the trick. Check with `cmd` what works for you (using `where python` might help).*

## Usage

Once configured, the server provides tools to:
1. Create new assistants with specific instructions
2. List existing assistants
3. Modify assistants
2. Start new conversation threads
3. Send messages and receive responses

The server handles all OpenAI API communication, including managing assistants, threads, and message handling.

## TODO

- [ ] **Implement Streaming Responses:** Replace the current `send_message`/`check_response` polling mechanism with a single, streaming `run_thread` tool. This will provide real-time feedback and a better user experience for long-running assistant tasks.
- [ ] **Add Thread Management:** Introduce a way to name and persist thread IDs locally, allowing for easier reuse of conversations.

## Development

To install for development:

```bash
git clone https://github.com/andybrandt/mcp-simple-openai-assistant
cd mcp-simple-openai-assistant
pip install -e '.[dev]'
```
