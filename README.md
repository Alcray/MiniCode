# MiniCode

Minimal, hackable **agentic coding runtime**. A foundation layer for building AI-powered coding tools.

## Philosophy

- **Minimal**: ~1500 lines of Python, few dependencies
- **Hackable**: Flat structure, no framework gravity
- **Debuggable**: Structured logging, session replay
- **Transparent**: See every tool call, every LLM response

## Quick Start

```bash
# Install
pip install -e .

# Set API key
export OPENAI_API_KEY=sk-...

# Run
minicode "add a hello world endpoint to app.py"
```

## Configuration

MiniCode uses OpenAI-compatible endpoints. Configure via environment variables:

```bash
# Required
export OPENAI_API_KEY=sk-...        # or MINICODE_API_KEY

# Optional
export MINICODE_MODEL=gpt-4o        # Model name
export MINICODE_BASE_URL=https://api.openai.com/v1  # API endpoint
export MINICODE_TEMPERATURE=0.0     # Temperature (0-2)
export MINICODE_MAX_TOKENS=4096     # Max response tokens
```

### Using Alternative Providers

Any OpenAI-compatible endpoint works:

```bash
# Claude via Anthropic's OpenAI-compatible endpoint
export MINICODE_BASE_URL=https://api.anthropic.com/v1
export MINICODE_API_KEY=sk-ant-...
export MINICODE_MODEL=claude-3-5-sonnet-20241022

# Local models via Ollama
export MINICODE_BASE_URL=http://localhost:11434/v1
export MINICODE_MODEL=llama3.1

# Cerebras (if OpenAI-compatible)
export MINICODE_BASE_URL=https://api.cerebras.ai/v1
export MINICODE_API_KEY=...
```

## CLI Usage

```bash
# Basic usage
minicode "your request"

# Specify workspace
minicode --root /path/to/project "your request"

# Override model
minicode --model gpt-4o-mini "your request"

# Plan mode (read-only exploration)
minicode --plan "how would you add caching?"

# Plain text output (for CI/scripts)
minicode --no-tui "your request"

# View sessions
minicode --list-sessions
minicode --replay SESSION_ID
minicode --show SESSION_ID
```

## Tools

MiniCode provides exactly 7 tools:

| Tool | Description |
|------|-------------|
| `list(path)` | List directory entries |
| `glob(pattern)` | Find files matching pattern |
| `grep(query, path, glob_filter)` | Search for text in files |
| `read(path, start_line, end_line)` | Read file content |
| `write(path, content)` | Create/overwrite file |
| `edit(path, old_string, new_string)` | Search/replace in file |
| `bash(command, timeout_ms)` | Run shell command |

All tools respect `.gitignore` and built-in ignore patterns.

## Permissions

MiniCode gates tool execution with a permission system.

### Defaults

- **Read tools** (`list`, `glob`, `grep`, `read`): `allow`
- **Write tools** (`write`, `patch`, `bash`): `ask`

### File Patterns

- `.env*` files are **denied** by default (except `.env.example`)
- Dangerous bash patterns are **denied**:
  - `rm -rf /`, `sudo`, `curl | sh`, etc.

### Permission Prompt

When a tool requires permission, you'll see:
```
Permission Required
Tool: bash
Args: {"command": "npm install"}

Allow? [y]es / [n]o / [a]lways
```

- `y`: Allow this once
- `a`: Allow for this session
- `n`: Deny

### Config File

Create `.minicode.json` in your project root:

```json
{
  "tool_defaults": {
    "bash": "allow"
  },
  "deny_read_patterns": [".env", ".env.*", "!.env.example"],
  "deny_write_patterns": []
}
```

## Safety Features

### Doom Loop Detection

If the agent calls the same tool with identical arguments 3 times consecutively, MiniCode stops and asks for confirmation. This prevents infinite loops.

### Max Steps

Default: 30 steps. Override with `--max-steps`.

### Max Tool Calls Per Step

Default: 8 tool calls per LLM response.

## Logging & History

Every run creates a session in `.minicode/sessions/<session_id>/`:

```
.minicode/sessions/20240115_143052_1234/
├── events.jsonl    # Append-only event log
├── messages.json   # Final conversation state
└── summary.json    # Run summary
```

### Event Log Format

```json
{"timestamp": "2024-01-15T14:30:52Z", "session_id": "...", "step": 1, "event_type": "tool_call", "payload": {...}}
{"timestamp": "2024-01-15T14:30:53Z", "session_id": "...", "step": 1, "event_type": "tool_result", "payload": {...}}
```

Event types:
- `step_start`, `step_end`
- `llm_request`, `llm_response`
- `tool_call`, `tool_result`
- `permission_decision`
- `doom_loop`, `error`

### Replay

```bash
# List sessions
minicode --list-sessions

# Replay a session
minicode --replay 20240115_143052_1234
```

## REST API (FastAPI)

MiniCode can be run as an HTTP service using [FastAPI](https://fastapi.tiangolo.com/).

### Install

```bash
pip install -e ".[server]"
```

### Start the server

```bash
# Default: http://127.0.0.1:8000
minicode --serve

# Custom host / port
minicode --serve --host 0.0.0.0 --port 9000
```

### Endpoints

#### `POST /run`

Run the agent with a task.

**Request body**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `request` | string | *required* | The coding task to perform |
| `root` | string | `"."` | Workspace root directory |
| `max_steps` | int | `30` | Maximum agent steps (1–200) |
| `plan_mode` | bool | `false` | Explore only, no file modifications |
| `model` | string | env default | LLM model override |
| `base_url` | string | env default | LLM API base URL override |

**Example**

```bash
curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{"request": "add a hello world function to app.py", "root": "/path/to/project"}'
```

**Response**

```json
{
  "success": true,
  "final_message": "Done. Added hello_world() to app.py.",
  "step_count": 3,
  "tool_call_count": 4,
  "files_touched": ["app.py"],
  "status": "complete",
  "session_id": "20240115_143052_1234"
}
```

#### `GET /sessions`

List recent sessions for a workspace.

```bash
curl "http://localhost:8000/sessions?root=/path/to/project"
```

#### `GET /sessions/{session_id}`

Get details for a specific session.

```bash
curl "http://localhost:8000/sessions/20240115_143052_1234?root=/path/to/project"
```

### Interactive API docs

FastAPI automatically generates interactive documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`



Run with `--plan` to explore without making changes:

```bash
minicode --plan "how would you refactor the auth module?"
```

In plan mode:
- `write`, `patch`, `bash` are disabled
- Agent can only explore with read tools
- Output is a structured plan

## Project Structure

```
minicode/
├── agent.py       # Core agentic loop
├── llm.py         # OpenAI-compatible client
├── tools.py       # Tool registry + implementations
├── workspace.py   # Filesystem + ignore rules
├── permissions.py # Permission system
├── logging_.py    # Structured logging
├── history.py     # Session management
├── prompts.py     # System prompts
├── tui.py         # Terminal UI (Rich)
└── cli.py         # CLI entry point
```

## Testing

```bash
pip install -e ".[dev]"
pytest
```

Tests use a mock LLM client - no API key required.

## Extending MiniCode

MiniCode is designed to be a foundation. To build on it:

1. **Custom tools**: Add to `tools.py`, update `TOOL_SCHEMAS`
2. **Custom prompts**: Modify `prompts.py`
3. **Custom UI**: Implement `UICallback` protocol
4. **Custom permissions**: Extend `PermissionManager`

## Non-Goals

MiniCode intentionally does NOT include:

- LangChain/LlamaIndex integration
- Embeddings or vector databases
- Multi-model routing
- Plugin ecosystem
- Web browsing
- Telemetry
- Background services

This is a foundation layer, not a product.

## License

MIT
