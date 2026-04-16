"""Integration tests that make real LLM API calls.

Skipped when no API key is configured. Set NVIDIA_API_KEY or OPENAI_API_KEY
to run these tests.

    NVIDIA_API_KEY=nvapi-... pytest tests/test_llm_integration.py -v
"""

import json
import os

import pytest

from minicode.llm import LLMClient, LLMConfig

# ---------------------------------------------------------------------------
# Skip unless an API key is available
# ---------------------------------------------------------------------------

NVIDIA_KEY = os.environ.get("NVIDIA_API_KEY", "")
OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")

if NVIDIA_KEY:
    API_KEY = NVIDIA_KEY
    BASE_URL = "https://integrate.api.nvidia.com/v1"
    MODEL = "openai/gpt-oss-120b"
elif OPENAI_KEY:
    API_KEY = OPENAI_KEY
    BASE_URL = "https://api.openai.com/v1"
    MODEL = "gpt-4o-mini"
else:
    API_KEY = ""
    BASE_URL = ""
    MODEL = ""

requires_api_key = pytest.mark.skipif(
    not API_KEY,
    reason="No API key (set NVIDIA_API_KEY or OPENAI_API_KEY to run)",
)


@pytest.fixture
def client():
    c = LLMClient(LLMConfig(
        api_key=API_KEY,
        base_url=BASE_URL,
        model=MODEL,
        temperature=0.0,
        max_tokens=1024,
    ))
    yield c
    c.close()


# ===========================================================================
# Real API call tests
# ===========================================================================

@requires_api_key
class TestRealLLMCalls:

    def test_simple_text_response(self, client):
        """Model returns text content."""
        response = client.complete([
            {"role": "user", "content": "Reply with exactly the word: PONG"},
        ])
        assert response.content is not None or response.reasoning_content is not None
        assert response.finish_reason in ("stop", "length")
        assert response.usage["completion_tokens"] > 0
        assert response.raw_response is not None
        assert "choices" in response.raw_response

    def test_tool_call_response(self, client):
        """Model produces a tool call when given tools."""
        tools = [{
            "type": "function",
            "function": {
                "name": "read",
                "description": "Read a file from the workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "File path to read",
                        },
                    },
                    "required": ["path"],
                },
            },
        }]
        response = client.complete(
            [{"role": "user", "content": "Read the file called config.py"}],
            tools=tools,
        )
        assert response.has_tool_calls
        assert len(response.tool_calls) >= 1
        tc = response.tool_calls[0]
        assert tc.name == "read"
        assert "path" in tc.arguments
        assert tc.id  # must have an ID
        assert response.finish_reason == "tool_calls"

    def test_reasoning_fields_parsed(self, client):
        """Reasoning content is captured when the model produces it."""
        response = client.complete([
            {"role": "user", "content": "What is 17 * 23? Think step by step."},
        ])
        # The model may or may not produce reasoning depending on provider
        # But the fields should be populated correctly
        if response.reasoning_content:
            assert isinstance(response.reasoning_content, str)
            assert len(response.reasoning_content) > 0

        assert response.usage["completion_tokens"] > 0
        assert response.raw_response is not None

    def test_raw_response_has_full_structure(self, client):
        """raw_response contains the complete API JSON."""
        response = client.complete([
            {"role": "user", "content": "Say hi"},
        ])
        raw = response.raw_response
        assert raw is not None
        assert "id" in raw
        assert "choices" in raw
        assert "usage" in raw
        assert raw["choices"][0]["message"]["role"] == "assistant"

    def test_usage_tokens_present(self, client):
        """Usage stats are populated."""
        response = client.complete([
            {"role": "user", "content": "Count to 3"},
        ])
        assert response.usage["prompt_tokens"] > 0
        assert response.usage["completion_tokens"] > 0
        assert response.usage["total_tokens"] > 0


# ===========================================================================
# Full agent round-trip with real LLM
# ===========================================================================

@requires_api_key
class TestRealAgentRoundTrip:

    def test_agent_read_and_respond(self, tmp_path):
        """Run a real agent session: create a file, ask the agent to read it."""
        from minicode.agent import Agent, AgentConfig

        # Create a test file
        test_file = tmp_path / "hello.txt"
        test_file.write_text("Hello from the test!\n")

        llm = LLMClient(LLMConfig(
            api_key=API_KEY,
            base_url=BASE_URL,
            model=MODEL,
            temperature=0.0,
            max_tokens=2048,
        ))

        agent = Agent(
            workspace_root=tmp_path,
            llm_client=llm,
            config=AgentConfig(max_steps=3),
        )

        result = agent.run("Read the file hello.txt and tell me what it says.")

        # Verify session data was saved
        messages_file = agent.logger.session_dir / "messages.json"
        assert messages_file.exists()
        messages = json.loads(messages_file.read_text())
        # Should have at minimum: system, user, assistant
        assert len(messages) >= 3
        roles = [m["role"] for m in messages]
        assert "assistant" in roles

        # Raw responses should be saved
        raw_file = agent.logger.session_dir / "raw_responses.jsonl"
        assert raw_file.exists()
        raw_lines = raw_file.read_text().strip().split("\n")
        assert len(raw_lines) >= 1
        first_raw = json.loads(raw_lines[0])
        assert "raw" in first_raw
        assert "choices" in first_raw["raw"]

        # Events should include llm_response with content_preview
        events_file = agent.logger.session_dir / "events.jsonl"
        events = [json.loads(l) for l in events_file.read_text().strip().split("\n")]
        llm_events = [e for e in events if e["event_type"] == "llm_response"]
        assert len(llm_events) >= 1
        assert "content_preview" in llm_events[0]["payload"]
        assert "reasoning_length" in llm_events[0]["payload"]

        llm.close()
