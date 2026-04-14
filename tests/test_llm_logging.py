"""Tests for LLM response parsing and full response logging pipeline."""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from minicode.llm import LLMClient, LLMConfig, LLMResponse, ToolCall


# ===========================================================================
# _parse_response: test against real API response shapes
# ===========================================================================

class TestParseResponse:
    """Test _parse_response against actual API JSON from different providers."""

    @pytest.fixture
    def client(self):
        return LLMClient(LLMConfig(api_key="test"))

    def test_standard_text_response(self, client):
        """Normal text response -- GPT-4o style."""
        data = {
            "choices": [{
                "message": {"role": "assistant", "content": "Hello world"},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }
        r = client._parse_response(data)
        assert r.content == "Hello world"
        assert r.has_tool_calls is False
        assert r.finish_reason == "stop"
        assert r.usage["completion_tokens"] == 5
        assert r.reasoning_content is None
        assert r.raw_response is data

    def test_tool_call_response(self, client):
        """Response with tool calls -- GPT-4o style."""
        data = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_abc123",
                        "type": "function",
                        "function": {
                            "name": "read",
                            "arguments": '{"path": "main.py"}',
                        },
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
            },
        }
        r = client._parse_response(data)
        assert r.content is None
        assert r.has_tool_calls is True
        assert len(r.tool_calls) == 1
        assert r.tool_calls[0].name == "read"
        assert r.tool_calls[0].arguments == {"path": "main.py"}
        assert r.tool_calls[0].id == "call_abc123"
        assert r.finish_reason == "tool_calls"

    def test_deepseek_reasoning_response(self, client):
        """DeepSeek-style response with reasoning_content field."""
        data = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "reasoning_content": "Let me think about this step by step...\n1. First I need to read the file\n2. Then find the bug",
                    "content": "I'll start by reading the file.",
                    "tool_calls": [{
                        "id": "call_ds_001",
                        "type": "function",
                        "function": {
                            "name": "read",
                            "arguments": '{"path": "app.py"}',
                        },
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {
                "prompt_tokens": 500,
                "completion_tokens": 150,
                "total_tokens": 650,
                "completion_tokens_details": {
                    "reasoning_tokens": 100,
                },
            },
        }
        r = client._parse_response(data)
        assert r.content == "I'll start by reading the file."
        assert r.reasoning_content == "Let me think about this step by step...\n1. First I need to read the file\n2. Then find the bug"
        assert r.has_tool_calls is True
        assert r.usage["reasoning_tokens"] == 100

    def test_thinking_only_empty_content(self, client):
        """Model burns tokens thinking but produces empty content -- the bug scenario."""
        data = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "reasoning_content": "I need to analyze the HTML file and apply the suggestion about using hidden attribute...",
                    "content": None,
                },
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 2000,
                "completion_tokens": 690,
                "total_tokens": 2690,
                "completion_tokens_details": {
                    "reasoning_tokens": 690,
                },
            },
        }
        r = client._parse_response(data)
        assert r.content is None
        assert r.has_tool_calls is False
        assert r.reasoning_content is not None
        assert "hidden attribute" in r.reasoning_content
        assert r.usage["reasoning_tokens"] == 690
        assert r.usage["completion_tokens"] == 690

    def test_openai_o3_hidden_reasoning(self, client):
        """OpenAI o3 style -- reasoning tokens counted but text not exposed."""
        data = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "The answer is 42.",
                },
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 50,
                "completion_tokens": 200,
                "total_tokens": 250,
                "completion_tokens_details": {
                    "reasoning_tokens": 180,
                    "accepted_prediction_tokens": 0,
                    "rejected_prediction_tokens": 0,
                },
            },
        }
        r = client._parse_response(data)
        assert r.content == "The answer is 42."
        assert r.reasoning_content is None
        assert r.usage["reasoning_tokens"] == 180

    def test_malformed_tool_call_arguments(self, client):
        """Tool call with invalid JSON arguments -- should not crash."""
        data = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_bad",
                        "type": "function",
                        "function": {
                            "name": "edit",
                            "arguments": "not valid json {{{",
                        },
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        r = client._parse_response(data)
        assert r.has_tool_calls is True
        assert r.tool_calls[0].arguments == {}

    def test_no_completion_tokens_details(self, client):
        """Provider that doesn't include completion_tokens_details."""
        data = {
            "choices": [{
                "message": {"role": "assistant", "content": "hi"},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 2,
                "total_tokens": 7,
            },
        }
        r = client._parse_response(data)
        assert r.usage["reasoning_tokens"] == 0

    def test_raw_response_preserved(self, client):
        """The full raw API response dict is stored for debugging."""
        data = {
            "id": "chatcmpl-xyz",
            "object": "chat.completion",
            "model": "gpt-4o",
            "choices": [{
                "message": {"role": "assistant", "content": "test"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        r = client._parse_response(data)
        assert r.raw_response is data
        assert r.raw_response["id"] == "chatcmpl-xyz"
        assert r.raw_response["model"] == "gpt-4o"


# ===========================================================================
# Agent logging pipeline: messages.json + raw_responses.jsonl
# ===========================================================================

class TestAgentLogging:
    """Test that assistant messages and raw responses are properly saved."""

    @pytest.fixture
    def workspace(self, tmp_path):
        from minicode.workspace import Workspace
        return Workspace(tmp_path)

    def _make_response(self, content=None, tool_calls=None, reasoning=None, raw=None):
        return LLMResponse(
            content=content,
            tool_calls=tool_calls or [],
            finish_reason="stop" if not tool_calls else "tool_calls",
            usage={"prompt_tokens": 10, "completion_tokens": 50, "total_tokens": 60, "reasoning_tokens": 0},
            reasoning_content=reasoning,
            raw_response=raw or {"choices": [{"message": {"content": content}}]},
        )

    def test_text_only_response_saved_in_messages(self, workspace, tmp_path):
        """Text-only responses (no tool calls) must appear in messages.json."""
        from minicode.agent import Agent, AgentConfig
        from minicode.llm import MockLLMClient

        mock = MockLLMClient()
        mock.add_response(self._make_response(content="Here is my analysis."))

        agent = Agent(
            workspace_root=tmp_path,
            llm_client=mock,
            config=AgentConfig(max_steps=1),
        )
        result = agent.run("analyze this")

        # Load messages.json
        messages_file = agent.logger.session_dir / "messages.json"
        messages = json.loads(messages_file.read_text())

        # Should have 3 messages: system, user, assistant
        assert len(messages) == 3
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[2]["role"] == "assistant"
        assert messages[2]["content"] == "Here is my analysis."

    def test_empty_response_still_saved(self, workspace, tmp_path):
        """Even empty content responses should be in messages.json."""
        from minicode.agent import Agent, AgentConfig
        from minicode.llm import MockLLMClient

        mock = MockLLMClient()
        mock.add_response(self._make_response(content=None))

        agent = Agent(
            workspace_root=tmp_path,
            llm_client=mock,
            config=AgentConfig(max_steps=1),
        )
        result = agent.run("do something")

        messages_file = agent.logger.session_dir / "messages.json"
        messages = json.loads(messages_file.read_text())

        assert len(messages) == 3
        assert messages[2]["role"] == "assistant"

    def test_tool_call_response_saved(self, workspace, tmp_path):
        """Tool call responses should include both assistant and tool messages."""
        from minicode.agent import Agent, AgentConfig
        from minicode.llm import MockLLMClient

        mock = MockLLMClient()
        mock.add_response(self._make_response(
            content=None,
            tool_calls=[ToolCall(id="call_1", name="list", arguments={"path": "."})],
        ))
        # Second call returns text (completes)
        mock.add_response(self._make_response(content="Done listing."))

        agent = Agent(
            workspace_root=tmp_path,
            llm_client=mock,
            config=AgentConfig(max_steps=5),
        )
        result = agent.run("list files")

        messages_file = agent.logger.session_dir / "messages.json"
        messages = json.loads(messages_file.read_text())

        roles = [m["role"] for m in messages]
        assert roles == ["system", "user", "assistant", "tool", "assistant"]
        assert messages[4]["content"] == "Done listing."

    def test_reasoning_not_in_messages(self, workspace, tmp_path):
        """Reasoning content must NOT be in the conversation messages."""
        from minicode.agent import Agent, AgentConfig
        from minicode.llm import MockLLMClient

        mock = MockLLMClient()
        resp = self._make_response(
            content="Final answer.",
            reasoning="Long chain of thought about the problem...",
        )
        mock.add_response(resp)

        agent = Agent(
            workspace_root=tmp_path,
            llm_client=mock,
            config=AgentConfig(max_steps=1),
        )
        agent.run("question")

        messages_file = agent.logger.session_dir / "messages.json"
        messages = json.loads(messages_file.read_text())

        # Reasoning should not appear anywhere in messages
        messages_text = json.dumps(messages)
        assert "chain of thought" not in messages_text
        # But content should be there
        assert "Final answer." in messages_text

    def test_raw_responses_saved(self, workspace, tmp_path):
        """Raw API responses should be saved to raw_responses.jsonl."""
        from minicode.agent import Agent, AgentConfig
        from minicode.llm import MockLLMClient

        raw_data = {
            "id": "chatcmpl-test123",
            "choices": [{"message": {"content": "hi", "reasoning_content": "thinking..."}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
        }
        mock = MockLLMClient()
        mock.add_response(self._make_response(content="hi", raw=raw_data))

        agent = Agent(
            workspace_root=tmp_path,
            llm_client=mock,
            config=AgentConfig(max_steps=1),
        )
        agent.run("hello")

        raw_file = agent.logger.session_dir / "raw_responses.jsonl"
        assert raw_file.exists()
        line = json.loads(raw_file.read_text().strip())
        assert line["step"] == 1
        assert line["raw"]["id"] == "chatcmpl-test123"
        assert "reasoning_content" in json.dumps(line["raw"])

    def test_events_include_content_preview(self, workspace, tmp_path):
        """The llm_response event should now include content_preview."""
        from minicode.agent import Agent, AgentConfig
        from minicode.llm import MockLLMClient

        mock = MockLLMClient()
        mock.add_response(self._make_response(content="A detailed analysis of the code."))

        agent = Agent(
            workspace_root=tmp_path,
            llm_client=mock,
            config=AgentConfig(max_steps=1),
        )
        agent.run("analyze")

        events_file = agent.logger.session_dir / "events.jsonl"
        events = [json.loads(line) for line in events_file.read_text().strip().split("\n")]
        llm_events = [e for e in events if e["event_type"] == "llm_response"]

        assert len(llm_events) == 1
        payload = llm_events[0]["payload"]
        assert payload["content_preview"] == "A detailed analysis of the code."
        assert payload["content_length"] == len("A detailed analysis of the code.")
