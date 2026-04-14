"""LLM Client: OpenAI-compatible API client."""

import json
import os
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass
class LLMConfig:
    """Configuration for the LLM client."""
    provider: str = "openai"
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "gpt-4o"
    temperature: float = 0.0
    max_tokens: int = 4096

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Load config from environment variables."""
        return cls(
            provider=os.environ.get("MINICODE_PROVIDER", "openai"),
            base_url=os.environ.get("MINICODE_BASE_URL", "https://api.openai.com/v1"),
            api_key=os.environ.get("MINICODE_API_KEY", os.environ.get("OPENAI_API_KEY", "")),
            model=os.environ.get("MINICODE_MODEL", "gpt-4o"),
            temperature=float(os.environ.get("MINICODE_TEMPERATURE", "0.0")),
            max_tokens=int(os.environ.get("MINICODE_MAX_TOKENS", "4096")),
        )


@dataclass
class ToolCall:
    """A tool call from the LLM."""
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    """Response from the LLM."""
    content: str | None
    tool_calls: list[ToolCall]
    finish_reason: str
    usage: dict[str, int]
    reasoning_content: str | None = None
    raw_response: dict | None = None

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


class LLMClient:
    """OpenAI-compatible LLM client."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self._client = httpx.Client(timeout=120.0)

    def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        """Send a completion request."""
        url = f"{self.config.base_url.rstrip('/')}/chat/completions"

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
        }

        response = self._client.post(url, json=payload, headers=headers)
        response.raise_for_status()

        data = response.json()
        return self._parse_response(data)

    def _parse_response(self, data: dict) -> LLMResponse:
        """Parse OpenAI-style response."""
        choice = data["choices"][0]
        message = choice["message"]

        content = message.get("content")
        finish_reason = choice.get("finish_reason", "stop")

        # Capture reasoning/thinking tokens (varies by provider)
        reasoning_content = (
            message.get("reasoning_content")
            or message.get("reasoning")
            or message.get("thinking")
        )

        # Parse tool calls
        tool_calls = []
        if "tool_calls" in message and message["tool_calls"]:
            for tc in message["tool_calls"]:
                try:
                    args = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
                    args = {}

                tool_calls.append(ToolCall(
                    id=tc["id"],
                    name=tc["function"]["name"],
                    arguments=args,
                ))

        usage = data.get("usage", {})

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage={
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
                "reasoning_tokens": usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0),
            },
            reasoning_content=reasoning_content,
            raw_response=data,
        )

    def close(self):
        """Close the HTTP client."""
        self._client.close()


class MockLLMClient:
    """Mock LLM client for testing."""

    def __init__(self, responses: list[LLMResponse] | None = None):
        self.config = LLMConfig(api_key="mock", model="mock-model", base_url="http://mock")
        self.responses = responses or []
        self._call_index = 0
        self.call_history: list[tuple[list[dict], list[dict] | None]] = []

    def add_response(self, response: LLMResponse):
        """Add a response to the queue."""
        self.responses.append(response)

    def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        """Return next mock response."""
        self.call_history.append((messages, tools))

        if self._call_index >= len(self.responses):
            # Default: return empty final response
            return LLMResponse(
                content="Done.",
                tool_calls=[],
                finish_reason="stop",
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            )

        response = self.responses[self._call_index]
        self._call_index += 1
        return response

    def close(self):
        pass
