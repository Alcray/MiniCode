"""LLM Client: OpenAI-compatible API client."""

import json
import os
from dataclasses import dataclass
from copy import deepcopy
from typing import Any

import httpx


DEFAULT_PROVIDER = "openai"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o"

GOOGLE_API_PROVIDER_ALIASES = {
    "google",
    "google_ai",
    "google-ai",
    "google_ai_studio",
    "google-ai-studio",
    "gemini",
    "gemini_api",
    "gemini-api",
}
DEFAULT_GOOGLE_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
DEFAULT_GOOGLE_API_MODEL = "gemini-2.5-pro"

GOOGLE_CLOUD_PROVIDER_ALIASES = {
    "google_cloud",
    "google-cloud",
    "gcp",
    "vertex_ai",
    "vertex-ai",
}
DEFAULT_GOOGLE_CLOUD_LOCATION = "us-central1"
DEFAULT_GOOGLE_CLOUD_API_VERSION = "v1beta1"
DEFAULT_GOOGLE_CLOUD_MODEL = "google/gemini-2.5-pro"
DEFAULT_GOOGLE_GENAI_MODEL = "gemini-2.5-pro"
OPENAI_TRANSPORT = "openai"
GOOGLE_GENAI_TRANSPORT = "google-genai"


def _env_first(*names: str) -> str:
    for name in names:
        value = os.environ.get(name, "")
        if value:
            return value
    return ""


def _provider_key(provider: str) -> str:
    return provider.strip().lower()


@dataclass
class LLMConfig:
    """Configuration for the LLM client."""
    provider: str = DEFAULT_PROVIDER
    base_url: str = DEFAULT_BASE_URL
    api_key: str = ""
    model: str = DEFAULT_MODEL
    temperature: float = 0.0
    max_tokens: int = 16384
    transport: str = OPENAI_TRANSPORT

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Load config from environment variables."""
        base_url = os.environ.get("MINICODE_BASE_URL")
        api_key = _env_first("MINICODE_API_KEY", "OPENAI_API_KEY")
        model = os.environ.get("MINICODE_MODEL")

        config = cls(
            provider=os.environ.get("MINICODE_PROVIDER", DEFAULT_PROVIDER),
            base_url=base_url or DEFAULT_BASE_URL,
            api_key=api_key,
            model=model or DEFAULT_MODEL,
            temperature=float(os.environ.get("MINICODE_TEMPERATURE", "0.0")),
            max_tokens=int(os.environ.get("MINICODE_MAX_TOKENS", "16384")),
        )
        config.apply_provider_defaults(
            base_url_explicit=base_url is not None,
            api_key_explicit=bool(os.environ.get("MINICODE_API_KEY")),
            model_explicit=model is not None,
        )
        return config

    @staticmethod
    def google_cloud_base_url(
        project_id: str,
        location: str = DEFAULT_GOOGLE_CLOUD_LOCATION,
        api_version: str = DEFAULT_GOOGLE_CLOUD_API_VERSION,
    ) -> str:
        """Build the Google Cloud OpenAI-compatible Vertex AI base URL."""
        return (
            f"https://{location}-aiplatform.googleapis.com/{api_version}/"
            f"projects/{project_id}/locations/{location}/endpoints/openapi"
        )

    def apply_provider_defaults(
        self,
        *,
        base_url_explicit: bool = False,
        api_key_explicit: bool = False,
        model_explicit: bool = False,
    ) -> None:
        """Apply provider-specific OpenAI-compatible defaults."""
        provider = _provider_key(self.provider)

        # Always reset transport first so a previously set GOOGLE_GENAI_TRANSPORT
        # does not bleed into unrelated providers.
        self.transport = OPENAI_TRANSPORT

        if provider in GOOGLE_API_PROVIDER_ALIASES:
            if not base_url_explicit:
                self.base_url = DEFAULT_GOOGLE_API_BASE_URL

            if not api_key_explicit:
                self.api_key = _env_first(
                    "MINICODE_GOOGLE_API_KEY",
                    "GEMINI_API_KEY",
                    "GOOGLE_API_KEY",
                    "GOOGLE_AI_API_KEY",
                ) or self.api_key

            if not model_explicit:
                self.model = DEFAULT_GOOGLE_API_MODEL
            return

        if provider not in GOOGLE_CLOUD_PROVIDER_ALIASES:
            return

        # Resolve project_id up-front so we can detect a misconfigured state later.
        project_id = (
            _env_first(
                "MINICODE_GOOGLE_CLOUD_PROJECT",
                "GOOGLE_CLOUD_PROJECT",
                "GOOGLE_CLOUD_PROJECT_ID",
            )
            if not base_url_explicit
            else ""
        )

        if not base_url_explicit and project_id:
            location = _env_first(
                "MINICODE_GOOGLE_CLOUD_LOCATION",
                "GOOGLE_CLOUD_LOCATION",
                "GOOGLE_CLOUD_REGION",
            ) or DEFAULT_GOOGLE_CLOUD_LOCATION
            api_version = (
                os.environ.get("MINICODE_GOOGLE_CLOUD_API_VERSION")
                or DEFAULT_GOOGLE_CLOUD_API_VERSION
            )
            self.base_url = self.google_cloud_base_url(project_id, location, api_version)

        if not api_key_explicit:
            google_cloud_api_key = _env_first(
                "MINICODE_GOOGLE_CLOUD_API_KEY",
                "GOOGLE_CLOUD_API_KEY",
            )
            if google_cloud_api_key:
                self.api_key = google_cloud_api_key
                self.transport = GOOGLE_GENAI_TRANSPORT
            else:
                self.api_key = _env_first(
                    "MINICODE_GOOGLE_CLOUD_ACCESS_TOKEN",
                    "GOOGLE_CLOUD_ACCESS_TOKEN",
                    "GOOGLE_OAUTH_ACCESS_TOKEN",
                ) or self.api_key

        if self.transport == GOOGLE_GENAI_TRANSPORT:
            if not model_explicit:
                self.model = DEFAULT_GOOGLE_GENAI_MODEL
            elif self.model.startswith("google/"):
                self.model = self.model.removeprefix("google/")
            return

        # OpenAI-compatible Vertex AI path: a valid Vertex base_url is required.
        # Fail fast rather than silently routing to the wrong endpoint.
        if not base_url_explicit and not project_id:
            raise ValueError(
                f"MINICODE_PROVIDER={self.provider!r} requires either an explicit "
                "MINICODE_BASE_URL or a Google Cloud project ID "
                "(MINICODE_GOOGLE_CLOUD_PROJECT / GOOGLE_CLOUD_PROJECT). "
                "To use the Google GenAI SDK transport instead, set GOOGLE_CLOUD_API_KEY."
            )

        if not api_key_explicit:
            self.api_key = _env_first(
                "MINICODE_GOOGLE_CLOUD_ACCESS_TOKEN",
                "GOOGLE_CLOUD_ACCESS_TOKEN",
                "GOOGLE_OAUTH_ACCESS_TOKEN",
            ) or self.api_key

        if not model_explicit:
            self.model = DEFAULT_GOOGLE_CLOUD_MODEL
        elif "/" not in self.model:
            self.model = f"google/{self.model}"


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
        if self.config.transport == GOOGLE_GENAI_TRANSPORT:
            return self._complete_google_genai(messages, tools)

        url = f"{self.config.base_url.rstrip('/')}/chat/completions"

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
        }

        # max_tokens=0 means no limit (let the model use its native max)
        if self.config.max_tokens > 0:
            payload["max_tokens"] = self.config.max_tokens

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

    def _complete_google_genai(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        """Send a completion request through Google GenAI Vertex API-key mode."""
        try:
            from google import genai
            from google.genai import types
        except ImportError as e:
            raise ImportError(
                "google-genai is required for MINICODE_PROVIDER=google-cloud with "
                "GOOGLE_CLOUD_API_KEY. Install with: pip install --upgrade google-genai"
            ) from e

        client = genai.Client(vertexai=True, api_key=self.config.api_key)
        contents, system_instruction = self._to_google_contents(messages, types)
        config_kwargs: dict[str, Any] = {
            "temperature": self.config.temperature,
        }
        if self.config.max_tokens > 0:
            config_kwargs["max_output_tokens"] = self.config.max_tokens
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        google_tools = self._to_google_tools(tools, types)
        if google_tools:
            config_kwargs["tools"] = google_tools

        response = client.models.generate_content(
            model=self.config.model,
            contents=contents,
            config=types.GenerateContentConfig(**config_kwargs),
        )
        return self._parse_google_genai_response(response)

    @staticmethod
    def _google_schema(schema: dict) -> dict:
        """Convert JSON Schema type names to Google GenAI's uppercase form."""
        type_map = {
            "object": "OBJECT",
            "array": "ARRAY",
            "string": "STRING",
            "integer": "INTEGER",
            "number": "NUMBER",
            "boolean": "BOOLEAN",
        }

        def convert(value: Any) -> Any:
            if isinstance(value, list):
                return [convert(item) for item in value]
            if not isinstance(value, dict):
                return value

            converted = {}
            for key, item in value.items():
                if key == "type" and isinstance(item, str):
                    converted[key] = type_map.get(item.lower(), item)
                else:
                    converted[key] = convert(item)
            return converted

        return convert(deepcopy(schema))

    @classmethod
    def _to_google_tools(cls, tools: list[dict] | None, types: Any) -> list[Any]:
        if not tools:
            return []

        declarations = []
        for tool in tools:
            function = tool.get("function", {})
            declarations.append(types.FunctionDeclaration(
                name=function.get("name", ""),
                description=function.get("description", ""),
                parameters=cls._google_schema(function.get("parameters", {})),
            ))

        return [types.Tool(function_declarations=declarations)]

    @staticmethod
    def _to_google_contents(messages: list[dict], types: Any) -> tuple[list[Any], str | None]:
        contents = []
        system_parts = []
        tool_call_names: dict[str, str] = {}

        for message in messages:
            role = message.get("role")
            content = message.get("content")

            if role == "system":
                if content:
                    system_parts.append(str(content))
                continue

            if role == "tool":
                tool_call_id = message.get("tool_call_id", "")
                name = tool_call_names.get(tool_call_id, "tool_result")
                try:
                    response = json.loads(content or "{}")
                except json.JSONDecodeError:
                    response = {"result": content or ""}
                contents.append(types.Content(
                    role="tool",
                    parts=[types.Part.from_function_response(name=name, response=response)],
                ))
                continue

            parts = []
            if content:
                parts.append(types.Part.from_text(text=str(content)))

            for tool_call in message.get("tool_calls", []) or []:
                function = tool_call.get("function", {})
                name = function.get("name", "")
                args_raw = function.get("arguments", "{}")
                try:
                    args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                except json.JSONDecodeError:
                    args = {}
                tool_call_id = tool_call.get("id")
                if tool_call_id:
                    tool_call_names[tool_call_id] = name
                parts.append(types.Part(function_call=types.FunctionCall(
                    id=tool_call_id,
                    name=name,
                    args=args,
                )))

            if parts:
                contents.append(types.Content(
                    role="model" if role == "assistant" else "user",
                    parts=parts,
                ))

        system_instruction = "\n\n".join(system_parts) if system_parts else None
        return contents, system_instruction

    @staticmethod
    def _parse_google_genai_response(response: Any) -> LLMResponse:
        content = getattr(response, "text", None)
        function_calls = getattr(response, "function_calls", None) or []
        tool_calls = []
        for i, function_call in enumerate(function_calls):
            name = getattr(function_call, "name", "")
            args = getattr(function_call, "args", {}) or {}
            tool_call_id = getattr(function_call, "id", None) or f"call_google_{i}"
            tool_calls.append(ToolCall(
                id=tool_call_id,
                name=name,
                arguments=dict(args),
            ))

        usage_metadata = getattr(response, "usage_metadata", None)
        prompt_tokens = getattr(usage_metadata, "prompt_token_count", 0) if usage_metadata else 0
        completion_tokens = (
            getattr(usage_metadata, "candidates_token_count", 0) if usage_metadata else 0
        )
        total_tokens = getattr(usage_metadata, "total_token_count", 0) if usage_metadata else 0
        reasoning_tokens = getattr(usage_metadata, "thoughts_token_count", 0) if usage_metadata else 0

        if hasattr(response, "model_dump"):
            raw_response = response.model_dump(exclude_none=True)
        else:
            raw_response = {"text": content, "function_calls": [tc.__dict__ for tc in function_calls]}

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason="tool_calls" if tool_calls else "stop",
            usage={
                "prompt_tokens": prompt_tokens or 0,
                "completion_tokens": completion_tokens or 0,
                "total_tokens": total_tokens or 0,
                "reasoning_tokens": reasoning_tokens or 0,
            },
            raw_response=raw_response,
        )

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
