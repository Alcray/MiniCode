"""Tests for the Google GenAI transport."""

import sys
import types as pytypes
from unittest.mock import patch

from minicode.llm import GOOGLE_GENAI_TRANSPORT, LLMClient, LLMConfig


class FakeFunctionDeclaration:
    def __init__(self, name, description, parameters):
        self.name = name
        self.description = description
        self.parameters = parameters


class FakeFunctionCall:
    def __init__(self, name, args, id=None):
        self.id = id
        self.name = name
        self.args = args


class FakePart:
    def __init__(self, text=None, function_call=None, function_response=None):
        self.text = text
        self.function_call = function_call
        self.function_response = function_response

    @classmethod
    def from_text(cls, text):
        return cls(text=text)

    @classmethod
    def from_function_response(cls, name, response):
        return cls(function_response={"name": name, "response": response})


class FakeContent:
    def __init__(self, role, parts):
        self.role = role
        self.parts = parts


class FakeTool:
    def __init__(self, function_declarations):
        self.function_declarations = function_declarations


class FakeGenerateContentConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeUsageMetadata:
    prompt_token_count = 10
    candidates_token_count = 5
    total_token_count = 15
    thoughts_token_count = 2


class FakeResponse:
    text = None
    function_calls = [FakeFunctionCall(id=None, name="read", args={"path": "main.py"})]
    usage_metadata = FakeUsageMetadata()

    def model_dump(self, exclude_none=True):
        return {"text": self.text, "function_calls": [{"name": "read"}]}


class FakeModels:
    def __init__(self, holder):
        self.holder = holder

    def generate_content(self, model, contents, config):
        self.holder["model"] = model
        self.holder["contents"] = contents
        self.holder["config"] = config
        return FakeResponse()


class FakeClient:
    holder = {}

    def __init__(self, vertexai, api_key):
        self.holder["vertexai"] = vertexai
        self.holder["api_key"] = api_key
        self.models = FakeModels(self.holder)


def _fake_google_modules():
    google_module = pytypes.ModuleType("google")
    genai_module = pytypes.ModuleType("google.genai")
    types_module = pytypes.SimpleNamespace(
        Content=FakeContent,
        FunctionCall=FakeFunctionCall,
        FunctionDeclaration=FakeFunctionDeclaration,
        GenerateContentConfig=FakeGenerateContentConfig,
        Part=FakePart,
        Tool=FakeTool,
    )
    genai_module.Client = FakeClient
    genai_module.types = types_module
    google_module.genai = genai_module
    return {
        "google": google_module,
        "google.genai": genai_module,
    }


def test_google_genai_transport_converts_tools_and_tool_calls():
    FakeClient.holder = {}
    client = LLMClient(LLMConfig(
        provider="google-cloud",
        api_key="cloud-api-key",
        model="gemini-3.1-flash-lite-preview",
        transport=GOOGLE_GENAI_TRANSPORT,
    ))

    tools = [{
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read a file.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    }]

    with patch.dict(sys.modules, _fake_google_modules()):
        response = client.complete(
            [{"role": "system", "content": "Be concise."}, {"role": "user", "content": "read main"}],
            tools=tools,
        )

    assert FakeClient.holder["vertexai"] is True
    assert FakeClient.holder["api_key"] == "cloud-api-key"
    assert FakeClient.holder["model"] == "gemini-3.1-flash-lite-preview"
    assert FakeClient.holder["config"].kwargs["system_instruction"] == "Be concise."

    declaration = FakeClient.holder["config"].kwargs["tools"][0].function_declarations[0]
    assert declaration.name == "read"
    assert declaration.parameters["type"] == "OBJECT"
    assert declaration.parameters["properties"]["path"]["type"] == "STRING"

    assert response.has_tool_calls
    assert response.tool_calls[0].id == "call_google_0"
    assert response.tool_calls[0].name == "read"
    assert response.tool_calls[0].arguments == {"path": "main.py"}
    assert response.usage["reasoning_tokens"] == 2
