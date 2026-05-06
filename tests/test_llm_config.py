"""Tests for LLM provider configuration."""

import os
from unittest.mock import patch

from minicode.llm import GOOGLE_GENAI_TRANSPORT, OPENAI_TRANSPORT, LLMConfig


def test_openai_env_config_uses_existing_shape():
    with patch.dict(
        os.environ,
        {
            "OPENAI_API_KEY": "sk-test",
            "MINICODE_MODEL": "gpt-4o-mini",
        },
        clear=True,
    ):
        config = LLMConfig.from_env()

    assert config.provider == "openai"
    assert config.base_url == "https://api.openai.com/v1"
    assert config.api_key == "sk-test"
    assert config.model == "gpt-4o-mini"


def test_google_cloud_provider_builds_openai_compatible_config():
    with patch.dict(
        os.environ,
        {
            "MINICODE_PROVIDER": "google-cloud",
            "MINICODE_GOOGLE_CLOUD_PROJECT": "my-project",
            "MINICODE_GOOGLE_CLOUD_LOCATION": "europe-west4",
            "MINICODE_GOOGLE_CLOUD_ACCESS_TOKEN": "ya29.token",
            "MINICODE_MODEL": "gemini-2.5-pro",
        },
        clear=True,
    ):
        config = LLMConfig.from_env()

    assert config.provider == "google-cloud"
    assert config.base_url == (
        "https://europe-west4-aiplatform.googleapis.com/v1beta1/"
        "projects/my-project/locations/europe-west4/endpoints/openapi"
    )
    assert config.api_key == "ya29.token"
    assert config.model == "google/gemini-2.5-pro"
    assert config.transport == OPENAI_TRANSPORT


def test_google_cloud_provider_uses_genai_transport_for_api_key():
    with patch.dict(
        os.environ,
        {
            "MINICODE_PROVIDER": "google-cloud",
            "GOOGLE_CLOUD_API_KEY": "cloud-api-key",
            "MINICODE_MODEL": "google/gemini-3.1-flash-lite-preview",
        },
        clear=True,
    ):
        config = LLMConfig.from_env()

    assert config.api_key == "cloud-api-key"
    assert config.model == "gemini-3.1-flash-lite-preview"
    assert config.transport == GOOGLE_GENAI_TRANSPORT


def test_google_provider_uses_direct_api_key_endpoint():
    with patch.dict(
        os.environ,
        {
            "MINICODE_PROVIDER": "google",
            "GEMINI_API_KEY": "gemini-key",
            "MINICODE_MODEL": "gemini-2.5-pro",
        },
        clear=True,
    ):
        config = LLMConfig.from_env()

    assert config.provider == "google"
    assert config.base_url == "https://generativelanguage.googleapis.com/v1beta/openai"
    assert config.api_key == "gemini-key"
    assert config.model == "gemini-2.5-pro"


def test_google_provider_defaults_model_for_direct_api_key_endpoint():
    with patch.dict(
        os.environ,
        {
            "MINICODE_PROVIDER": "gemini",
            "MINICODE_GOOGLE_API_KEY": "gemini-key",
        },
        clear=True,
    ):
        config = LLMConfig.from_env()

    assert config.base_url == "https://generativelanguage.googleapis.com/v1beta/openai"
    assert config.api_key == "gemini-key"
    assert config.model == "gemini-2.5-pro"


def test_google_provider_prefers_google_specific_key_over_openai_key():
    with patch.dict(
        os.environ,
        {
            "MINICODE_PROVIDER": "google",
            "OPENAI_API_KEY": "openai-key",
            "GEMINI_API_KEY": "gemini-key",
        },
        clear=True,
    ):
        config = LLMConfig.from_env()

    assert config.api_key == "gemini-key"


def test_google_provider_keeps_minicode_api_key_as_explicit_override():
    with patch.dict(
        os.environ,
        {
            "MINICODE_PROVIDER": "google",
            "MINICODE_API_KEY": "explicit-key",
            "GEMINI_API_KEY": "gemini-key",
        },
        clear=True,
    ):
        config = LLMConfig.from_env()

    assert config.api_key == "explicit-key"


def test_google_cloud_provider_respects_explicit_base_url_and_prefixed_model():
    with patch.dict(
        os.environ,
        {
            "MINICODE_PROVIDER": "vertex-ai",
            "MINICODE_BASE_URL": "https://example.test/v1",
            "GOOGLE_CLOUD_ACCESS_TOKEN": "ya29.token",
            "MINICODE_MODEL": "google/gemini-custom",
        },
        clear=True,
    ):
        config = LLMConfig.from_env()

    assert config.base_url == "https://example.test/v1"
    assert config.api_key == "ya29.token"
    assert config.model == "google/gemini-custom"
