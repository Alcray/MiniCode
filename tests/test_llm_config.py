"""Tests for LLM provider configuration."""

import os
import pytest
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


def test_transport_resets_for_non_google_providers():
    """transport must be reset to OPENAI_TRANSPORT when provider is not google-cloud,
    even if it was previously set to GOOGLE_GENAI_TRANSPORT."""
    config = LLMConfig(
        provider="openai",
        transport=GOOGLE_GENAI_TRANSPORT,  # simulate previously-set value
        api_key="sk-test",
    )
    config.apply_provider_defaults()
    assert config.transport == OPENAI_TRANSPORT


def test_apply_provider_defaults_resets_transport_for_google_api_provider():
    """Verifies transport is reset to OPENAI_TRANSPORT for Google API (AI Studio) provider
    even when previously set to GOOGLE_GENAI_TRANSPORT."""
    config = LLMConfig(
        provider="gemini",
        transport=GOOGLE_GENAI_TRANSPORT,
        api_key="gemini-key",
    )
    config.apply_provider_defaults(api_key_explicit=True)
    assert config.transport == OPENAI_TRANSPORT


def test_google_cloud_provider_raises_when_no_project_and_no_base_url():
    """google-cloud provider with no project_id and no explicit base_url must raise."""
    with patch.dict(
        os.environ,
        {
            "MINICODE_PROVIDER": "google-cloud",
            "GOOGLE_CLOUD_ACCESS_TOKEN": "ya29.token",
        },
        clear=True,
    ):
        with pytest.raises(ValueError, match="MINICODE_GOOGLE_CLOUD_PROJECT"):
            LLMConfig.from_env()


def test_google_cloud_provider_genai_transport_does_not_require_project_id():
    """google-cloud + API key (GenAI transport) should NOT require project_id."""
    with patch.dict(
        os.environ,
        {
            "MINICODE_PROVIDER": "google-cloud",
            "GOOGLE_CLOUD_API_KEY": "cloud-key",
        },
        clear=True,
    ):
        config = LLMConfig.from_env()

    assert config.transport == GOOGLE_GENAI_TRANSPORT
    assert config.api_key == "cloud-key"
