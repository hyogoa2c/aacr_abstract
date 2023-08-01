import pytest
from src.utils import (
    ApiKeyManager, OpenAiHandler, SummaryGenerator, AbstractSummaryGenerator
)
from unittest.mock import MagicMock, patch


class TestApiKeyManager:
    def test_get_openai_key(self):
        secrets = {"OPENAI_API_KEY": "test_key"}
        api_key_manager = ApiKeyManager(secrets)
        assert api_key_manager.get_openai_key() == "test_key"

    def test_get_gcp_service_account(self):
        secrets = {"gcp_service_account": "test_account"}
        api_key_manager = ApiKeyManager(secrets)
        assert api_key_manager.get_gcp_service_account() == "test_account"


@pytest.fixture
def openai_handler():
    api_key_manager = MagicMock(spec=ApiKeyManager)
    return OpenAiHandler(api_key_manager)


@patch("openai.Completion")
def test_chat_completion_request(mock_completion, openai_handler):
    messages = ["Hello", "How are you?"]
    result = []
    model = "davinci"
    functions = []
    mock_completion().choices[0].text = "I'm doing well, thanks for asking."
    openai_handler.api_key_manager.get_openai_key.return_value = "test_key"
    openai_handler.chat_completion_request(messages, result, model, functions)
    assert result


@patch("openai.Embedding")
def test_generate_text_embedding(mock_embedding, openai_handler):
    text = "Hello, world!"
    model = "davinci"
    embedding = [0.1, 0.2, 0.3]
    mock_embedding.create.return_value = {"data": [{"embedding": embedding}]}
    openai_handler.api_key_manager.get_openai_key.return_value = "test_key"
    result = openai_handler.generate_text_embedding(text, model)
    assert result == mock_embedding.create.return_value["data"][0]["embedding"]
