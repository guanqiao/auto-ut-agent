"""Tests for LLM client module."""

import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from pydantic import SecretStr

from pyutagent.llm.client import LLMClient
from pyutagent.llm.config import LLMConfig, LLMProvider


class TestLLMClient:
    """Tests for LLMClient class."""

    @pytest.fixture
    def client(self):
        """Create an LLMClient instance."""
        return LLMClient(
            endpoint="https://api.openai.com/v1",
            api_key="test-key",
            model="gpt-4",
            timeout=60,
            max_retries=3,
            temperature=0.5,
            max_tokens=2048,
            provider="openai"
        )

    def test_initialization(self, client):
        """Test client initialization."""
        assert client.endpoint == "https://api.openai.com/v1"
        assert client.api_key == "test-key"
        assert client.model == "gpt-4"
        assert client.timeout == 60
        assert client.max_retries == 3
        assert client.temperature == 0.5
        assert client.max_tokens == 2048
        assert client.provider == "openai"
        assert client._client is None
        assert client._total_calls == 0
        assert client._total_tokens == 0

    def test_initialization_with_ca_cert(self):
        """Test client initialization with CA cert."""
        client = LLMClient(
            endpoint="https://api.example.com",
            api_key="key",
            model="model",
            ca_cert="/path/to/cert.pem"
        )
        assert client.ca_cert == "/path/to/cert.pem"

    def test_from_config(self):
        """Test creating client from config."""
        config = LLMConfig(
            endpoint="https://api.test.com",
            api_key=SecretStr("config-key"),
            model="gpt-3.5",
            timeout=120,
            max_retries=5,
            temperature=0.8,
            max_tokens=4096,
            provider=LLMProvider.ANTHROPIC
        )

        client = LLMClient.from_config(config)

        assert client.endpoint == "https://api.test.com"
        assert client.api_key == "config-key"
        assert client.model == "gpt-3.5"
        assert client.timeout == 120
        assert client.max_retries == 5
        assert client.temperature == 0.8
        assert client.max_tokens == 4096
        assert client.provider == LLMProvider.ANTHROPIC

    def test_from_config_with_ca_cert(self):
        """Test creating client from config with CA cert."""
        from pathlib import Path
        config = LLMConfig(
            endpoint="https://api.test.com",
            api_key=SecretStr("key"),
            model="model",
            ca_cert=Path("/path/to/cert.pem")
        )

        client = LLMClient.from_config(config)

        # On Windows, path separators may be converted
        assert "cert.pem" in client.ca_cert

    @patch('langchain_openai.ChatOpenAI')
    def test_get_client(self, mock_chat_openai, client):
        """Test getting/creating LangChain client."""
        mock_chat_openai.return_value = Mock()

        lc_client = client._get_client()

        assert lc_client is not None
        mock_chat_openai.assert_called_once()
        call_kwargs = mock_chat_openai.call_args.kwargs
        assert call_kwargs['model'] == "gpt-4"
        assert call_kwargs['api_key'] == "test-key"
        assert call_kwargs['base_url'] == "https://api.openai.com/v1"
        assert call_kwargs['temperature'] == 0.5
        assert call_kwargs['max_tokens'] == 2048
        assert call_kwargs['timeout'] == 60
        assert call_kwargs['max_retries'] == 3

    @patch('langchain_openai.ChatOpenAI')
    def test_get_client_caches(self, mock_chat_openai, client):
        """Test that client is cached."""
        mock_chat_openai.return_value = Mock()

        client1 = client._get_client()
        client2 = client._get_client()

        assert client1 is client2
        mock_chat_openai.assert_called_once()

    @patch('langchain_openai.ChatOpenAI')
    @patch('httpx.Client')
    def test_get_client_with_ca_cert(self, mock_httpx_client, mock_chat_openai):
        """Test creating client with CA cert."""
        client = LLMClient(
            endpoint="https://api.example.com",
            api_key="key",
            model="model",
            ca_cert="/path/to/cert.pem",
            timeout=30
        )
        mock_chat_openai.return_value = Mock()
        mock_httpx_client.return_value = Mock()

        client._get_client()

        call_kwargs = mock_chat_openai.call_args.kwargs
        assert call_kwargs['http_client'] is not None

    @patch('langchain_openai.ChatOpenAI')
    def test_generate(self, mock_chat_openai, client):
        """Test synchronous generation."""
        mock_response = Mock()
        mock_response.content = "Generated response"
        mock_chat_openai.return_value = Mock(invoke=Mock(return_value=mock_response))

        result = client.generate("Test prompt")

        assert result == "Generated response"
        assert client._total_calls == 1
        assert client._total_tokens > 0

    @patch('langchain_openai.ChatOpenAI')
    def test_generate_with_system_prompt(self, mock_chat_openai, client):
        """Test generation with system prompt."""
        mock_response = Mock()
        mock_response.content = "Response"
        mock_chat_openai.return_value = Mock(invoke=Mock(return_value=mock_response))

        result = client.generate("User prompt", "System prompt")

        assert result == "Response"

    @patch('langchain_openai.ChatOpenAI')
    def test_generate_error(self, mock_chat_openai, client):
        """Test generation error handling."""
        mock_chat_openai.return_value = Mock(invoke=Mock(side_effect=Exception("API Error")))

        with pytest.raises(Exception, match="API Error"):
            client.generate("Test prompt")

    @pytest.mark.asyncio
    @patch('langchain_openai.ChatOpenAI')
    async def test_agenerate(self, mock_chat_openai, client):
        """Test asynchronous generation."""
        mock_response = Mock()
        mock_response.content = "Async response"
        mock_chat_openai.return_value = Mock(ainvoke=AsyncMock(return_value=mock_response))

        result = await client.agenerate("Test prompt")

        assert result == "Async response"
        assert client._total_calls == 1

    @pytest.mark.asyncio
    @patch('langchain_openai.ChatOpenAI')
    async def test_complete(self, mock_chat_openai, client):
        """Test completing a conversation."""
        mock_response = Mock()
        mock_response.content = "Completion"
        mock_chat_openai.return_value = Mock(ainvoke=AsyncMock(return_value=mock_response))

        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
            {"role": "user", "content": "How are you?"}
        ]

        result = await client.complete(messages)

        assert result == "Completion"
        assert client._total_calls == 1

    @pytest.mark.asyncio
    @patch('langchain_openai.ChatOpenAI')
    async def test_complete_unknown_role(self, mock_chat_openai, client):
        """Test completing with unknown role."""
        mock_response = Mock()
        mock_response.content = "Response"
        mock_chat_openai.return_value = Mock(ainvoke=AsyncMock(return_value=mock_response))

        messages = [
            {"role": "unknown", "content": "Test"}
        ]

        result = await client.complete(messages)

        assert result == "Response"

    @patch('langchain_openai.ChatOpenAI')
    def test_stream(self, mock_chat_openai, client):
        """Test streaming generation."""
        chunks = [
            Mock(content="Hello "),
            Mock(content="world"),
            Mock(content="!"),
            Mock(content=None)  # Empty chunk
        ]
        mock_chat_openai.return_value = Mock(stream=Mock(return_value=iter(chunks)))

        results = list(client.stream("Test prompt"))

        assert results == ["Hello ", "world", "!"]
        assert client._total_calls == 1

    @pytest.mark.asyncio
    @patch('langchain_openai.ChatOpenAI')
    async def test_astream(self, mock_chat_openai, client):
        """Test async streaming generation."""
        chunks = [
            Mock(content="Chunk 1"),
            Mock(content="Chunk 2"),
        ]

        async def async_generator(messages):
            for chunk in chunks:
                yield chunk

        mock_chat_openai.return_value = Mock(astream=async_generator)

        results = []
        async for chunk in client.astream("Test prompt"):
            results.append(chunk)

        assert results == ["Chunk 1", "Chunk 2"]
        assert client._total_calls == 1

    def test_count_tokens(self, client):
        """Test token counting."""
        # Approximate: 4 chars per token
        text = "a" * 100
        count = client.count_tokens(text)
        assert count == 25  # 100 // 4

    def test_count_tokens_empty(self, client):
        """Test token counting for empty string."""
        assert client.count_tokens("") == 0

    @pytest.mark.asyncio
    @patch('langchain_openai.ChatOpenAI')
    async def test_test_connection_success(self, mock_chat_openai, client):
        """Test connection testing - success."""
        mock_response = Mock()
        mock_response.content = "OK"
        mock_chat_openai.return_value = Mock(ainvoke=AsyncMock(return_value=mock_response))

        success, message = await client.test_connection()

        assert success is True
        assert "successful" in message.lower()
        assert "gpt-4" in message

    @pytest.mark.asyncio
    @patch('langchain_openai.ChatOpenAI')
    async def test_test_connection_empty_response(self, mock_chat_openai, client):
        """Test connection testing - empty response."""
        mock_response = Mock()
        mock_response.content = ""
        mock_chat_openai.return_value = Mock(ainvoke=AsyncMock(return_value=mock_response))

        success, message = await client.test_connection()

        assert success is False
        assert "empty" in message.lower()

    @pytest.mark.asyncio
    @patch('langchain_openai.ChatOpenAI')
    async def test_test_connection_failure(self, mock_chat_openai, client):
        """Test connection testing - failure."""
        mock_chat_openai.return_value = Mock(
            ainvoke=AsyncMock(side_effect=Exception("Connection refused"))
        )

        success, message = await client.test_connection()

        assert success is False
        assert "failed" in message.lower()
        assert "connection refused" in message.lower()

    def test_get_usage_stats(self, client):
        """Test getting usage statistics."""
        client._total_calls = 10
        client._total_tokens = 5000

        stats = client.get_usage_stats()

        assert stats["total_calls"] == 10
        assert stats["total_tokens"] == 5000
        assert stats["model"] == "gpt-4"
        assert stats["provider"] == "openai"

    def test_reset_stats(self, client):
        """Test resetting statistics."""
        client._total_calls = 10
        client._total_tokens = 5000

        client.reset_stats()

        assert client._total_calls == 0
        assert client._total_tokens == 0
