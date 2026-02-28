"""LLM client for PyUT Agent."""

import asyncio
import logging
from typing import AsyncIterator, Iterator, Optional

from .config import LLMConfig, LLMProvider

logger = logging.getLogger(__name__)


class LLMClient:
    """Unified LLM client supporting multiple providers.
    
    Features:
    - Synchronous and asynchronous generation
    - Streaming support
    - Automatic retry with exponential backoff
    - Token counting
    - Connection testing
    """
    
    def __init__(
        self,
        endpoint: str,
        api_key: str,
        model: str,
        ca_cert: Optional[str] = None,
        timeout: int = 300,
        max_retries: int = 5,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        provider: str = "openai",
        **kwargs
    ):
        """Initialize LLM client.
        
        Args:
            endpoint: API endpoint URL
            api_key: API key
            model: Model name
            ca_cert: Path to CA certificate
            timeout: Request timeout
            max_retries: Maximum retry attempts
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            provider: Provider name
        """
        self.endpoint = endpoint
        self.api_key = api_key
        self.model = model
        self.ca_cert = ca_cert
        self.timeout = timeout
        self.max_retries = max_retries
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.provider = provider
        
        # Initialize LangChain client (lazy import)
        self._client = None
        
        # Usage tracking
        self._total_calls = 0
        self._total_tokens = 0
    
    def _get_client(self):
        """Get or create LangChain client (lazy import)."""
        if self._client is None:
            try:
                from langchain_openai import ChatOpenAI
                
                # Configure SSL if CA cert provided
                http_client = None
                if self.ca_cert:
                    import httpx
                    http_client = httpx.Client(
                        verify=self.ca_cert,
                        timeout=self.timeout
                    )
                
                self._client = ChatOpenAI(
                    model=self.model,
                    api_key=self.api_key,
                    base_url=self.endpoint,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    timeout=self.timeout,
                    max_retries=self.max_retries,
                    http_client=http_client,
                )
            except Exception as e:
                logger.error(f"Failed to initialize LLM client: {e}")
                raise
        
        return self._client
    
    @classmethod
    def from_config(cls, config: LLMConfig) -> "LLMClient":
        """Create client from configuration.
        
        Args:
            config: LLM configuration
            
        Returns:
            LLMClient instance
        """
        return cls(
            endpoint=config.endpoint,
            api_key=config.api_key.get_secret_value(),
            model=config.model,
            ca_cert=str(config.ca_cert) if config.ca_cert else None,
            timeout=config.timeout,
            max_retries=config.max_retries,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            provider=config.provider,
        )
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """Generate text synchronously.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            
        Returns:
            Generated text
        """
        from langchain_core.messages import HumanMessage, SystemMessage
        
        client = self._get_client()
        
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))
        
        try:
            response = client.invoke(messages)
            self._total_calls += 1
            # Estimate tokens (actual count may vary)
            self._total_tokens += len(prompt.split()) + len(response.content.split())
            return response.content
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
    
    async def agenerate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """Generate text asynchronously.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            
        Returns:
            Generated text
        """
        from langchain_core.messages import HumanMessage, SystemMessage
        
        client = self._get_client()
        
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))
        
        try:
            response = await client.ainvoke(messages)
            self._total_calls += 1
            self._total_tokens += len(prompt.split()) + len(response.content.split())
            return response.content
        except Exception as e:
            logger.error(f"Async generation failed: {e}")
            raise
    
    def stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> Iterator[str]:
        """Stream text generation.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            
        Yields:
            Text chunks
        """
        from langchain_core.messages import HumanMessage, SystemMessage
        
        client = self._get_client()
        
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))
        
        try:
            for chunk in client.stream(messages):
                if chunk.content:
                    yield chunk.content
            self._total_calls += 1
        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            raise
    
    async def astream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> AsyncIterator[str]:
        """Stream text generation asynchronously.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            
        Yields:
            Text chunks
        """
        from langchain_core.messages import HumanMessage, SystemMessage
        
        client = self._get_client()
        
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))
        
        try:
            async for chunk in client.astream(messages):
                if chunk.content:
                    yield chunk.content
            self._total_calls += 1
        except Exception as e:
            logger.error(f"Async streaming failed: {e}")
            raise
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text (approximate).
        
        Args:
            text: Input text
            
        Returns:
            Estimated token count
        """
        # Simple approximation: ~4 characters per token
        return len(text) // 4
    
    async def test_connection(self) -> tuple[bool, str]:
        """Test connection to LLM service.
        
        Returns:
            Tuple of (success, message)
        """
        try:
            response = await self.agenerate(
                prompt="Hello, this is a connection test. Please respond with 'OK'.",
                system_prompt="You are a helpful assistant."
            )
            if response and len(response) > 0:
                return True, f"Connection successful! Model: {self.model}"
            return False, "Empty response from model"
        except Exception as e:
            return False, f"Connection failed: {str(e)}"
    
    def get_usage_stats(self) -> dict:
        """Get usage statistics.
        
        Returns:
            Dictionary with usage stats
        """
        return {
            "total_calls": self._total_calls,
            "total_tokens": self._total_tokens,
            "model": self.model,
            "provider": self.provider,
        }
    
    def reset_stats(self):
        """Reset usage statistics."""
        self._total_calls = 0
        self._total_tokens = 0
