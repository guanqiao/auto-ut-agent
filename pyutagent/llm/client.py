"""LLM client for PyUT Agent."""

import asyncio
import logging
import time
from typing import AsyncIterator, Iterator, Optional

from ..core.config import LLMConfig, LLMProvider

logger = logging.getLogger(__name__)


class LLMClient:
    """Unified LLM client supporting multiple providers.
    
    Features:
    - Synchronous and asynchronous generation
    - Streaming support
    - Automatic retry with exponential backoff
    - Token counting
    - Connection testing
    - Performance statistics
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
        
        self._client = None
        
        self._total_calls = 0
        self._total_tokens = 0
        
        # Performance statistics
        self._call_stats = {
            "generate": [],
            "agenerate": [],
            "complete": [],
            "stream": [],
            "astream": [],
        }
        
        logger.info(f"[LLMClient] Initializing - Model: {model}, Provider: {provider}, Endpoint: {endpoint}")
        logger.debug(f"[LLMClient] Configuration - Timeout: {timeout}s, MaxRetries: {max_retries}, Temperature: {temperature}")
        if ca_cert:
            logger.info(f"[LLMClient] Using custom CA certificate: {ca_cert}")
    
    def _get_client(self):
        """Get or create LangChain client (lazy import)."""
        if self._client is None:
            try:
                logger.debug(f"[LLMClient] Creating LangChain client - Model: {self.model}")
                from langchain_openai import ChatOpenAI

                http_client = None
                http_async_client = None
                if self.ca_cert:
                    import httpx
                    from pathlib import Path
                    cert_path = Path(self.ca_cert)
                    if cert_path.exists():
                        logger.info(f"[LLMClient] CA certificate file exists: {cert_path}")
                        http_client = httpx.Client(
                            verify=str(cert_path),
                            timeout=self.timeout
                        )
                        http_async_client = httpx.AsyncClient(
                            verify=str(cert_path),
                            timeout=self.timeout
                        )
                        logger.info(f"[LLMClient] Configured HTTPS clients with CA certificate: {cert_path}")
                    else:
                        logger.warning(f"[LLMClient] CA certificate file not found: {cert_path}")

                self._client = ChatOpenAI(
                    model=self.model,
                    api_key=self.api_key,
                    base_url=self.endpoint,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    timeout=self.timeout,
                    max_retries=self.max_retries,
                    http_client=http_client,
                    http_async_client=http_async_client,
                )
                logger.info(f"[LLMClient] LangChain client created successfully - http_client: {http_client is not None}, http_async_client: {http_async_client is not None}")
            except Exception as e:
                logger.exception(f"[LLMClient] Failed to create LangChain client: {e}")
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
        logger.info(f"[LLMClient] Creating client from config - ConfigID: {config.id}, Name: {config.name}")
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
        
        prompt_preview = prompt[:100] + "..." if len(prompt) > 100 else prompt
        logger.info(f"[LLM] Starting sync generation - Model: {self.model}, Endpoint: {self.endpoint}, Provider: {self.provider}, PromptLength: {len(prompt)}")
        logger.debug(f"[LLM] Prompt preview: {prompt_preview}")
        
        start_time = time.time()
        try:
            response = client.invoke(messages)
            self._total_calls += 1
            self._total_tokens += len(prompt.split()) + len(response.content.split())
            elapsed = time.time() - start_time
            self._call_stats["generate"].append(elapsed)
            logger.info(f"[LLM] Sync generation complete - Model: {self.model}, Endpoint: {self.endpoint}, ResponseLength: {len(response.content)}, Elapsed: {elapsed:.2f}s")
            return response.content
        except Exception as e:
            elapsed = time.time() - start_time
            self._call_stats["generate"].append(elapsed)
            logger.exception(f"[LLM] Sync generation failed - Model: {self.model}, Endpoint: {self.endpoint}, Elapsed: {elapsed:.2f}s, Error: {e}")
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
        
        prompt_preview = prompt[:100] + "..." if len(prompt) > 100 else prompt
        logger.info(f"[LLM] 🚀 Starting async generation - Model: {self.model}, Endpoint: {self.endpoint}, Provider: {self.provider}, PromptLength: {len(prompt)} chars")
        logger.info(f"[LLM] ⏳ Waiting for LLM response... (this may take 10-60 seconds)")
        logger.debug(f"[LLM] Prompt preview: {prompt_preview}")
        
        start_time = time.time()
        try:
            response = await client.ainvoke(messages)
            self._total_calls += 1
            self._total_tokens += len(prompt.split()) + len(response.content.split())
            elapsed = time.time() - start_time
            self._call_stats["agenerate"].append(elapsed)
            logger.info(f"[LLM] ✅ Async generation complete - Model: {self.model}, Endpoint: {self.endpoint}, ResponseLength: {len(response.content)} chars, Elapsed: {elapsed:.2f}s")
            return response.content
        except Exception as e:
            elapsed = time.time() - start_time
            self._call_stats["agenerate"].append(elapsed)
            logger.exception(f"[LLM] ❌ Async generation failed - Model: {self.model}, Endpoint: {self.endpoint}, Elapsed: {elapsed:.2f}s, Error: {e}")
            raise
    
    async def complete(
        self,
        messages: list[dict[str, str]],
        **kwargs
    ) -> str:
        """Complete a conversation with messages.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            **kwargs: Additional arguments
            
        Returns:
            Generated response text
        """
        from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
        
        client = self._get_client()
        
        lc_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "user":
                lc_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
            else:
                lc_messages.append(HumanMessage(content=content))
        
        logger.info(f"[LLM] Starting conversation completion - Model: {self.model}, Endpoint: {self.endpoint}, Provider: {self.provider}, MessageCount: {len(messages)}")
        
        start_time = time.time()
        try:
            response = await client.ainvoke(lc_messages)
            self._total_calls += 1
            total_input = sum(len(m.content.split()) for m in lc_messages)
            self._total_tokens += total_input + len(response.content.split())
            elapsed = time.time() - start_time
            self._call_stats["complete"].append(elapsed)
            logger.info(f"[LLM] Conversation completion done - Model: {self.model}, Endpoint: {self.endpoint}, ResponseLength: {len(response.content)}, Elapsed: {elapsed:.2f}s")
            return response.content
        except Exception as e:
            elapsed = time.time() - start_time
            self._call_stats["complete"].append(elapsed)
            logger.exception(f"[LLM] Conversation completion failed - Model: {self.model}, Endpoint: {self.endpoint}, Elapsed: {elapsed:.2f}s, Error: {e}")
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
        
        logger.info(f"[LLM] Starting stream generation - Model: {self.model}, Endpoint: {self.endpoint}, Provider: {self.provider}, PromptLength: {len(prompt)}")
        
        start_time = time.time()
        try:
            chunk_count = 0
            for chunk in client.stream(messages):
                if chunk.content:
                    chunk_count += 1
                    yield chunk.content
            self._total_calls += 1
            elapsed = time.time() - start_time
            self._call_stats["stream"].append(elapsed)
            logger.info(f"[LLM] Stream generation complete - Model: {self.model}, Endpoint: {self.endpoint}, TotalChunks: {chunk_count}, Elapsed: {elapsed:.2f}s")
        except Exception as e:
            elapsed = time.time() - start_time
            self._call_stats["stream"].append(elapsed)
            logger.exception(f"[LLM] Stream generation failed - Model: {self.model}, Endpoint: {self.endpoint}, Elapsed: {elapsed:.2f}s, Error: {e}")
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
        
        logger.info(f"[LLM] Starting async stream generation - Model: {self.model}, Endpoint: {self.endpoint}, Provider: {self.provider}, PromptLength: {len(prompt)}")
        
        start_time = time.time()
        try:
            chunk_count = 0
            async for chunk in client.astream(messages):
                if chunk.content:
                    chunk_count += 1
                    yield chunk.content
            self._total_calls += 1
            elapsed = time.time() - start_time
            self._call_stats["astream"].append(elapsed)
            logger.info(f"[LLM] Async stream generation complete - Model: {self.model}, Endpoint: {self.endpoint}, TotalChunks: {chunk_count}, Elapsed: {elapsed:.2f}s")
        except Exception as e:
            elapsed = time.time() - start_time
            self._call_stats["astream"].append(elapsed)
            logger.exception(f"[LLM] Async stream generation failed - Model: {self.model}, Endpoint: {self.endpoint}, Elapsed: {elapsed:.2f}s, Error: {e}")
            raise
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text (approximate).
        
        Args:
            text: Input text
            
        Returns:
            Estimated token count
        """
        return len(text) // 4
    
    async def test_connection(self) -> tuple[bool, str]:
        """Test connection to LLM service.
        
        Returns:
            Tuple of (success, message)
        """
        logger.info(f"[LLM] Starting connection test - Model: {self.model}, Endpoint: {self.endpoint}")
        start_time = time.time()
        try:
            response = await self.agenerate(
                prompt="Hello, this is a connection test. Please respond with 'OK'.",
                system_prompt="You are a helpful assistant."
            )
            elapsed = time.time() - start_time
            if response and len(response) > 0:
                logger.info(f"[LLM] Connection test successful - Elapsed: {elapsed:.2f}s, Model: {self.model}")
                return True, f"Connection successful! Model: {self.model}"
            logger.warning(f"[LLM] Connection test returned empty response - Elapsed: {elapsed:.2f}s")
            return False, "Empty response from model"
        except Exception as e:
            elapsed = time.time() - start_time
            logger.exception(f"[LLM] Connection test failed - Elapsed: {elapsed:.2f}s, Error: {e}")
            return False, f"Connection failed: {str(e)}"
    
    def get_usage_stats(self) -> dict:
        """Get usage statistics.
        
        Returns:
            Dictionary with usage stats
        """
        stats = {
            "total_calls": self._total_calls,
            "total_tokens": self._total_tokens,
            "model": self.model,
            "provider": self.provider,
            "endpoint": self.endpoint,
        }
        logger.debug(f"[LLM] Usage stats: {stats}")
        return stats
    
    def get_performance_report(self) -> dict:
        """Get detailed performance statistics report.
        
        Returns:
            Dictionary with performance stats for each operation type
        """
        report = {
            "model": self.model,
            "provider": self.provider,
            "endpoint": self.endpoint,
            "operations": {}
        }
        
        for op_name, times in self._call_stats.items():
            if times:
                report["operations"][op_name] = {
                    "count": len(times),
                    "total_time": sum(times),
                    "avg_time": sum(times) / len(times),
                    "min_time": min(times),
                    "max_time": max(times),
                }
            else:
                report["operations"][op_name] = {
                    "count": 0,
                    "total_time": 0.0,
                    "avg_time": 0.0,
                    "min_time": 0.0,
                    "max_time": 0.0,
                }
        
        logger.info(f"[LLM] Performance report generated - Model: {self.model}, Operations: {list(report['operations'].keys())}")
        return report
    
    def print_performance_report(self):
        """Print performance report to logs."""
        report = self.get_performance_report()
        
        logger.info("=" * 60)
        logger.info(f"[LLM] Performance Report - Model: {report['model']}, Provider: {report['provider']}")
        logger.info(f"[LLM] Endpoint: {report['endpoint']}")
        logger.info("-" * 60)
        
        for op_name, stats in report["operations"].items():
            if stats["count"] > 0:
                logger.info(f"[LLM] {op_name:15} - Count: {stats['count']:3d}, "
                           f"Total: {stats['total_time']:6.2f}s, "
                           f"Avg: {stats['avg_time']:5.2f}s, "
                           f"Min: {stats['min_time']:5.2f}s, "
                           f"Max: {stats['max_time']:5.2f}s")
        
        logger.info("=" * 60)
    
    def reset_stats(self):
        """Reset usage statistics."""
        logger.info(f"[LLM] Resetting usage stats - Previous: calls={self._total_calls}, tokens={self._total_tokens}")
        self._total_calls = 0
        self._total_tokens = 0
        for op_name in self._call_stats:
            self._call_stats[op_name] = []
