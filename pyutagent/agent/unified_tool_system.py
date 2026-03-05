"""Unified Tool System - Standardized tool interface and execution.

This module provides:
- ToolExecutor: Unified tool execution
- RetryMixin: Retry logic for tools
- CacheMixin: Caching for tool results
"""

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ToolExecutionConfig:
    """Configuration for tool execution."""
    timeout_seconds: int = 30
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    enable_cache: bool = True
    cache_ttl_seconds: int = 300


@dataclass
class ToolCall:
    """A tool call request."""
    tool_name: str
    parameters: Dict[str, Any]
    call_id: Optional[str] = None
    timestamp: Optional[datetime] = None


@dataclass
class ToolResponse:
    """A tool call response."""
    tool_name: str
    success: bool
    output: Any
    error: Optional[str] = None
    duration_ms: int = 0
    call_id: Optional[str] = None
    cached: bool = False


class ToolExecutor:
    """Unified tool executor.

    Features:
    - Standardized execution
    - Retry logic
    - Result caching
    - Timeout handling
    - Metrics collection
    """

    def __init__(self, config: Optional[ToolExecutionConfig] = None):
        """Initialize tool executor.

        Args:
            config: Execution configuration
        """
        self.config = config or ToolExecutionConfig()
        self._tool_registry: Dict[str, Callable] = {}
        self._cache: Dict[str, tuple] = {}
        self._metrics: Dict[str, List[float]] = {}

    def register_tool(self, name: str, func: Callable):
        """Register a tool.

        Args:
            name: Tool name
            func: Tool function
        """
        self._tool_registry[name] = func
        logger.debug(f"[ToolExecutor] Registered tool: {name}")

    def unregister_tool(self, name: str) -> bool:
        """Unregister a tool.

        Args:
            name: Tool name

        Returns:
            True if unregistered
        """
        if name in self._tool_registry:
            del self._tool_registry[name]
            return True
        return False

    async def execute(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        call_id: Optional[str] = None
    ) -> ToolResponse:
        """Execute a tool.

        Args:
            tool_name: Tool name
            parameters: Tool parameters
            call_id: Optional call ID

        Returns:
            Tool response
        """
        if tool_name not in self._tool_registry:
            return ToolResponse(
                tool_name=tool_name,
                success=False,
                output=None,
                error=f"Tool not found: {tool_name}",
                call_id=call_id
            )

        cache_key = self._get_cache_key(tool_name, parameters)

        if self.config.enable_cache and cache_key in self._cache:
            cached_result, timestamp = self._cache[cache_key]
            if time.time() - timestamp < self.config.cache_ttl_seconds:
                logger.debug(f"[ToolExecutor] Cache hit: {tool_name}")
                return ToolResponse(
                    tool_name=tool_name,
                    success=cached_result.success,
                    output=cached_result.output,
                    error=cached_result.error,
                    duration_ms=0,
                    call_id=call_id,
                    cached=True
                )

        tool_func = self._tool_registry[tool_name]
        last_error = None

        for attempt in range(self.config.max_retries):
            try:
                start_time = time.time()

                if asyncio.iscoroutinefunction(tool_func):
                    result = await asyncio.wait_for(
                        tool_func(**parameters),
                        timeout=self.config.timeout_seconds
                    )
                else:
                    result = tool_func(**parameters)

                duration_ms = int((time.time() - start_time) * 1000)

                response = ToolResponse(
                    tool_name=tool_name,
                    success=True,
                    output=result,
                    duration_ms=duration_ms,
                    call_id=call_id
                )

                if self.config.enable_cache:
                    self._cache[cache_key] = (response, time.time())

                self._record_metrics(tool_name, duration_ms, True)

                logger.info(f"[ToolExecutor] Executed: {tool_name} in {duration_ms}ms")
                return response

            except asyncio.TimeoutError:
                last_error = f"Timeout after {self.config.timeout_seconds}s"
                logger.warning(f"[ToolExecutor] Timeout: {tool_name} (attempt {attempt + 1})")

            except Exception as e:
                last_error = str(e)
                logger.warning(f"[ToolExecutor] Error: {tool_name} - {e}")
                self._record_metrics(tool_name, 0, False)

            if attempt < self.config.max_retries - 1:
                await asyncio.sleep(self.config.retry_delay_seconds * (attempt + 1))

        return ToolResponse(
            tool_name=tool_name,
            success=False,
            output=None,
            error=last_error,
            call_id=call_id
        )

    async def execute_batch(
        self,
        calls: List[ToolCall],
        parallel: bool = True
    ) -> List[ToolResponse]:
        """Execute multiple tool calls.

        Args:
            calls: List of tool calls
            parallel: Execute in parallel

        Returns:
            List of tool responses
        """
        if parallel:
            tasks = [
                self.execute(call.tool_name, call.parameters, call.call_id)
                for call in calls
            ]
            return await asyncio.gather(*tasks)
        else:
            results = []
            for call in calls:
                result = await self.execute(call.tool_name, call.parameters, call.call_id)
                results.append(result)
            return results

    def _get_cache_key(self, tool_name: str, parameters: Dict[str, Any]) -> str:
        """Get cache key for tool call."""
        param_str = json.dumps(parameters, sort_keys=True)
        return hashlib.sha256(f"{tool_name}:{param_str}".encode()).hexdigest()

    def _record_metrics(self, tool_name: str, duration_ms: int, success: bool):
        """Record execution metrics."""
        if tool_name not in self._metrics:
            self._metrics[tool_name] = []

        if success and duration_ms > 0:
            self._metrics[tool_name].append(duration_ms)

    def get_metrics(self, tool_name: Optional[str] = None) -> Dict[str, Any]:
        """Get execution metrics.

        Args:
            tool_name: Specific tool or None for all

        Returns:
            Metrics dictionary
        """
        if tool_name:
            durations = self._metrics.get(tool_name, [])
            return {
                "tool": tool_name,
                "executions": len(durations),
                "avg_duration_ms": sum(durations) / len(durations) if durations else 0,
                "min_duration_ms": min(durations) if durations else 0,
                "max_duration_ms": max(durations) if durations else 0
            }

        return {
            tool: self.get_metrics(tool)
            for tool in self._metrics.keys()
        }

    def clear_cache(self):
        """Clear the execution cache."""
        self._cache.clear()
        logger.info("[ToolExecutor] Cache cleared")


class RetryMixin:
    """Mixin providing retry logic."""

    def __init__(self):
        self._max_retries = 3
        self._retry_delay = 1.0

    async def execute_with_retry(
        self,
        func: Callable,
        *args,
        max_retries: Optional[int] = None,
        **kwargs
    ) -> Any:
        """Execute function with retry.

        Args:
            func: Function to execute
            args: Function arguments
            max_retries: Max retry attempts
            kwargs: Function keyword arguments

        Returns:
            Function result
        """
        max_retries = max_retries or self._max_retries
        last_error = None

        for attempt in range(max_retries):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)

            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    await asyncio.sleep(self._retry_delay * (attempt + 1))
                    logger.warning(f"[RetryMixin] Retry {attempt + 1}/{max_retries}: {e}")

        raise last_error


class CacheMixin:
    """Mixin providing caching logic."""

    def __init__(self):
        self._cache: Dict[str, tuple] = {}
        self._cache_ttl = 300

    def _get_cache(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key in self._cache:
            value, timestamp = self._cache[key]
            if time.time() - timestamp < self._cache_ttl:
                return value
            del self._cache[key]
        return None

    def _set_cache(self, key: str, value: Any):
        """Set value in cache."""
        self._cache[key] = (value, time.time())

    def _clear_cache(self):
        """Clear cache."""
        self._cache.clear()


def create_tool_executor(
    timeout_seconds: int = 30,
    max_retries: int = 3,
    enable_cache: bool = True
) -> ToolExecutor:
    """Create a configured tool executor.

    Args:
        timeout_seconds: Execution timeout
        max_retries: Max retry attempts
        enable_cache: Enable caching

    Returns:
        ToolExecutor instance
    """
    config = ToolExecutionConfig(
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        enable_cache=enable_cache
    )
    return ToolExecutor(config)
