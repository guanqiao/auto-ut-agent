"""Enhanced tool execution with robust error handling and retry.

This module provides:
- RetryToolExecutor: Tool execution with retry logic
- TimeoutToolExecutor: Tool execution with timeout
- RobustToolExecutor: Combined retry + timeout + fallback
"""

import asyncio
import logging
from typing import Any, Callable, Dict, Optional

from ..tools.tool import ToolResult
from ..core.retry_config import RetryConfig, RetryStrategy

logger = logging.getLogger(__name__)


class ToolRetryConfig:
    """Tool-specific retry configuration.
    
    This is a simplified config for tool execution, wrapping RetryConfig.
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        backoff_factor: float = 2.0,
        max_delay: float = 30.0,
        retriable_errors: Optional[list] = None
    ):
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.backoff_factor = backoff_factor
        self.max_delay = max_delay
        self.retriable_errors = retriable_errors or [
            "timeout",
            "connection",
            "network",
            "temporarily"
        ]
        
        self._core_config = RetryConfig(
            max_step_attempts=max_retries + 1,
            backoff_base=initial_delay,
            backoff_max=max_delay,
            backoff_strategy=RetryStrategy.EXPONENTIAL_BACKOFF
        )
    
    def get_delay(self, attempt: int) -> float:
        """Get delay for given attempt."""
        return self._core_config.get_delay(attempt)


class RetryToolExecutor:
    """Execute tools with automatic retry on failure."""
    
    def __init__(self, tool_service: Any, config: Optional[ToolRetryConfig] = None):
        """Initialize retry executor.
        
        Args:
            tool_service: Tool service
            config: Retry configuration
        """
        self.tool_service = tool_service
        self.config = config or ToolRetryConfig()
    
    async def execute(
        self,
        tool_name: str,
        params: Dict[str, Any],
        on_retry: Optional[Callable] = None
    ) -> ToolResult:
        """Execute tool with retry.
        
        Args:
            tool_name: Tool to execute
            params: Tool parameters
            on_retry: Optional callback on retry
        
        Returns:
            ToolResult
        """
        last_error = None
        delay = self.config.initial_delay
        
        for attempt in range(self.config.max_retries + 1):
            try:
                result = await self.tool_service.execute_tool(tool_name, params)
                
                if result.success:
                    return result
                
                last_error = result.error
                
                if not self._is_retriable(last_error):
                    logger.info(f"[RetryToolExecutor] Non-retriable error: {last_error}")
                    return result
                
                logger.info(f"[RetryToolExecutor] Attempt {attempt + 1} failed: {last_error}")
                
                if attempt < self.config.max_retries:
                    logger.info(f"[RetryToolExecutor] Retrying in {delay}s...")
                    
                    if on_retry:
                        on_retry(attempt, last_error)
                    
                    await asyncio.sleep(delay)
                    delay = min(delay * self.config.backoff_factor, self.config.max_delay)
                    
            except Exception as e:
                last_error = str(e)
                logger.error(f"[RetryToolExecutor] Exception: {e}")
                
                if not self._is_retriable(last_error):
                    return ToolResult(success=False, error=last_error)
        
        return ToolResult(
            success=False,
            error=f"Failed after {self.config.max_retries + 1} attempts: {last_error}"
        )
    
    def _is_retriable(self, error: str) -> bool:
        """Check if error is retriable."""
        if not error:
            return True
        
        error_lower = error.lower()
        return any(retriable in error_lower for retriable in self.config.retriable_errors)


class TimeoutToolExecutor:
    """Execute tools with timeout."""
    
    def __init__(self, tool_service: Any, default_timeout: float = 60.0):
        """Initialize timeout executor.
        
        Args:
            tool_service: Tool service
            default_timeout: Default timeout in seconds
        """
        self.tool_service = tool_service
        self.default_timeout = default_timeout
    
    async def execute(
        self,
        tool_name: str,
        params: Dict[str, Any],
        timeout: Optional[float] = None
    ) -> ToolResult:
        """Execute tool with timeout.
        
        Args:
            tool_name: Tool to execute
            params: Tool parameters
            timeout: Optional timeout override
        
        Returns:
            ToolResult
        """
        timeout_value = timeout or self.default_timeout
        
        try:
            result = await asyncio.wait_for(
                self.tool_service.execute_tool(tool_name, params),
                timeout=timeout_value
            )
            return result
            
        except asyncio.TimeoutError:
            logger.warning(f"[TimeoutToolExecutor] Tool {tool_name} timed out after {timeout_value}s")
            return ToolResult(
                success=False,
                error=f"Tool execution timed out after {timeout_value} seconds"
            )
        except Exception as e:
            logger.error(f"[TimeoutToolExecutor] Error: {e}")
            return ToolResult(success=False, error=str(e))


class RobustToolExecutor:
    """Combined retry + timeout + fallback executor."""
    
    def __init__(
        self,
        tool_service: Any,
        retry_config: Optional[ToolRetryConfig] = None,
        default_timeout: float = 60.0
    ):
        """Initialize robust executor.
        
        Args:
            tool_service: Tool service
            retry_config: Retry configuration
            default_timeout: Default timeout
        """
        self.tool_service = tool_service
        self.retry_executor = RetryToolExecutor(tool_service, retry_config)
        self.timeout_executor = TimeoutToolExecutor(tool_service, default_timeout)
    
    async def execute(
        self,
        tool_name: str,
        params: Dict[str, Any],
        timeout: Optional[float] = None,
        use_retry: bool = True,
        use_timeout: bool = True,
        fallback_result: Optional[Any] = None
    ) -> ToolResult:
        """Execute tool with retry, timeout, and fallback.
        
        Args:
            tool_name: Tool to execute
            params: Tool parameters
            timeout: Optional timeout
            use_retry: Enable retry
            use_timeout: Enable timeout
            fallback_result: Fallback on failure
        
        Returns:
            ToolResult
        """
        async def execute_with_timeout():
            return await self.timeout_executor.execute(tool_name, params, timeout)
        
        if use_retry:
            result = await self.retry_executor.execute(tool_name, params)
            
            if result.success:
                return result
            
            if not use_timeout:
                if fallback_result is not None:
                    return fallback_result
                return result
        
        if use_timeout:
            result = await execute_with_timeout()
            if result.success or fallback_result is None:
                return result
        
        if fallback_result is not None:
            logger.info(f"[RobustToolExecutor] Using fallback for {tool_name}")
            return fallback_result
        
        return ToolResult(
            success=False,
            error=f"Tool {tool_name} failed with retry and timeout"
        )


def create_robust_executor(
    tool_service: Any,
    max_retries: int = 3,
    timeout: float = 60.0
) -> RobustToolExecutor:
    """Create a robust executor.
    
    Args:
        tool_service: Tool service
        max_retries: Max retries
        timeout: Default timeout
    
    Returns:
        RobustToolExecutor
    """
    return RobustToolExecutor(
        tool_service=tool_service,
        retry_config=ToolRetryConfig(max_retries=max_retries),
        default_timeout=timeout
    )
