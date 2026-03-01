"""Multi-provider LLM client with automatic fallback.

This module provides:
- LLMProviderManager: Manages multiple LLM providers
- Automatic fallback when primary provider fails
- Provider health checking
- Request routing based on provider capabilities
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from enum import Enum, auto
from datetime import datetime, timedelta

from ..core.config import LLMConfig, LLMProvider
from .client import LLMClient

logger = logging.getLogger(__name__)


class ProviderStatus(Enum):
    """Status of a provider."""
    HEALTHY = auto()
    DEGRADED = auto()
    UNHEALTHY = auto()
    UNKNOWN = auto()


@dataclass
class ProviderHealth:
    """Health status of a provider."""
    status: ProviderStatus
    last_check: datetime
    success_count: int = 0
    failure_count: int = 0
    avg_latency_ms: float = 0.0
    error_message: Optional[str] = None

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0


@dataclass
class ProviderConfig:
    """Configuration for a provider in the fallback chain."""
    config: LLMConfig
    priority: int = 0
    max_retries: int = 3
    timeout_multiplier: float = 1.0
    enabled: bool = True


class LLMProviderManager:
    """Manages multiple LLM providers with automatic fallback.

    Features:
    - Multiple provider configuration
    - Automatic fallback on failure
    - Health monitoring and circuit breaker
    - Request routing
    - Performance tracking
    """

    def __init__(
        self,
        providers: Optional[List[ProviderConfig]] = None,
        health_check_interval: int = 60,
        failure_threshold: int = 3,
        recovery_threshold: int = 2
    ):
        """Initialize provider manager.

        Args:
            providers: List of provider configurations
            health_check_interval: Interval for health checks in seconds
            failure_threshold: Failures before marking provider unhealthy
            recovery_threshold: Successes before marking provider healthy
        """
        self._providers: List[ProviderConfig] = providers or []
        self._clients: Dict[str, LLMClient] = {}
        self._health: Dict[str, ProviderHealth] = {}
        self._health_check_interval = health_check_interval
        self._failure_threshold = failure_threshold
        self._recovery_threshold = recovery_threshold

        for pc in self._providers:
            self._initialize_provider(pc)

        logger.info(f"[LLMProviderManager] Initialized with {len(self._providers)} providers")

    def _initialize_provider(self, pc: ProviderConfig):
        """Initialize a provider client."""
        provider_id = pc.config.id
        try:
            client = LLMClient.from_config(pc.config)
            self._clients[provider_id] = client
            self._health[provider_id] = ProviderHealth(
                status=ProviderStatus.UNKNOWN,
                last_check=datetime.now()
            )
            logger.info(f"[LLMProviderManager] Initialized provider: {provider_id}")
        except Exception as e:
            logger.error(f"[LLMProviderManager] Failed to initialize provider {provider_id}: {e}")
            self._health[provider_id] = ProviderHealth(
                status=ProviderStatus.UNHEALTHY,
                last_check=datetime.now(),
                error_message=str(e)
            )

    def add_provider(self, config: LLMConfig, priority: int = 0):
        """Add a provider to the fallback chain.

        Args:
            config: LLM configuration
            priority: Provider priority (lower = higher priority)
        """
        pc = ProviderConfig(config=config, priority=priority)
        self._providers.append(pc)
        self._initialize_provider(pc)
        self._providers.sort(key=lambda x: x.priority)
        logger.info(f"[LLMProviderManager] Added provider: {config.id}, priority: {priority}")

    def remove_provider(self, provider_id: str):
        """Remove a provider from the fallback chain.

        Args:
            provider_id: Provider ID to remove
        """
        self._providers = [p for p in self._providers if p.config.id != provider_id]
        self._clients.pop(provider_id, None)
        self._health.pop(provider_id, None)
        logger.info(f"[LLMProviderManager] Removed provider: {provider_id}")

    def get_provider(self, provider_id: str) -> Optional[LLMClient]:
        """Get a provider client by ID.

        Args:
            provider_id: Provider ID

        Returns:
            LLMClient or None
        """
        return self._clients.get(provider_id)

    def get_healthy_providers(self) -> List[ProviderConfig]:
        """Get list of healthy providers sorted by priority.

        Returns:
            List of healthy ProviderConfig
        """
        healthy = []
        for pc in self._providers:
            if not pc.enabled:
                continue

            health = self._health.get(pc.config.id)
            if health and health.status in [ProviderStatus.HEALTHY, ProviderStatus.DEGRADED]:
                healthy.append(pc)

        return healthy

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        preferred_provider: Optional[str] = None
    ) -> tuple[str, str]:
        """Generate text with automatic fallback.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            preferred_provider: Preferred provider ID

        Returns:
            Tuple of (response, provider_id)
        """
        tried_providers = []

        if preferred_provider:
            pc = self._find_provider_config(preferred_provider)
            if pc and pc.enabled:
                result = await self._try_provider(pc, prompt, system_prompt)
                if result:
                    return result

        for pc in self._providers:
            if not pc.enabled:
                continue

            if pc.config.id in tried_providers:
                continue

            tried_providers.append(pc.config.id)

            result = await self._try_provider(pc, prompt, system_prompt)
            if result:
                return result

        raise Exception(f"All providers failed. Tried: {tried_providers}")

    async def _try_provider(
        self,
        pc: ProviderConfig,
        prompt: str,
        system_prompt: Optional[str]
    ) -> Optional[tuple[str, str]]:
        """Try generating with a specific provider.

        Args:
            pc: Provider configuration
            prompt: User prompt
            system_prompt: Optional system prompt

        Returns:
            Tuple of (response, provider_id) or None if failed
        """
        provider_id = pc.config.id
        client = self._clients.get(provider_id)

        if not client:
            return None

        logger.info(f"[LLMProviderManager] Trying provider: {provider_id}")

        for attempt in range(pc.max_retries):
            try:
                start_time = datetime.now()
                response = await client.agenerate(prompt, system_prompt)
                latency = (datetime.now() - start_time).total_seconds() * 1000

                self._record_success(provider_id, latency)
                logger.info(f"[LLMProviderManager] Provider {provider_id} succeeded")

                return response, provider_id

            except Exception as e:
                logger.warning(f"[LLMProviderManager] Provider {provider_id} attempt {attempt + 1} failed: {e}")
                self._record_failure(provider_id, str(e))

                if attempt < pc.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)

        return None

    def _record_success(self, provider_id: str, latency_ms: float):
        """Record a successful request."""
        health = self._health.get(provider_id)
        if not health:
            return

        health.success_count += 1

        if health.avg_latency_ms > 0:
            health.avg_latency_ms = (health.avg_latency_ms + latency_ms) / 2
        else:
            health.avg_latency_ms = latency_ms

        if health.failure_count > 0:
            if health.success_count >= self._recovery_threshold:
                health.status = ProviderStatus.HEALTHY
                health.failure_count = 0
        else:
            health.status = ProviderStatus.HEALTHY

        health.last_check = datetime.now()

    def _record_failure(self, provider_id: str, error: str):
        """Record a failed request."""
        health = self._health.get(provider_id)
        if not health:
            return

        health.failure_count += 1
        health.error_message = error
        health.last_check = datetime.now()

        if health.failure_count >= self._failure_threshold:
            health.status = ProviderStatus.UNHEALTHY
            logger.warning(f"[LLMProviderManager] Provider {provider_id} marked as UNHEALTHY")

        elif health.status == ProviderStatus.HEALTHY:
            health.status = ProviderStatus.DEGRADED

    def _find_provider_config(self, provider_id: str) -> Optional[ProviderConfig]:
        """Find provider configuration by ID."""
        for pc in self._providers:
            if pc.config.id == provider_id:
                return pc
        return None

    async def check_health(self, provider_id: Optional[str] = None):
        """Check health of providers.

        Args:
            provider_id: Optional specific provider to check
        """
        providers_to_check = (
            [self._find_provider_config(provider_id)] if provider_id
            else self._providers
        )

        for pc in providers_to_check:
            if not pc:
                continue

            provider_id = pc.config.id
            client = self._clients.get(provider_id)

            if not client:
                continue

            try:
                start_time = datetime.now()
                await asyncio.wait_for(
                    client.agenerate("ping", "You are a ping service. Respond with 'pong'."),
                    timeout=10
                )
                latency = (datetime.now() - start_time).total_seconds() * 1000

                health = self._health.get(provider_id)
                if health:
                    health.status = ProviderStatus.HEALTHY
                    health.last_check = datetime.now()
                    health.avg_latency_ms = latency

                logger.info(f"[LLMProviderManager] Provider {provider_id} health check: OK ({latency:.0f}ms)")

            except Exception as e:
                health = self._health.get(provider_id)
                if health:
                    health.status = ProviderStatus.UNHEALTHY
                    health.last_check = datetime.now()
                    health.error_message = str(e)

                logger.warning(f"[LLMProviderManager] Provider {provider_id} health check: FAILED - {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get provider statistics.

        Returns:
            Dictionary with provider stats
        """
        stats = {
            "total_providers": len(self._providers),
            "healthy_providers": len(self.get_healthy_providers()),
            "providers": []
        }

        for pc in self._providers:
            health = self._health.get(pc.config.id)
            stats["providers"].append({
                "id": pc.config.id,
                "name": pc.config.name,
                "provider": pc.config.provider,
                "priority": pc.priority,
                "enabled": pc.enabled,
                "status": health.status.name if health else "UNKNOWN",
                "success_rate": health.success_rate if health else 0,
                "avg_latency_ms": health.avg_latency_ms if health else 0,
                "error": health.error_message if health else None
            })

        return stats

    def enable_provider(self, provider_id: str):
        """Enable a provider."""
        pc = self._find_provider_config(provider_id)
        if pc:
            pc.enabled = True
            logger.info(f"[LLMProviderManager] Enabled provider: {provider_id}")

    def disable_provider(self, provider_id: str):
        """Disable a provider."""
        pc = self._find_provider_config(provider_id)
        if pc:
            pc.enabled = False
            logger.info(f"[LLMProviderManager] Disabled provider: {provider_id}")

    def reset_health(self, provider_id: Optional[str] = None):
        """Reset health status for providers.

        Args:
            provider_id: Optional specific provider to reset
        """
        if provider_id:
            health = self._health.get(provider_id)
            if health:
                health.status = ProviderStatus.UNKNOWN
                health.success_count = 0
                health.failure_count = 0
        else:
            for health in self._health.values():
                health.status = ProviderStatus.UNKNOWN
                health.success_count = 0
                health.failure_count = 0


def create_provider_manager(
    configs: List[LLMConfig],
    default_priority: bool = True
) -> LLMProviderManager:
    """Create a provider manager from configurations.

    Args:
        configs: List of LLM configurations
        default_priority: Use config order as priority (first = highest)

    Returns:
        Configured LLMProviderManager
    """
    providers = []

    for i, config in enumerate(configs):
        priority = i if default_priority else 0
        providers.append(ProviderConfig(
            config=config,
            priority=priority
        ))

    return LLMProviderManager(providers=providers)
