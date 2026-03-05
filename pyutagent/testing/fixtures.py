"""Testing Framework and Utilities.

This module provides:
- Test fixtures
- Mock helpers
- Assertion helpers
- Performance benchmarking
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, TypeVar

T = TypeVar("T")


@dataclass
class TestFixture:
    """A test fixture."""
    name: str
    setup: Callable
    teardown: Optional[Callable] = None


@dataclass
class BenchmarkResult:
    """Benchmark result."""
    name: str
    iterations: int
    total_time_ms: float
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    operations_per_second: float


class MockLLMClient:
    """Mock LLM client for testing."""

    def __init__(
        self,
        responses: Optional[Dict[str, str]] = None,
        delay_ms: int = 0
    ):
        self.responses = responses or {}
        self.delay_ms = delay_ms
        self.calls: List[Dict[str, Any]] = []

    async def agenerate(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        self.calls.append({
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "timestamp": datetime.now()
        })

        if self.delay_ms > 0:
            await asyncio.sleep(self.delay_ms / 1000)

        for key, response in self.responses.items():
            if key.lower() in prompt.lower():
                return response

        return "Mock response"

    def set_response(self, pattern: str, response: str):
        self.responses[pattern] = response

    def get_call_count(self) -> int:
        return len(self.calls)


class MockTool:
    """Mock tool for testing."""

    def __init__(
        self,
        name: str = "mock_tool",
        success: bool = True,
        output: Any = None,
        error: Optional[str] = None,
        delay_ms: int = 0
    ):
        self.name = name
        self.success = success
        self.output = output
        self.error = error
        self.delay_ms = delay_ms
        self.calls: List[Dict[str, Any]] = []

    async def execute(self, **kwargs) -> Any:
        self.calls.append({
            "parameters": kwargs,
            "timestamp": datetime.now()
        })

        if self.delay_ms > 0:
            await asyncio.sleep(self.delay_ms / 1000)

        class ToolResult:
            def __init__(s, success, output, error):
                s.success = success
                s.output = output
                s.error = error

        return ToolResult(self.success, self.output, self.error)

    def get_call_count(self) -> int:
        return len(self.calls)


class MockMemory:
    """Mock memory for testing."""

    def __init__(self):
        self._storage: Dict[str, Any] = {}
        self.calls: List[str] = []

    async def store(self, key: str, value: Any):
        self.calls.append(f"store:{key}")
        self._storage[key] = value

    async def retrieve(self, key: str, default: Any = None) -> Any:
        self.calls.append(f"retrieve:{key}")
        return self._storage.get(key, default)

    async def delete(self, key: str):
        self.calls.append(f"delete:{key}")
        self._storage.pop(key, None)

    async def clear(self):
        self.calls.append("clear")
        self._storage.clear()

    def get_call_history(self) -> List[str]:
        return self.calls


class Benchmark:
    """Performance benchmarking utility."""

    def __init__(self, name: str = "benchmark"):
        self.name = name
        self.results: List[BenchmarkResult] = []

    async def run(
        self,
        func: Callable,
        iterations: int = 100,
        **kwargs
    ) -> BenchmarkResult:
        times = []

        for _ in range(iterations):
            start = time.perf_counter()
            if asyncio.iscoroutinefunction(func):
                await func(**kwargs)
            else:
                func(**kwargs)
            end = time.perf_counter()
            times.append((end - start) * 1000)

        result = BenchmarkResult(
            name=self.name,
            iterations=iterations,
            total_time_ms=sum(times),
            avg_time_ms=sum(times) / len(times),
            min_time_ms=min(times),
            max_time_ms=max(times),
            operations_per_second=1000 / (sum(times) / len(times))
        )

        self.results.append(result)
        return result

    def get_results(self) -> List[BenchmarkResult]:
        return self.results

    def print_summary(self):
        print(f"\n=== {self.name} ===")
        for result in self.results:
            print(f"{result.name}:")
            print(f"  Iterations: {result.iterations}")
            print(f"  Avg: {result.avg_time_ms:.2f}ms")
            print(f"  Min: {result.min_time_ms:.2f}ms")
            print(f"  Max: {result.max_time_ms:.2f}ms")
            print(f"  Ops/sec: {result.operations_per_second:.2f}")


class AssertHelper:
    """Assertion helpers for tests."""

    @staticmethod
    def assert_success(result: Any):
        assert hasattr(result, "success"), "Result has no 'success' attribute"
        assert result.success is True, f"Expected success, got: {result}"

    @staticmethod
    def assert_failure(result: Any):
        assert hasattr(result, "success"), "Result has no 'success' attribute"
        assert result.success is False, f"Expected failure, got: {result}"

    @staticmethod
    def assert_has_attribute(obj: Any, attr: str):
        assert hasattr(obj, attr), f"Object missing attribute: {attr}"

    @staticmethod
    def assert_type(obj: Any, expected_type: type):
        assert isinstance(obj, expected_type), f"Expected {expected_type}, got {type(obj)}"

    @staticmethod
    def assert_contains(container: Any, item: Any):
        assert item in container, f"Container does not contain item: {item}"

    @staticmethod
    def assert_not_empty(container: Any):
        assert len(container) > 0, "Container is empty"


def create_mock_llm(responses=None, delay_ms=0):
    return MockLLMClient(responses, delay_ms)


def create_mock_tool(name="mock_tool", success=True, output=None):
    return MockTool(name, success, output)


def create_mock_memory():
    return MockMemory()


def create_sample_context():
    return {
        "task": "test task",
        "project": "test_project",
        "language": "python",
        "max_iterations": 10
    }


def create_sample_config():
    return {
        "llm_model": "claude-3-sonnet-20240229",
        "max_tokens": 4096,
        "temperature": 0.7,
        "timeout": 300
    }
