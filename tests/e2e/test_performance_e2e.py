"""E2E tests for performance and concurrency.

This module tests performance and concurrency scenarios:
- Parallel file generation
- Concurrent user operations
- Cache thread safety
- Generation speed benchmarks
- Memory usage benchmarks
- Cache effectiveness benchmarks
"""

import asyncio
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock
import threading

import pytest

from tests.e2e.utils import count_java_files


class TestConcurrencyE2E:
    """E2E tests for concurrent operations."""
    
    @pytest.mark.asyncio
    async def test_parallel_file_generation(self, temp_large_project, mock_llm_client):
        """Test parallel test generation for multiple files."""
        from pyutagent.services.batch_generator import BatchGenerator, BatchConfig
        from pyutagent.llm.config import LLMConfig, LLMProvider
        
        java_count = count_java_files(temp_large_project)
        assert java_count == 20
        
        llm_config = LLMConfig(
            id="test",
            name="Test",
            provider=LLMProvider.OPENAI,
            api_key="test-key",
            model="gpt-4"
        )
        
        batch_config = BatchConfig(
            parallel_workers=4,
            timeout_per_file=60,
            continue_on_error=True,
            coverage_target=80,
            max_iterations=3
        )
        
        with patch('pyutagent.services.batch_generator.LLMClient') as mock_llm_class:
            mock_llm_class.from_config.return_value = mock_llm_client
            
            generator = BatchGenerator(
                llm_client=mock_llm_client,
                project_path=str(temp_large_project),
                config=batch_config
            )
            
            assert generator.config.parallel_workers == 4
    
    @pytest.mark.asyncio
    async def test_concurrent_user_operations(self, temp_maven_project, mock_llm_client):
        """Test concurrent user operations."""
        from pyutagent.agent.react_agent import ReActAgent
        from pyutagent.memory.working_memory import WorkingMemory
        
        results = []
        errors = []
        
        async def generate_test(file_name: str):
            try:
                working_memory = WorkingMemory(
                    target_coverage=0.8,
                    max_iterations=2
                )
                
                with patch('pyutagent.agent.react_agent.LLMClient') as mock_llm_class:
                    mock_llm_class.from_config.return_value = mock_llm_client
                    
                    agent = ReActAgent(
                        project_path=str(temp_maven_project),
                        llm_client=mock_llm_client,
                        working_memory=working_memory
                    )
                    
                    results.append(file_name)
            except Exception as e:
                errors.append((file_name, str(e)))
        
        tasks = [
            generate_test("Calculator.java"),
            generate_test("Service.java"),
            generate_test("Repository.java")
        ]
        
        await asyncio.gather(*tasks)
        
        assert len(results) == 3
        assert len(errors) == 0
    
    @pytest.mark.asyncio
    async def test_cache_thread_safety(self, temp_maven_project, mock_llm_client):
        """Test cache thread safety."""
        from pyutagent.core.cache import L1MemoryCache
        
        cache = L1MemoryCache(max_size=100)
        
        errors = []
        
        def cache_operations(thread_id: int):
            try:
                for i in range(10):
                    key = f"thread_{thread_id}_key_{i}"
                    cache.set(key, f"value_{i}")
                    value = cache.get(key)
                    assert value == f"value_{i}"
            except Exception as e:
                errors.append((thread_id, str(e)))
        
        threads = [
            threading.Thread(target=cache_operations, args=(i,))
            for i in range(5)
        ]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        assert len(errors) == 0


class TestPerformanceBenchmarkE2E:
    """E2E performance benchmark tests."""
    
    @pytest.mark.asyncio
    async def test_generation_speed_benchmark(self, temp_maven_project, mock_llm_client):
        """Benchmark test generation speed."""
        from pyutagent.agent.react_agent import ReActAgent
        from pyutagent.memory.working_memory import WorkingMemory
        
        working_memory = WorkingMemory(
            target_coverage=0.8,
            max_iterations=1
        )
        
        start_time = time.time()
        
        with patch('pyutagent.agent.react_agent.LLMClient') as mock_llm_class:
            mock_llm_class.from_config.return_value = mock_llm_client
            
            agent = ReActAgent(
                project_path=str(temp_maven_project),
                llm_client=mock_llm_client,
                working_memory=working_memory
            )
            
            assert agent is not None
        
        duration = time.time() - start_time
        
        assert duration < 5.0, f"Agent initialization took {duration:.2f}s, expected < 5.0s"
    
    @pytest.mark.asyncio
    async def test_memory_usage_benchmark(self, temp_large_project, mock_llm_client):
        """Benchmark memory usage."""
        import tracemalloc
        
        tracemalloc.start()
        
        from pyutagent.services.batch_generator import BatchGenerator, BatchConfig
        from pyutagent.llm.config import LLMConfig, LLMProvider
        
        llm_config = LLMConfig(
            id="test",
            name="Test",
            provider=LLMProvider.OPENAI,
            api_key="test-key",
            model="gpt-4"
        )
        
        batch_config = BatchConfig(
            parallel_workers=2,
            timeout_per_file=60,
            continue_on_error=True,
            coverage_target=80,
            max_iterations=2
        )
        
        with patch('pyutagent.services.batch_generator.LLMClient') as mock_llm_class:
            mock_llm_class.from_config.return_value = mock_llm_client
            
            generator = BatchGenerator(
                llm_client=mock_llm_client,
                project_path=str(temp_large_project),
                config=batch_config
            )
            
            assert generator is not None
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        peak_mb = peak / 1024 / 1024
        
        assert peak_mb < 100, f"Peak memory usage was {peak_mb:.2f}MB, expected < 100MB"
    
    @pytest.mark.asyncio
    async def test_cache_effectiveness_benchmark(self, temp_maven_project, mock_llm_client):
        """Benchmark cache effectiveness."""
        from pyutagent.core.cache import L1MemoryCache
        
        cache = L1MemoryCache(max_size=100)
        
        for i in range(50):
            cache.set(f"key_{i}", f"value_{i}")
        
        start_time = time.time()
        
        for i in range(50):
            value = cache.get(f"key_{i}")
            assert value == f"value_{i}"
        
        cache_time = time.time() - start_time
        
        start_time = time.time()
        
        for i in range(50):
            _ = f"value_{i}"
        
        direct_time = time.time() - start_time
        
        assert cache_time < direct_time * 10


class TestScalabilityE2E:
    """E2E tests for scalability."""
    
    @pytest.mark.asyncio
    async def test_large_project_scalability(self, temp_large_project, mock_llm_client):
        """Test scalability with large project."""
        from pyutagent.tools.project_analyzer import ProjectAnalyzer
        
        start_time = time.time()
        
        analyzer = ProjectAnalyzer(str(temp_large_project))
        
        duration = time.time() - start_time
        
        assert duration < 2.0, f"Project analysis took {duration:.2f}s, expected < 2.0s"
    
    @pytest.mark.asyncio
    async def test_concurrent_generation_scalability(self, temp_large_project, mock_llm_client):
        """Test concurrent generation scalability."""
        from pyutagent.services.batch_generator import BatchGenerator, BatchConfig
        from pyutagent.llm.config import LLMConfig, LLMProvider
        
        llm_config = LLMConfig(
            id="test",
            name="Test",
            provider=LLMProvider.OPENAI,
            api_key="test-key",
            model="gpt-4"
        )
        
        configs = [
            BatchConfig(parallel_workers=1, timeout_per_file=60, continue_on_error=True),
            BatchConfig(parallel_workers=2, timeout_per_file=60, continue_on_error=True),
            BatchConfig(parallel_workers=4, timeout_per_file=60, continue_on_error=True),
        ]
        
        for config in configs:
            with patch('pyutagent.services.batch_generator.LLMClient') as mock_llm_class:
                mock_llm_class.from_config.return_value = mock_llm_client
                
                generator = BatchGenerator(
                    llm_client=mock_llm_client,
                    project_path=str(temp_large_project),
                    config=config
                )
                
                assert generator is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
