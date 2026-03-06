"""Test stop functionality for batch generation."""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, MagicMock
from pyutagent.services.batch_generator import BatchGenerator, BatchConfig, FileResult


@pytest.mark.asyncio
async def test_batch_generator_stop():
    """Test that batch generator can be stopped."""
    mock_llm_client = Mock()
    mock_llm_client.model = "test-model"
    mock_llm_client.cancel = Mock()
    mock_llm_client.cancel_current_task = Mock()
    
    config = BatchConfig(
        parallel_workers=1,
        timeout_per_file=10,
        continue_on_error=True
    )
    
    generator = BatchGenerator(
        llm_client=mock_llm_client,
        project_path="/test/path",
        config=config
    )
    
    # Test stop method
    generator.stop()
    assert generator._stop_requested is True
    
    # Test terminate method
    generator._stop_requested = False
    generator.terminate()
    
    assert generator._stop_requested is True
    mock_llm_client.cancel.assert_called_once()
    mock_llm_client.cancel_current_task.assert_called_once()


@pytest.mark.asyncio
async def test_batch_generator_handles_cancellation():
    """Test that batch generator handles asyncio cancellation properly."""
    mock_llm_client = Mock()
    mock_llm_client.model = "test-model"
    mock_llm_client.cancel = Mock()
    
    config = BatchConfig(
        parallel_workers=1,
        timeout_per_file=10,
        continue_on_error=True
    )
    
    generator = BatchGenerator(
        llm_client=mock_llm_client,
        project_path="/test/path",
        config=config
    )
    
    # Mock _generate_single to raise CancelledError
    async def mock_generate(file_path):
        await asyncio.sleep(0.1)
        raise asyncio.CancelledError()
    
    generator._generate_single = mock_generate
    
    # Run generation
    result = await generator.generate_all(["/test/file1.java"])
    
    # Should handle cancellation gracefully
    assert result is not None
    assert len(result.results) == 1
    assert result.results[0].success is False
    assert "cancelled" in result.results[0].error.lower()


@pytest.mark.asyncio
async def test_batch_generator_stop_before_start():
    """Test that batch generator respects stop flag before starting tasks."""
    mock_llm_client = Mock()
    mock_llm_client.model = "test-model"
    
    config = BatchConfig(
        parallel_workers=1,
        timeout_per_file=10,
        continue_on_error=True
    )
    
    generator = BatchGenerator(
        llm_client=mock_llm_client,
        project_path="/test/path",
        config=config
    )
    
    # Mock _generate_single to track if it was called
    generate_called = []
    async def mock_generate(file_path):
        generate_called.append(file_path)
        return FileResult(file_path=file_path, success=True)
    
    generator._generate_single = mock_generate
    
    # Set stop flag before generation
    generator._stop_requested = True
    
    # Run generation
    result = await generator.generate_all(["/test/file1.java", "/test/file2.java"])
    
    # Should stop immediately without calling _generate_single
    assert result is not None
    assert all(r.success is False for r in result.results)
    assert all("stopped" in r.error.lower() for r in result.results)
    assert len(generate_called) == 0  # Should not have called generate


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
