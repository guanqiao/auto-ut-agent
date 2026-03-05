"""Tests for ToolMemory.

This module tests:
- Recording successful and failed tool calls
- Retrieving recommended tools
- Tool statistics tracking
- Persistence functionality
"""

import asyncio
import json
import pytest
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

from pyutagent.memory.tool_memory import (
    ToolMemory,
    ToolCallRecord,
    ToolRecommendation,
    create_tool_memory,
)


@pytest.fixture
def temp_storage():
    """Create a temporary storage file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        path = f.name
    yield path
    # Cleanup
    Path(path).unlink(missing_ok=True)


@pytest.fixture
def tool_memory(temp_storage):
    """Create a ToolMemory instance with temporary storage."""
    return ToolMemory(storage_path=temp_storage)


class TestToolMemoryInitialization:
    """Tests for ToolMemory initialization."""
    
    def test_init_without_storage(self):
        """Test initialization without storage path."""
        memory = ToolMemory()
        assert memory.storage_path is None
        assert len(memory._records) == 0
    
    def test_init_with_storage(self, temp_storage):
        """Test initialization with storage path."""
        memory = ToolMemory(storage_path=temp_storage)
        assert memory.storage_path == Path(temp_storage)
    
    def test_init_loads_existing_data(self, temp_storage):
        """Test loading existing data on initialization."""
        # Create some data first
        memory = ToolMemory(storage_path=temp_storage)
        asyncio.run(memory.record_success(
            tool_name="test_tool",
            params={"param1": "value1"},
            context={"ctx": "data"},
            result="success",
            task_type="test"
        ))
        
        # Create new instance with same storage
        memory2 = ToolMemory(storage_path=temp_storage)
        assert len(memory2._records) == 1
        assert memory2._records[0].tool_name == "test_tool"


class TestRecordSuccess:
    """Tests for recording successful tool calls."""
    
    @pytest.mark.asyncio
    async def test_record_success_basic(self, tool_memory):
        """Test basic success recording."""
        await tool_memory.record_success(
            tool_name="parse_code",
            params={"file_path": "test.java"},
            context={"goal": "test"},
            result={"class_info": {}},
            task_type="code_analysis"
        )
        
        assert len(tool_memory._records) == 1
        record = tool_memory._records[0]
        assert record.tool_name == "parse_code"
        assert record.success is True
        assert record.task_type == "code_analysis"
    
    @pytest.mark.asyncio
    async def test_record_success_updates_stats(self, tool_memory):
        """Test that success recording updates statistics."""
        await tool_memory.record_success(
            tool_name="test_tool",
            params={},
            context={},
            result="ok"
        )
        
        stats = tool_memory.get_tool_stats("test_tool")
        assert stats is not None
        assert stats["total_calls"] == 1
        assert stats["successes"] == 1
        assert stats["success_rate"] == 1.0
    
    @pytest.mark.asyncio
    async def test_record_success_updates_task_type_tools(self, tool_memory):
        """Test that success recording updates task type mappings."""
        await tool_memory.record_success(
            tool_name="parse_code",
            params={},
            context={},
            result="ok",
            task_type="code_analysis"
        )
        
        assert "code_analysis" in tool_memory._task_type_tools
        assert "parse_code" in tool_memory._task_type_tools["code_analysis"]
        assert tool_memory._task_type_tools["code_analysis"]["parse_code"] == 1
    
    @pytest.mark.asyncio
    async def test_record_success_truncates_result(self, tool_memory):
        """Test that long results are truncated."""
        long_result = "x" * 1000
        
        await tool_memory.record_success(
            tool_name="test_tool",
            params={},
            context={},
            result=long_result
        )
        
        record = tool_memory._records[0]
        assert len(record.result) <= 500


class TestRecordFailure:
    """Tests for recording failed tool calls."""
    
    @pytest.mark.asyncio
    async def test_record_failure_basic(self, tool_memory):
        """Test basic failure recording."""
        await tool_memory.record_failure(
            tool_name="parse_code",
            params={"file_path": "test.java"},
            context={"goal": "test"},
            error="File not found",
            task_type="code_analysis"
        )
        
        assert len(tool_memory._records) == 1
        record = tool_memory._records[0]
        assert record.tool_name == "parse_code"
        assert record.success is False
        assert record.error == "File not found"
    
    @pytest.mark.asyncio
    async def test_record_failure_updates_stats(self, tool_memory):
        """Test that failure recording updates statistics."""
        await tool_memory.record_failure(
            tool_name="test_tool",
            params={},
            context={},
            error="error"
        )
        
        stats = tool_memory.get_tool_stats("test_tool")
        assert stats["total_calls"] == 1
        assert stats["failures"] == 1
        assert stats["success_rate"] == 0.0
    
    @pytest.mark.asyncio
    async def test_record_failure_truncates_error(self, tool_memory):
        """Test that long errors are truncated."""
        long_error = "x" * 1000
        
        await tool_memory.record_failure(
            tool_name="test_tool",
            params={},
            context={},
            error=long_error
        )
        
        record = tool_memory._records[0]
        assert len(record.error) <= 500


class TestGetRecommendedTools:
    """Tests for getting tool recommendations."""
    
    @pytest.mark.asyncio
    async def test_get_recommended_tools_empty(self, tool_memory):
        """Test getting recommendations for unknown task type."""
        recommendations = await tool_memory.get_recommended_tools("unknown_type")
        assert recommendations == []
    
    @pytest.mark.asyncio
    async def test_get_recommended_tools_basic(self, tool_memory):
        """Test basic recommendations."""
        # Record some successes
        await tool_memory.record_success(
            tool_name="tool_a",
            params={},
            context={},
            result="ok",
            task_type="test_generation"
        )
        await tool_memory.record_success(
            tool_name="tool_b",
            params={},
            context={},
            result="ok",
            task_type="test_generation"
        )
        
        recommendations = await tool_memory.get_recommended_tools("test_generation")
        
        assert len(recommendations) == 2
        assert all(isinstance(r, ToolRecommendation) for r in recommendations)
    
    @pytest.mark.asyncio
    async def test_get_recommended_tools_sorted(self, tool_memory):
        """Test that recommendations are sorted by success rate."""
        # Record multiple uses
        for _ in range(5):
            await tool_memory.record_success(
                tool_name="reliable_tool",
                params={},
                context={},
                result="ok",
                task_type="test"
            )
        
        await tool_memory.record_success(
            tool_name="reliable_tool",
            params={},
            context={},
            result="ok",
            task_type="test"
        )
        await tool_memory.record_failure(
            tool_name="unreliable_tool",
            params={},
            context={},
            error="fail",
            task_type="test"
        )
        
        recommendations = await tool_memory.get_recommended_tools("test")
        
        # reliable_tool should be first (higher success rate * usage)
        assert recommendations[0].tool_name == "reliable_tool"
    
    @pytest.mark.asyncio
    async def test_get_recommended_tools_limit(self, tool_memory):
        """Test that limit parameter works."""
        for i in range(10):
            await tool_memory.record_success(
                tool_name=f"tool_{i}",
                params={},
                context={},
                result="ok",
                task_type="test"
            )
        
        recommendations = await tool_memory.get_recommended_tools("test", limit=3)
        
        assert len(recommendations) == 3


class TestGetToolStats:
    """Tests for getting tool statistics."""
    
    def test_get_tool_stats_nonexistent(self, tool_memory):
        """Test getting stats for non-existent tool."""
        stats = tool_memory.get_tool_stats("nonexistent")
        assert stats is None
    
    @pytest.mark.asyncio
    async def test_get_tool_stats_existing(self, tool_memory):
        """Test getting stats for existing tool."""
        await tool_memory.record_success(
            tool_name="test_tool",
            params={},
            context={},
            result="ok"
        )
        
        stats = tool_memory.get_tool_stats("test_tool")
        
        assert stats is not None
        assert "total_calls" in stats
        assert "successes" in stats
        assert "failures" in stats
        assert "success_rate" in stats
    
    def test_get_all_stats_empty(self, tool_memory):
        """Test getting all stats when empty."""
        stats = tool_memory.get_all_stats()
        assert stats == {}
    
    @pytest.mark.asyncio
    async def test_get_all_stats(self, tool_memory):
        """Test getting all stats."""
        await tool_memory.record_success(
            tool_name="tool_a",
            params={},
            context={},
            result="ok"
        )
        await tool_memory.record_success(
            tool_name="tool_b",
            params={},
            context={},
            result="ok"
        )
        
        stats = tool_memory.get_all_stats()
        
        assert "tool_a" in stats
        assert "tool_b" in stats


class TestGetMostSuccessfulTools:
    """Tests for getting most successful tools."""
    
    def test_get_most_successful_empty(self, tool_memory):
        """Test with no records."""
        result = tool_memory.get_most_successful_tools()
        assert result == []
    
    @pytest.mark.asyncio
    async def test_get_most_successful_basic(self, tool_memory):
        """Test basic functionality."""
        # Create tools with different success rates
        for _ in range(5):
            await tool_memory.record_success(
                tool_name="perfect_tool",
                params={},
                context={},
                result="ok"
            )
        
        for _ in range(3):
            await tool_memory.record_failure(
                tool_name="bad_tool",
                params={},
                context={},
                error="fail"
            )
        
        result = tool_memory.get_most_successful_tools(min_calls=1)
        
        assert len(result) == 2
        assert result[0]["tool_name"] == "perfect_tool"
        assert result[0]["success_rate"] == 1.0
    
    @pytest.mark.asyncio
    async def test_get_most_successful_min_calls(self, tool_memory):
        """Test min_calls parameter."""
        await tool_memory.record_success(
            tool_name="rare_tool",
            params={},
            context={},
            result="ok"
        )
        
        for _ in range(5):
            await tool_memory.record_success(
                tool_name="common_tool",
                params={},
                context={},
                result="ok"
            )
        
        result = tool_memory.get_most_successful_tools(min_calls=3)
        
        assert len(result) == 1
        assert result[0]["tool_name"] == "common_tool"


class TestGetLeastReliableTools:
    """Tests for getting least reliable tools."""
    
    @pytest.mark.asyncio
    async def test_get_least_reliable_basic(self, tool_memory):
        """Test basic functionality."""
        for _ in range(5):
            await tool_memory.record_failure(
                tool_name="bad_tool",
                params={},
                context={},
                error="fail"
            )
        
        for _ in range(5):
            await tool_memory.record_success(
                tool_name="good_tool",
                params={},
                context={},
                result="ok"
            )
        
        result = tool_memory.get_least_reliable_tools(min_calls=1)
        
        assert len(result) == 2
        assert result[0]["tool_name"] == "bad_tool"
        assert result[0]["success_rate"] == 0.0


class TestGetRecentRecords:
    """Tests for getting recent records."""
    
    def test_get_recent_empty(self, tool_memory):
        """Test with no records."""
        records = tool_memory.get_recent_records()
        assert records == []
    
    @pytest.mark.asyncio
    async def test_get_recent_basic(self, tool_memory):
        """Test basic functionality."""
        await tool_memory.record_success(
            tool_name="tool_a",
            params={},
            context={},
            result="ok"
        )
        await tool_memory.record_failure(
            tool_name="tool_b",
            params={},
            context={},
            error="fail"
        )
        
        records = tool_memory.get_recent_records()
        
        assert len(records) == 2
        assert records[0]["tool_name"] == "tool_b"  # Most recent first
    
    @pytest.mark.asyncio
    async def test_get_recent_limit(self, tool_memory):
        """Test limit parameter."""
        for i in range(10):
            await tool_memory.record_success(
                tool_name=f"tool_{i}",
                params={},
                context={},
                result="ok"
            )
        
        records = tool_memory.get_recent_records(limit=3)
        
        assert len(records) == 3


class TestClearHistory:
    """Tests for clearing history."""
    
    @pytest.mark.asyncio
    async def test_clear_history(self, tool_memory):
        """Test clearing all history."""
        await tool_memory.record_success(
            tool_name="test_tool",
            params={},
            context={},
            result="ok"
        )
        
        tool_memory.clear_history()
        
        assert len(tool_memory._records) == 0
        assert len(tool_memory._task_type_tools) == 0
        assert len(tool_memory._tool_stats) == 0


class TestPersistence:
    """Tests for persistence functionality."""
    
    @pytest.mark.asyncio
    async def test_save_and_load(self, temp_storage):
        """Test saving and loading data."""
        # Create and populate memory
        memory1 = ToolMemory(storage_path=temp_storage)
        await memory1.record_success(
            tool_name="test_tool",
            params={"key": "value"},
            context={"ctx": "data"},
            result="success",
            task_type="test_type"
        )
        
        # Create new instance - should load from file
        memory2 = ToolMemory(storage_path=temp_storage)
        
        assert len(memory2._records) == 1
        assert memory2._records[0].tool_name == "test_tool"
        assert memory2._records[0].task_type == "test_type"
        assert "test_type" in memory2._task_type_tools
    
    @pytest.mark.asyncio
    async def test_save_limits_records(self, temp_storage):
        """Test that save limits to last 1000 records."""
        memory = ToolMemory(storage_path=temp_storage)
        
        # Create more than 1000 records
        for i in range(1100):
            await memory.record_success(
                tool_name="test_tool",
                params={},
                context={},
                result=f"result_{i}"
            )
        
        # Load in new instance
        memory2 = ToolMemory(storage_path=temp_storage)
        
        # Should only have last 1000
        assert len(memory2._records) == 1000


class TestCreateToolMemory:
    """Tests for create_tool_memory factory function."""
    
    def test_create_tool_memory_without_path(self):
        """Test factory without storage path."""
        memory = create_tool_memory()
        assert isinstance(memory, ToolMemory)
        assert memory.storage_path is None
    
    def test_create_tool_memory_with_path(self, temp_storage):
        """Test factory with storage path."""
        memory = create_tool_memory(storage_path=temp_storage)
        assert isinstance(memory, ToolMemory)
        assert memory.storage_path == Path(temp_storage)


class TestEdgeCases:
    """Tests for edge cases."""
    
    @pytest.mark.asyncio
    async def test_record_with_none_result(self, tool_memory):
        """Test recording with None result."""
        await tool_memory.record_success(
            tool_name="test_tool",
            params={},
            context={},
            result=None
        )
        
        assert len(tool_memory._records) == 1
        assert tool_memory._records[0].result == ""
    
    @pytest.mark.asyncio
    async def test_record_with_complex_params(self, tool_memory):
        """Test recording with complex parameter types."""
        complex_params = {
            "list": [1, 2, 3],
            "dict": {"nested": "value"},
            "number": 42
        }
        
        await tool_memory.record_success(
            tool_name="test_tool",
            params=complex_params,
            context={},
            result="ok"
        )
        
        assert tool_memory._records[0].parameters == complex_params
    
    def test_load_corrupted_file(self, temp_storage):
        """Test handling of corrupted storage file."""
        # Write invalid JSON
        with open(temp_storage, 'w') as f:
            f.write("not valid json")
        
        # Should not crash
        memory = ToolMemory(storage_path=temp_storage)
        assert len(memory._records) == 0
