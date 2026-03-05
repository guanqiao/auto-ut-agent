"""Tests for streaming handler component."""

import pytest
import time
import asyncio
from unittest.mock import MagicMock, patch, call

from pyutagent.ui.components.streaming_handler import (
    StreamingConfig,
    StreamingStats,
    StreamingMode
)


class TestStreamingConfig:
    """Tests for StreamingConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = StreamingConfig()
        assert config.mode == StreamingMode.CHUNK
        assert config.char_delay_ms == 20
        assert config.word_delay_ms == 50
        assert config.chunk_delay_ms == 10
        assert config.batch_size == 5
        assert config.max_fps == 60
        assert config.enable_buffering is True
        
    def test_custom_config(self):
        """Test custom configuration."""
        config = StreamingConfig(
            mode=StreamingMode.CHARACTER,
            char_delay_ms=50,
            batch_size=10
        )
        assert config.mode == StreamingMode.CHARACTER
        assert config.char_delay_ms == 50
        assert config.batch_size == 10


class TestStreamingStats:
    """Tests for StreamingStats dataclass."""
    
    def test_default_stats(self):
        """Test default statistics."""
        stats = StreamingStats()
        assert stats.total_chars == 0
        assert stats.total_words == 0
        assert stats.chunks_received == 0
        assert stats.render_updates == 0
        
    def test_duration_ms(self):
        """Test duration calculation."""
        start_time = time.time() - 1  # 1 second ago
        stats = StreamingStats(start_time=start_time)
        duration = stats.duration_ms
        assert duration >= 900  # At least 900ms
        
    def test_chars_per_second(self):
        """Test chars per second calculation."""
        start_time = time.time() - 1  # 1 second ago
        stats = StreamingStats(start_time=start_time, total_chars=100)
        cps = stats.chars_per_second
        assert cps >= 90  # Approximately 100 chars/sec
        
    def test_avg_chunk_size(self):
        """Test average chunk size calculation."""
        stats = StreamingStats(total_chars=100, chunks_received=5)
        assert stats.avg_chunk_size == 20.0
        
    def test_avg_chunk_size_zero_chunks(self):
        """Test avg chunk size with zero chunks."""
        stats = StreamingStats(total_chars=0, chunks_received=0)
        assert stats.avg_chunk_size == 0.0


class TestStreamingModes:
    """Tests for different streaming modes."""
    
    def test_character_mode(self):
        """Test character streaming mode."""
        config = StreamingConfig(mode=StreamingMode.CHARACTER)
        assert config.mode == StreamingMode.CHARACTER
        
    def test_word_mode(self):
        """Test word streaming mode."""
        config = StreamingConfig(mode=StreamingMode.WORD)
        assert config.mode == StreamingMode.WORD
        
    def test_chunk_mode(self):
        """Test chunk streaming mode."""
        config = StreamingConfig(mode=StreamingMode.CHUNK)
        assert config.mode == StreamingMode.CHUNK
        
    def test_instant_mode(self):
        """Test instant streaming mode."""
        config = StreamingConfig(mode=StreamingMode.INSTANT)
        assert config.mode == StreamingMode.INSTANT
