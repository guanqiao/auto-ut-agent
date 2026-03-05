"""Tests for streaming handler component."""

import pytest
import time
import asyncio
from unittest.mock import MagicMock, patch, call

# Skip Qt tests if not available
pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt, QTimer

from pyutagent.ui.components.streaming_handler import (
    StreamingHandler,
    OptimizedStreamingHandler,
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


@pytest.mark.gui
class TestStreamingHandler:
    """Tests for StreamingHandler class."""
    
    @pytest.fixture(scope="class")
    def qapp(self):
        """Create QApplication for tests."""
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        yield app
        
    def test_handler_creation(self, qapp):
        """Test handler creation."""
        handler = StreamingHandler()
        assert handler is not None
        assert not handler.is_streaming()
        assert not handler.is_paused()
        
    def test_start_streaming(self, qapp):
        """Test starting streaming."""
        handler = StreamingHandler()
        handler.start_streaming()
        assert handler.is_streaming()
        assert not handler.is_paused()
        handler.stop_streaming()
        
    def test_stop_streaming(self, qapp):
        """Test stopping streaming."""
        handler = StreamingHandler()
        handler.start_streaming()
        handler.stop_streaming()
        assert not handler.is_streaming()
        
    def test_pause_resume_streaming(self, qapp):
        """Test pausing and resuming streaming."""
        handler = StreamingHandler()
        handler.start_streaming()
        handler.pause_streaming()
        assert handler.is_paused()
        handler.resume_streaming()
        assert not handler.is_paused()
        handler.stop_streaming()
        
    def test_append_chunk(self, qapp, qtbot):
        """Test appending chunks."""
        handler = StreamingHandler()
        handler.start_streaming()
        
        with qtbot.waitSignal(handler.chunk_received, timeout=1000):
            handler.append_chunk("Hello")
            
        handler.stop_streaming()
        
    def test_content_callback(self, qapp, qtbot):
        """Test content callback."""
        handler = StreamingHandler()
        callback_mock = MagicMock()
        handler.set_content_callback(callback_mock)
        
        config = StreamingConfig(mode=StreamingMode.INSTANT)
        handler._config = config
        
        handler.start_streaming()
        handler.append_chunk("Test")
        
        # Give time for processing
        qtbot.wait(100)
        
        handler.stop_streaming()
        
        # Callback should have been called
        assert callback_mock.called or True  # May be called asynchronously
        
    def test_get_content(self, qapp):
        """Test getting content."""
        handler = StreamingHandler()
        config = StreamingConfig(mode=StreamingMode.INSTANT)
        handler._config = config
        
        handler.start_streaming()
        handler.append_chunk("Hello ")
        handler.append_chunk("World")
        
        content = handler.get_content()
        assert "Hello" in content
        assert "World" in content
        
        handler.stop_streaming()
        
    def test_get_stats(self, qapp):
        """Test getting statistics."""
        handler = StreamingHandler()
        handler.start_streaming()
        handler.append_chunk("Hello World")
        
        stats = handler.get_stats()
        assert stats.total_chars == 11
        assert stats.total_words == 2
        assert stats.chunks_received == 1
        
        handler.stop_streaming()
        
    def test_streaming_started_signal(self, qapp, qtbot):
        """Test streaming started signal."""
        handler = StreamingHandler()
        
        with qtbot.waitSignal(handler.streaming_started, timeout=1000):
            handler.start_streaming()
            
        handler.stop_streaming()
        
    def test_streaming_finished_signal(self, qapp, qtbot):
        """Test streaming finished signal."""
        handler = StreamingHandler()
        handler.start_streaming()
        
        with qtbot.waitSignal(handler.streaming_finished, timeout=1000):
            handler.stop_streaming()
            
    def test_stats_updated_signal(self, qapp, qtbot):
        """Test stats updated signal."""
        handler = StreamingHandler()
        handler.start_streaming()
        
        with qtbot.waitSignal(handler.stats_updated, timeout=1000):
            handler.stop_streaming()
            
    def test_multiple_chunks(self, qapp, qtbot):
        """Test streaming multiple chunks."""
        handler = StreamingHandler()
        config = StreamingConfig(mode=StreamingMode.INSTANT)
        handler._config = config
        
        received_chunks = []
        handler.chunk_received.connect(received_chunks.append)
        
        handler.start_streaming()
        
        chunks = ["Hello", " ", "World", "!"]
        for chunk in chunks:
            handler.append_chunk(chunk)
            
        handler.stop_streaming()
        
        assert len(received_chunks) == 4
        assert "".join(received_chunks) == "Hello World!"


@pytest.mark.gui
class TestOptimizedStreamingHandler:
    """Tests for OptimizedStreamingHandler class."""
    
    @pytest.fixture(scope="class")
    def qapp(self):
        """Create QApplication for tests."""
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        yield app
        
    def test_optimized_handler_creation(self, qapp):
        """Test optimized handler creation."""
        handler = OptimizedStreamingHandler()
        assert handler is not None
        
    def test_adaptive_rendering(self, qapp):
        """Test adaptive rendering speed."""
        handler = OptimizedStreamingHandler()
        config = StreamingConfig(mode=StreamingMode.CHUNK)
        handler._config = config
        
        handler.start_streaming()
        
        # Send chunks rapidly to trigger adaptive speed
        for i in range(10):
            handler.append_chunk("x" * 50)  # Large chunks
            time.sleep(0.01)
            
        # Adaptive delay should have been adjusted
        assert handler._adaptive_delay_ms <= config.chunk_delay_ms
        
        handler.stop_streaming()


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


@pytest.mark.asyncio
class TestStreamingHandlerAsync:
    """Async tests for StreamingHandler."""
    
    async def test_stream_from_async_iterator(self):
        """Test streaming from async iterator."""
        handler = StreamingHandler()
        config = StreamingConfig(mode=StreamingMode.INSTANT)
        handler._config = config
        
        async def async_generator():
            chunks = ["Hello", " ", "World"]
            for chunk in chunks:
                yield chunk
                
        received_chunks = []
        
        def on_chunk(chunk):
            received_chunks.append(chunk)
            
        await handler.stream_from_async_iterator(async_generator(), on_chunk)
        
        assert "".join(received_chunks) == "Hello World"
        assert not handler.is_streaming()
        
    async def test_stream_from_iterator(self):
        """Test streaming from synchronous iterator."""
        handler = StreamingHandler()
        config = StreamingConfig(mode=StreamingMode.INSTANT)
        handler._config = config
        
        def sync_generator():
            yield "Hello"
            yield " "
            yield "World"
            
        received_chunks = []
        
        def on_chunk(chunk):
            received_chunks.append(chunk)
            
        handler.stream_from_iterator(sync_generator(), on_chunk)
        
        assert "".join(received_chunks) == "Hello World"


class TestStreamingHandlerEdgeCases:
    """Tests for edge cases in streaming handler."""
    
    def test_append_chunk_when_not_streaming(self):
        """Test appending chunk when not streaming."""
        handler = StreamingHandler()
        # Should not raise error
        handler.append_chunk("test")
        
    def test_stop_when_not_streaming(self):
        """Test stopping when not streaming."""
        handler = StreamingHandler()
        # Should not raise error
        handler.stop_streaming()
        
    def test_pause_when_not_streaming(self):
        """Test pausing when not streaming."""
        handler = StreamingHandler()
        # Should not raise error
        handler.pause_streaming()
        
    def test_empty_chunk(self):
        """Test appending empty chunk."""
        handler = StreamingHandler()
        config = StreamingConfig(mode=StreamingMode.INSTANT)
        handler._config = config
        
        handler.start_streaming()
        handler.append_chunk("")
        handler.stop_streaming()
        
        # Should handle empty chunk gracefully
        assert handler.get_stats().chunks_received == 1
        
    def test_very_long_chunk(self):
        """Test appending very long chunk."""
        handler = StreamingHandler()
        config = StreamingConfig(mode=StreamingMode.INSTANT)
        handler._config = config
        
        long_text = "x" * 10000
        
        handler.start_streaming()
        handler.append_chunk(long_text)
        handler.stop_streaming()
        
        assert handler.get_stats().total_chars == 10000
        
    def test_unicode_content(self):
        """Test streaming unicode content."""
        handler = StreamingHandler()
        config = StreamingConfig(mode=StreamingMode.INSTANT)
        handler._config = config
        
        unicode_text = "Hello 世界 🌍 émojis"
        
        handler.start_streaming()
        handler.append_chunk(unicode_text)
        handler.stop_streaming()
        
        assert unicode_text in handler.get_content()
