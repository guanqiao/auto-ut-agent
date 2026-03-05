"""Streaming response handler with typewriter effect."""

import logging
import time
from typing import Optional, Callable, AsyncIterator
from dataclasses import dataclass, field
from enum import Enum

from PyQt6.QtCore import QObject, pyqtSignal, QTimer, Qt
from PyQt6.QtWidgets import QApplication

logger = logging.getLogger(__name__)


class StreamingMode(Enum):
    """Streaming display modes."""
    CHARACTER = "character"  # Character by character
    WORD = "word"            # Word by word
    CHUNK = "chunk"          # Chunk by chunk (default)
    INSTANT = "instant"      # No animation


@dataclass
class StreamingConfig:
    """Configuration for streaming display."""
    mode: StreamingMode = StreamingMode.CHUNK
    char_delay_ms: int = 20      # Delay between characters
    word_delay_ms: int = 50      # Delay between words
    chunk_delay_ms: int = 10     # Delay between chunks
    batch_size: int = 5          # Characters per batch in character mode
    max_fps: int = 60            # Maximum refresh rate
    
    # Performance optimization
    enable_buffering: bool = True
    buffer_flush_ms: int = 50    # Buffer flush interval
    min_content_length: int = 10  # Minimum content to render


@dataclass
class StreamingStats:
    """Statistics for streaming performance."""
    total_chars: int = 0
    total_words: int = 0
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    chunks_received: int = 0
    render_updates: int = 0
    
    @property
    def duration_ms(self) -> float:
        """Get streaming duration in milliseconds."""
        end = self.end_time or time.time()
        return (end - self.start_time) * 1000
    
    @property
    def chars_per_second(self) -> float:
        """Get average characters per second."""
        duration_sec = self.duration_ms / 1000
        if duration_sec > 0:
            return self.total_chars / duration_sec
        return 0.0
    
    @property
    def avg_chunk_size(self) -> float:
        """Get average chunk size."""
        if self.chunks_received > 0:
            return self.total_chars / self.chunks_received
        return 0.0


class StreamingHandler(QObject):
    """Handler for streaming AI responses with typewriter effect.
    
    Features:
    - Multiple streaming modes (character, word, chunk)
    - Performance optimization with buffering
    - Configurable delays and batching
    - Real-time statistics
    """
    
    # Signals
    content_updated = pyqtSignal(str)  # Full content so far
    chunk_received = pyqtSignal(str)   # Individual chunk
    streaming_started = pyqtSignal()
    streaming_finished = pyqtSignal()
    streaming_error = pyqtSignal(str)
    stats_updated = pyqtSignal(object)  # StreamingStats
    
    def __init__(self, config: Optional[StreamingConfig] = None, 
                 parent: Optional[QObject] = None):
        super().__init__(parent)
        self._config = config or StreamingConfig()
        self._stats = StreamingStats()
        
        # Content buffer
        self._buffer = ""
        self._displayed_content = ""
        self._pending_content = ""
        
        # State
        self._is_streaming = False
        self._is_paused = False
        self._should_stop = False
        
        # Timer for rendering
        self._render_timer: Optional[QTimer] = None
        self._buffer_timer: Optional[QTimer] = None
        
        # Callbacks
        self._content_callback: Optional[Callable[[str], None]] = None
        
    def set_content_callback(self, callback: Callable[[str], None]):
        """Set callback for content updates.
        
        Args:
            callback: Function to call with updated content
        """
        self._content_callback = callback
        
    def start_streaming(self):
        """Start a new streaming session."""
        self._is_streaming = True
        self._is_paused = False
        self._should_stop = False
        self._buffer = ""
        self._displayed_content = ""
        self._pending_content = ""
        self._stats = StreamingStats(start_time=time.time())
        
        # Setup render timer
        self._setup_timers()
        
        self.streaming_started.emit()
        logger.debug("Streaming started")
        
    def _setup_timers(self):
        """Setup rendering timers."""
        # Render timer for typewriter effect
        self._render_timer = QTimer(self)
        self._render_timer.timeout.connect(self._on_render_tick)
        
        # Buffer timer for flushing
        if self._config.enable_buffering:
            self._buffer_timer = QTimer(self)
            self._buffer_timer.timeout.connect(self._flush_buffer)
            self._buffer_timer.start(self._config.buffer_flush_ms)
        
        # Start render timer based on mode
        if self._config.mode == StreamingMode.CHARACTER:
            self._render_timer.start(self._config.char_delay_ms)
        elif self._config.mode == StreamingMode.WORD:
            self._render_timer.start(self._config.word_delay_ms)
        elif self._config.mode == StreamingMode.CHUNK:
            self._render_timer.start(self._config.chunk_delay_ms)
        else:  # INSTANT
            self._render_timer.start(1)
            
    def stop_streaming(self):
        """Stop the current streaming session."""
        self._should_stop = True
        self._is_streaming = False
        
        # Flush remaining content
        self._flush_buffer()
        
        # Stop timers
        if self._render_timer:
            self._render_timer.stop()
        if self._buffer_timer:
            self._buffer_timer.stop()
            
        self._stats.end_time = time.time()
        self.streaming_finished.emit()
        self.stats_updated.emit(self._stats)
        
        logger.debug(f"Streaming stopped. Stats: {self._stats.chars_per_second:.1f} chars/sec")
        
    def pause_streaming(self):
        """Pause streaming display."""
        self._is_paused = True
        if self._render_timer:
            self._render_timer.stop()
            
    def resume_streaming(self):
        """Resume streaming display."""
        if self._is_streaming and self._is_paused:
            self._is_paused = False
            if self._render_timer:
                self._render_timer.start()
                
    def append_chunk(self, chunk: str):
        """Append a chunk of content to the stream.
        
        Args:
            chunk: Content chunk to append
        """
        if not self._is_streaming or self._should_stop:
            return
            
        self._stats.chunks_received += 1
        self._stats.total_chars += len(chunk)
        self._stats.total_words += len(chunk.split())
        
        if self._config.mode == StreamingMode.INSTANT:
            # Instant mode: append directly
            self._displayed_content += chunk
            self._update_display()
        else:
            # Buffered mode: add to pending
            self._pending_content += chunk
            
        self.chunk_received.emit(chunk)
        
    def _on_render_tick(self):
        """Handle render timer tick."""
        if self._is_paused or not self._pending_content:
            return
            
        # Determine how much to render based on mode
        if self._config.mode == StreamingMode.CHARACTER:
            render_amount = self._config.batch_size
        elif self._config.mode == StreamingMode.WORD:
            # Find next word boundary
            space_pos = self._pending_content.find(' ', 1)
            if space_pos > 0:
                render_amount = space_pos + 1
            else:
                render_amount = len(self._pending_content)
        else:  # CHUNK
            render_amount = len(self._pending_content)
            
        # Render content
        to_render = self._pending_content[:render_amount]
        self._pending_content = self._pending_content[render_amount:]
        
        self._displayed_content += to_render
        self._stats.render_updates += 1
        
        self._update_display()
        
    def _flush_buffer(self):
        """Flush any remaining buffered content."""
        if self._pending_content:
            self._displayed_content += self._pending_content
            self._pending_content = ""
            self._update_display()
            
    def _update_display(self):
        """Update the display with current content."""
        self.content_updated.emit(self._displayed_content)
        
        if self._content_callback:
            try:
                self._content_callback(self._displayed_content)
            except Exception as e:
                logger.error(f"Content callback error: {e}")
                
    def get_content(self) -> str:
        """Get the full displayed content."""
        return self._displayed_content + self._pending_content
        
    def get_stats(self) -> StreamingStats:
        """Get streaming statistics."""
        return self._stats
        
    def is_streaming(self) -> bool:
        """Check if currently streaming."""
        return self._is_streaming
        
    def is_paused(self) -> bool:
        """Check if streaming is paused."""
        return self._is_paused
        
    async def stream_from_async_iterator(
        self, 
        iterator: AsyncIterator[str],
        on_chunk: Optional[Callable[[str], None]] = None
    ):
        """Stream content from an async iterator.
        
        Args:
            iterator: Async iterator yielding content chunks
            on_chunk: Optional callback for each chunk
        """
        self.start_streaming()
        
        try:
            async for chunk in iterator:
                if self._should_stop:
                    break
                    
                self.append_chunk(chunk)
                
                if on_chunk:
                    try:
                        on_chunk(chunk)
                    except Exception as e:
                        logger.error(f"Chunk callback error: {e}")
                        
                # Process events to keep UI responsive
                QApplication.processEvents()
                
        except Exception as e:
            logger.exception(f"Streaming error: {e}")
            self.streaming_error.emit(str(e))
        finally:
            self.stop_streaming()
            
    def stream_from_iterator(
        self, 
        iterator,
        on_chunk: Optional[Callable[[str], None]] = None
    ):
        """Stream content from a synchronous iterator.
        
        Args:
            iterator: Iterator yielding content chunks
            on_chunk: Optional callback for each chunk
        """
        self.start_streaming()
        
        try:
            for chunk in iterator:
                if self._should_stop:
                    break
                    
                self.append_chunk(chunk)
                
                if on_chunk:
                    try:
                        on_chunk(chunk)
                    except Exception as e:
                        logger.error(f"Chunk callback error: {e}")
                        
                # Process events to keep UI responsive
                QApplication.processEvents()
                
        except Exception as e:
            logger.exception(f"Streaming error: {e}")
            self.streaming_error.emit(str(e))
        finally:
            self.stop_streaming()


class OptimizedStreamingHandler(StreamingHandler):
    """Optimized streaming handler with advanced performance features.
    
    Features:
    - Adaptive rendering speed based on content rate
    - Smart buffering for high-frequency updates
    - Frame rate limiting
    """
    
    def __init__(self, config: Optional[StreamingConfig] = None,
                 parent: Optional[QObject] = None):
        super().__init__(config, parent)
        
        # Performance tracking
        self._last_render_time = 0.0
        self._render_interval_ms = 1000 / self._config.max_fps if self._config.max_fps > 0 else 16
        
        # Adaptive timing
        self._adaptive_delay_ms = self._config.chunk_delay_ms
        self._content_rate_history: list[float] = []
        
    def append_chunk(self, chunk: str):
        """Append chunk with adaptive rendering."""
        if not self._is_streaming:
            return
            
        # Update rate tracking
        current_time = time.time()
        if self._last_render_time > 0:
            interval = current_time - self._last_render_time
            if interval > 0:
                rate = len(chunk) / interval
                self._content_rate_history.append(rate)
                
                # Keep history manageable
                if len(self._content_rate_history) > 10:
                    self._content_rate_history.pop(0)
                    
                # Adapt delay based on content rate
                self._adapt_render_speed()
                
        super().append_chunk(chunk)
        
    def _adapt_render_speed(self):
        """Adapt render speed based on content rate."""
        if not self._content_rate_history:
            return
            
        avg_rate = sum(self._content_rate_history) / len(self._content_rate_history)
        
        # If content is coming fast, reduce delay
        if avg_rate > 100:  # More than 100 chars/sec
            self._adaptive_delay_ms = max(5, self._config.chunk_delay_ms // 2)
        elif avg_rate > 50:
            self._adaptive_delay_ms = max(10, self._config.chunk_delay_ms)
        else:
            self._adaptive_delay_ms = self._config.chunk_delay_ms
            
        # Update timer interval
        if self._render_timer and self._render_timer.isActive():
            self._render_timer.setInterval(self._adaptive_delay_ms)
            
    def _on_render_tick(self):
        """Optimized render tick with frame limiting."""
        current_time = time.time()
        elapsed_ms = (current_time - self._last_render_time) * 1000
        
        # Frame rate limiting
        if elapsed_ms < self._render_interval_ms:
            return
            
        self._last_render_time = current_time
        super()._on_render_tick()
