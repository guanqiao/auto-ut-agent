"""Streaming code generation for real-time output.

This module provides streaming capabilities for code generation:
- Real-time code streaming from LLM
- Progress callbacks for UI updates
- Interruptible generation
- Code preview during generation
"""

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Any,
    Union,
)

logger = logging.getLogger(__name__)


class StreamingState(Enum):
    """States for streaming generation."""
    IDLE = auto()
    STREAMING = auto()
    COMPLETED = auto()
    INTERRUPTED = auto()
    FAILED = auto()


@dataclass
class CodePreview:
    """Preview of code being generated."""
    partial_code: str
    accumulated_code: str
    is_complete: bool
    progress_percent: float = 0.0
    tokens_generated: int = 0
    elapsed_time: float = 0.0


@dataclass
class StreamingResult:
    """Result of streaming generation."""
    success: bool
    complete_code: str
    state: StreamingState
    total_tokens: int = 0
    total_time: float = 0.0
    error: Optional[Exception] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamingConfig:
    """Configuration for streaming generation."""
    chunk_callback: Optional[Callable[[str], None]] = None
    preview_callback: Optional[Callable[[CodePreview], None]] = None
    completion_callback: Optional[Callable[[str], None]] = None
    error_callback: Optional[Callable[[Exception], None]] = None
    progress_callback: Optional[Callable[[float], None]] = None
    enable_preview: bool = True
    preview_interval: float = 0.1
    max_tokens: int = 8192
    timeout: float = 300.0


class StreamingCodeGenerator:
    """Streaming code generator with real-time output.
    
    Features:
    - Real-time streaming from LLM
    - Progress callbacks for UI updates
    - Interruptible generation
    - Code preview during generation
    - Token counting and timing
    
    Example:
        generator = StreamingCodeGenerator(llm_client)
        
        async for preview in generator.generate_with_preview(prompt):
            print(f"Progress: {preview.progress_percent:.1f}%")
            print(f"Code so far: {preview.accumulated_code[:100]}...")
        
        result = await generator.get_result()
    """
    
    def __init__(
        self,
        llm_client: Any,
        config: Optional[StreamingConfig] = None
    ):
        self.llm_client = llm_client
        self.config = config or StreamingConfig()
        
        self._state = StreamingState.IDLE
        self._interrupt_event = asyncio.Event()
        self._interrupt_event.set()
        
        self._accumulated_code: List[str] = []
        self._total_tokens = 0
        self._start_time: Optional[float] = None
        self._result: Optional[StreamingResult] = None
    
    @property
    def state(self) -> StreamingState:
        return self._state
    
    @property
    def is_streaming(self) -> bool:
        return self._state == StreamingState.STREAMING
    
    def interrupt(self):
        """Interrupt the current streaming generation."""
        logger.info("[StreamingGenerator] Interrupting generation")
        self._interrupt_event.clear()
        self._state = StreamingState.INTERRUPTED
    
    def reset(self):
        """Reset the generator state."""
        self._state = StreamingState.IDLE
        self._interrupt_event.set()
        self._accumulated_code = []
        self._total_tokens = 0
        self._start_time = None
        self._result = None
    
    async def generate_with_streaming(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        on_chunk: Optional[Callable[[str], None]] = None,
        on_complete: Optional[Callable[[str], None]] = None,
        on_progress: Optional[Callable[[float], None]] = None
    ) -> StreamingResult:
        """Generate code with streaming output.
        
        Args:
            prompt: The prompt for code generation
            system_prompt: Optional system prompt
            on_chunk: Callback for each chunk received
            on_complete: Callback when generation completes
            on_progress: Callback for progress updates (0.0-1.0)
            
        Returns:
            StreamingResult with the complete generated code
        """
        self.reset()
        self._state = StreamingState.STREAMING
        self._start_time = time.time()
        last_progress_time = self._start_time
        
        chunk_callback = on_chunk or self.config.chunk_callback
        completion_callback = on_complete or self.config.completion_callback
        progress_callback = on_progress or self.config.progress_callback
        
        logger.info(f"[StreamingGenerator] 🚀 Starting streaming generation - PromptLength: {len(prompt)}")
        logger.info("[StreamingGenerator] ⏳ 等待 LLM 流式响应中... (通常需要 10-60 秒)")
        
        # Report initial progress
        if progress_callback:
            try:
                progress_callback(0.0)
            except Exception as e:
                logger.warning(f"[StreamingGenerator] Progress callback error: {e}")
        
        try:
            chunk_count = 0
            first_chunk_time = None
            
            # Create a timeout wrapper for the stream
            stream = self.llm_client.astream(prompt, system_prompt, timeout=self.config.timeout)
            
            async for chunk in stream:
                if not self._interrupt_event.is_set():
                    logger.info("[StreamingGenerator] ⚠️ Generation interrupted by user")
                    self._state = StreamingState.INTERRUPTED
                    break
                
                # Track first chunk time for timeout detection
                if first_chunk_time is None:
                    first_chunk_time = time.time()
                    logger.info("[StreamingGenerator] 📥 接收到第一个数据块")
                
                self._accumulated_code.append(chunk)
                self._total_tokens += len(chunk.split())
                chunk_count += 1
                
                if chunk_callback:
                    try:
                        chunk_callback(chunk)
                    except Exception as e:
                        logger.warning(f"[StreamingGenerator] Chunk callback error: {e}")
                
                if progress_callback:
                    progress = min(1.0, self._total_tokens / self.config.max_tokens)
                    try:
                        progress_callback(progress)
                    except Exception as e:
                        logger.warning(f"[StreamingGenerator] Progress callback error: {e}")
                
                # Report progress every 5 seconds
                current_time = time.time()
                elapsed = current_time - self._start_time
                if current_time - last_progress_time >= 5:
                    logger.info(f"[StreamingGenerator] ⏳ 正在接收流式响应... 已接收 {chunk_count} 个数据块，耗时 {elapsed:.0f} 秒")
                    last_progress_time = current_time
            
            # Check if we received any chunks
            if chunk_count == 0:
                elapsed = time.time() - self._start_time
                logger.warning(f"[StreamingGenerator] ⚠️ No chunks received after {elapsed:.1f}s. LLM may have failed silently.")
            
            complete_code = ''.join(self._accumulated_code)
            elapsed = time.time() - self._start_time
            
            if self._state == StreamingState.INTERRUPTED:
                self._result = StreamingResult(
                    success=False,
                    complete_code=complete_code,
                    state=StreamingState.INTERRUPTED,
                    total_tokens=self._total_tokens,
                    total_time=elapsed,
                    metadata={"reason": "User interrupted"}
                )
            else:
                self._state = StreamingState.COMPLETED
                self._result = StreamingResult(
                    success=True,
                    complete_code=complete_code,
                    state=StreamingState.COMPLETED,
                    total_tokens=self._total_tokens,
                    total_time=elapsed
                )
                
                if completion_callback:
                    try:
                        completion_callback(complete_code)
                    except Exception as e:
                        logger.warning(f"[StreamingGenerator] Completion callback error: {e}")
            
            if self._state == StreamingState.COMPLETED:
                logger.info(
                    f"[StreamingGenerator] ✅ 流式生成完成 - "
                    f"数据块: {chunk_count}, "
                    f"Token数: {self._total_tokens}, "
                    f"耗时: {elapsed:.2f}s"
                )
            else:
                logger.info(
                    f"[StreamingGenerator] Generation complete - "
                    f"State: {self._state.name}, "
                    f"Tokens: {self._total_tokens}, "
                    f"Time: {elapsed:.2f}s"
                )
            
            return self._result
            
        except Exception as e:
            elapsed = time.time() - self._start_time if self._start_time else 0
            self._state = StreamingState.FAILED
            
            if self.config.error_callback:
                try:
                    self.config.error_callback(e)
                except Exception as callback_error:
                    logger.warning(f"[StreamingGenerator] Error callback failed: {callback_error}")
            
            logger.exception(f"[StreamingGenerator] Generation failed: {e}")
            
            self._result = StreamingResult(
                success=False,
                complete_code=''.join(self._accumulated_code),
                state=StreamingState.FAILED,
                total_tokens=self._total_tokens,
                total_time=elapsed,
                error=e
            )
            
            return self._result
    
    async def generate_with_preview(
        self,
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> AsyncIterator[CodePreview]:
        """Generate code with real-time preview.
        
        Yields CodePreview objects during generation.
        
        Args:
            prompt: The prompt for code generation
            system_prompt: Optional system prompt
            
        Yields:
            CodePreview objects with partial code and progress
        """
        self.reset()
        self._state = StreamingState.STREAMING
        self._start_time = time.time()
        
        last_preview_time = 0.0
        preview_interval = self.config.preview_interval
        
        logger.info(f"[StreamingGenerator] Starting preview generation - PromptLength: {len(prompt)}")
        
        try:
            async for chunk in self.llm_client.astream(prompt, system_prompt, timeout=self.config.timeout):
                if not self._interrupt_event.is_set():
                    self._state = StreamingState.INTERRUPTED
                    break
                
                self._accumulated_code.append(chunk)
                self._total_tokens += len(chunk.split())
                
                current_time = time.time()
                elapsed = current_time - self._start_time
                
                if self.config.enable_preview and (current_time - last_preview_time >= preview_interval):
                    last_preview_time = current_time
                    
                    accumulated = ''.join(self._accumulated_code)
                    progress = min(1.0, self._total_tokens / self.config.max_tokens)
                    
                    preview = CodePreview(
                        partial_code=chunk,
                        accumulated_code=accumulated,
                        is_complete=False,
                        progress_percent=progress * 100,
                        tokens_generated=self._total_tokens,
                        elapsed_time=elapsed
                    )
                    
                    if self.config.preview_callback:
                        try:
                            self.config.preview_callback(preview)
                        except Exception as e:
                            logger.warning(f"[StreamingGenerator] Preview callback error: {e}")
                    
                    yield preview
            
            complete_code = ''.join(self._accumulated_code)
            elapsed = time.time() - self._start_time
            
            if self._state != StreamingState.INTERRUPTED:
                self._state = StreamingState.COMPLETED
            
            final_preview = CodePreview(
                partial_code="",
                accumulated_code=complete_code,
                is_complete=True,
                progress_percent=100.0,
                tokens_generated=self._total_tokens,
                elapsed_time=elapsed
            )
            
            yield final_preview
            
            self._result = StreamingResult(
                success=self._state == StreamingState.COMPLETED,
                complete_code=complete_code,
                state=self._state,
                total_tokens=self._total_tokens,
                total_time=elapsed
            )
            
        except Exception as e:
            self._state = StreamingState.FAILED
            elapsed = time.time() - self._start_time if self._start_time else 0
            
            self._result = StreamingResult(
                success=False,
                complete_code=''.join(self._accumulated_code),
                state=StreamingState.FAILED,
                total_tokens=self._total_tokens,
                total_time=elapsed,
                error=e
            )
            
            logger.exception(f"[StreamingGenerator] Preview generation failed: {e}")
            raise
    
    def get_result(self) -> Optional[StreamingResult]:
        """Get the result of the last generation."""
        return self._result
    
    def get_accumulated_code(self) -> str:
        """Get the code accumulated so far."""
        return ''.join(self._accumulated_code)


class StreamingTestGenerator(StreamingCodeGenerator):
    """Streaming generator specialized for test code generation.
    
    Additional features:
    - Java code extraction from streaming output
    - Test method detection during streaming
    - Import statement tracking
    """
    
    def __init__(
        self,
        llm_client: Any,
        config: Optional[StreamingConfig] = None
    ):
        super().__init__(llm_client, config)
        self._detected_methods: List[str] = []
        self._detected_imports: List[str] = []
        self._in_code_block = False
    
    async def generate_tests_with_preview(
        self,
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """Generate test code with enhanced preview information.
        
        Yields dictionaries with:
        - preview: CodePreview object
        - detected_methods: List of test methods found so far
        - detected_imports: List of imports found so far
        - in_code_block: Whether currently inside a code block
        """
        self._detected_methods = []
        self._detected_imports = []
        self._in_code_block = False
        
        async for preview in self.generate_with_preview(prompt, system_prompt):
            self._analyze_code(preview.accumulated_code)
            
            yield {
                "preview": preview,
                "detected_methods": self._detected_methods.copy(),
                "detected_imports": self._detected_imports.copy(),
                "in_code_block": self._in_code_block,
                "state": self._state
            }
    
    def _analyze_code(self, code: str):
        """Analyze accumulated code for structure."""
        if '```' in code:
            self._in_code_block = code.count('```') % 2 == 1
        
        method_pattern = r'@Test\s+(?:public\s+)?void\s+(\w+)\s*\('
        for match in re.finditer(method_pattern, code):
            method_name = match.group(1)
            if method_name not in self._detected_methods:
                self._detected_methods.append(method_name)
        
        import_pattern = r'import\s+(?:static\s+)?[\w.]+;'
        for match in re.finditer(import_pattern, code):
            import_stmt = match.group(0)
            if import_stmt not in self._detected_imports:
                self._detected_imports.append(import_stmt)
    
    def extract_java_code(self) -> str:
        """Extract Java code from the accumulated output."""
        code = self.get_accumulated_code()
        
        code_block_pattern = r'```(?:java)?\s*\n(.*?)```'
        matches = re.findall(code_block_pattern, code, re.DOTALL)
        
        if matches:
            return matches[0].strip()
        
        return code.strip()
    
    def get_detected_methods(self) -> List[str]:
        """Get list of detected test methods."""
        return self._detected_methods.copy()
    
    def get_detected_imports(self) -> List[str]:
        """Get list of detected imports."""
        return self._detected_imports.copy()


class StreamingManager:
    """Manager for coordinating multiple streaming operations.
    
    Features:
    - Queue multiple generation requests
    - Priority-based scheduling
    - Cancellation support
    - Progress aggregation
    """
    
    def __init__(self, llm_client: Any):
        self.llm_client = llm_client
        self._active_generators: Dict[str, StreamingCodeGenerator] = {}
        self._queue: asyncio.Queue = asyncio.Queue()
        self._lock = asyncio.Lock()
    
    async def start_generation(
        self,
        generation_id: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        config: Optional[StreamingConfig] = None
    ) -> StreamingResult:
        """Start a new streaming generation.
        
        Args:
            generation_id: Unique identifier for this generation
            prompt: The prompt for generation
            system_prompt: Optional system prompt
            config: Optional streaming configuration
            
        Returns:
            StreamingResult when generation completes
        """
        async with self._lock:
            if generation_id in self._active_generators:
                raise ValueError(f"Generation {generation_id} already exists")
            
            generator = StreamingCodeGenerator(self.llm_client, config)
            self._active_generators[generation_id] = generator
        
        try:
            result = await generator.generate_with_streaming(
                prompt=prompt,
                system_prompt=system_prompt
            )
            return result
        finally:
            async with self._lock:
                self._active_generators.pop(generation_id, None)
    
    def interrupt_generation(self, generation_id: str) -> bool:
        """Interrupt a specific generation.
        
        Args:
            generation_id: ID of the generation to interrupt
            
        Returns:
            True if generation was found and interrupted
        """
        generator = self._active_generators.get(generation_id)
        if generator:
            generator.interrupt()
            return True
        return False
    
    def interrupt_all(self):
        """Interrupt all active generations."""
        for generator in self._active_generators.values():
            generator.interrupt()
    
    def get_active_count(self) -> int:
        """Get number of active generations."""
        return len(self._active_generators)
    
    def get_generation_state(self, generation_id: str) -> Optional[StreamingState]:
        """Get state of a specific generation."""
        generator = self._active_generators.get(generation_id)
        return generator.state if generator else None


def create_streaming_generator(
    llm_client: Any,
    on_chunk: Optional[Callable[[str], None]] = None,
    on_complete: Optional[Callable[[str], None]] = None,
    on_progress: Optional[Callable[[float], None]] = None
) -> StreamingCodeGenerator:
    """Create a streaming generator with callbacks.
    
    Args:
        llm_client: LLM client instance
        on_chunk: Callback for each chunk
        on_complete: Callback when complete
        on_progress: Callback for progress updates
        
    Returns:
        Configured StreamingCodeGenerator
    """
    config = StreamingConfig(
        chunk_callback=on_chunk,
        completion_callback=on_complete,
        progress_callback=on_progress
    )
    return StreamingCodeGenerator(llm_client, config)
