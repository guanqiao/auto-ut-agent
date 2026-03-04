"""Voice interaction module for coding agent.

This module provides:
- VoiceInputHandler: Speech recognition input
- VoiceOutputHandler: TTS voice output
- VoiceCommandParser: Voice command parsing
"""

import asyncio
import base64
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable

logger = logging.getLogger(__name__)


class VoiceProvider(Enum):
    """Voice service providers."""
    WHISPER = "whisper"
    GOOGLE = "google"
    AZURE = "azure"
    OPENAI = "openai"
    LOCAL = "local"


class TTSProvider(Enum):
    """TTS service providers."""
    GTTS = "gtts"
    EDGE_TTS = "edge_tts"
    AZURE_TTS = "azure_tts"
    OPENAI_TTS = "openai"
    PYTTSX3 = "pyttsx3"


@dataclass
class VoiceConfig:
    """Voice configuration."""
    provider: VoiceProvider = VoiceProvider.WHISPER
    tts_provider: TTSProvider = TTSProvider.GTTS
    language: str = "zh-CN"
    api_key: Optional[str] = None
    endpoint: Optional[str] = None
    voice: str = "zh-CN-YunxiNeural"
    rate: str = "+0%"
    volume: str = "+0%"


@dataclass
class VoiceCommand:
    """Parsed voice command."""
    command: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    raw_text: str = ""


class VoiceCommandParser:
    """Parser for voice commands."""

    COMMAND_PATTERNS = [
        (r"(?:生成|写|创建).*测试", "generate_test", {"task": "test"}),
        (r"(?:修复|修复).*错误", "fix_error", {}),
        (r"(?:分析|审查).*代码", "analyze_code", {}),
        (r"暂停", "pause", {}),
        (r"(?:继续|恢复)", "resume", {}),
        (r"(?:停止|终止)", "stop", {}),
        (r"(?:查看|显示).*状态", "show_status", {}),
        (r"帮助", "help", {}),
    ]

    def __init__(self):
        self.custom_patterns: List[tuple] = []

    def add_pattern(self, pattern: str, command: str, params: Dict[str, Any] = None):
        """Add custom command pattern.

        Args:
            pattern: Regex pattern
            command: Command name
            params: Default parameters
        """
        self.custom_patterns.append((pattern, command, params or {}))

    async def parse_command(self, text: str) -> VoiceCommand:
        """Parse voice text into command.

        Args:
            text: Recognized text

        Returns:
            VoiceCommand
        """
        import re

        text = text.strip()

        for pattern, command, default_params in self.COMMAND_PATTERNS + self.custom_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return VoiceCommand(
                    command=command,
                    parameters=default_params.copy(),
                    confidence=0.9,
                    raw_text=text
                )

        return VoiceCommand(
            command="unknown",
            parameters={"raw": text},
            confidence=0.0,
            raw_text=text
        )


class VoiceInputHandler:
    """Handler for voice input (speech recognition)."""

    def __init__(self, config: Optional[VoiceConfig] = None):
        """Initialize voice input handler.

        Args:
            config: Voice configuration
        """
        self.config = config or VoiceConfig()
        self.recognizer = None
        self.is_listening = False
        self._audio_queue: asyncio.Queue = asyncio.Queue()

    async def initialize(self) -> bool:
        """Initialize the recognizer.

        Returns:
            True if initialization successful
        """
        try:
            if self.config.provider == VoiceProvider.WHISPER:
                await self._init_whisper()
            elif self.config.provider == VoiceProvider.LOCAL:
                await self._init_local()
            logger.info(f"[VoiceInputHandler] Initialized with provider: {self.config.provider.value}")
            return True
        except Exception as e:
            logger.error(f"[VoiceInputHandler] Initialization failed: {e}")
            return False

    async def _init_whisper(self):
        """Initialize Whisper for speech recognition."""
        pass

    async def _init_local(self):
        """Initialize local speech recognition."""
        pass

    async def start_listening(self) -> None:
        """Start listening for voice input."""
        self.is_listening = True
        logger.info("[VoiceInputHandler] Started listening")

    async def stop_listening(self) -> str:
        """Stop listening and return recognized text.

        Returns:
            Recognized text
        """
        self.is_listening = False
        logger.info("[VoiceInputHandler] Stopped listening")

        if not self._audio_queue.empty():
            audio_data = await self._audio_queue.get()
            return await self.transcribe_audio(audio_data)

        return ""

    async def transcribe_audio(self, audio_data: bytes) -> str:
        """Transcribe audio data to text.

        Args:
            audio_data: Raw audio bytes

        Returns:
            Transcribed text
        """
        try:
            if self.config.provider == VoiceProvider.WHISPER and self.config.api_key:
                return await self._transcribe_whisper(audio_data)
            elif self.config.provider == VoiceProvider.OPENAI and self.config.api_key:
                return await self._transcribe_openai(audio_data)
            else:
                return await self._transcribe_mock(audio_data)
        except Exception as e:
            logger.error(f"[VoiceInputHandler] Transcription failed: {e}")
            return ""

    async def _transcribe_whisper(self, audio_data: bytes) -> str:
        """Transcribe using Whisper API."""
        import httpx

        encoded = base64.b64encode(audio_data).decode()

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.openai.com/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {self.config.api_key}"},
                files={"file": ("audio.wav", audio_data, "audio/wav")},
                data={"model": "whisper-1", "language": self.config.language}
            )

        if response.status_code == 200:
            return response.json().get("text", "")
        return ""

    async def _transcribe_openai(self, audio_data: bytes) -> str:
        """Transcribe using OpenAI API."""
        return await self._transcribe_whisper(audio_data)

    async def _transcribe_mock(self, audio_data: bytes) -> str:
        """Mock transcription for testing."""
        return "Mock transcribed text"

    async def listen_continuously(
        self,
        callback: Callable[[str], Any],
        stop_event: asyncio.Event
    ) -> None:
        """Listen continuously until stop event.

        Args:
            callback: Callback for recognized text
            stop_event: Event to signal stop
        """
        logger.info("[VoiceInputHandler] Starting continuous listening")

        while not stop_event.is_set():
            if self.is_listening:
                text = await self._listen_once()
                if text:
                    await callback(text)

            await asyncio.sleep(0.1)

        logger.info("[VoiceInputHandler] Stopped continuous listening")

    async def _listen_once(self) -> str:
        """Listen for a single utterance."""
        await asyncio.sleep(0.5)
        return ""


class VoiceOutputHandler:
    """Handler for voice output (TTS)."""

    def __init__(self, config: Optional[VoiceConfig] = None):
        """Initialize voice output handler.

        Args:
            config: Voice configuration
        """
        self.config = config or VoiceConfig()
        self.is_speaking = False
        self._current_task: Optional[asyncio.Task] = None

    async def initialize(self) -> bool:
        """Initialize TTS engine.

        Returns:
            True if initialization successful
        """
        try:
            if self.config.tts_provider == TTSProvider.GTTS:
                await self._init_gtts()
            elif self.config.tts_provider == TTSProvider.EDGE_TTS:
                await self._init_edge_tts()

            logger.info(f"[VoiceOutputHandler] Initialized with provider: {self.config.tts_provider.value}")
            return True
        except Exception as e:
            logger.error(f"[VoiceOutputHandler] Initialization failed: {e}")
            return False

    async def _init_gtts(self):
        """Initialize Google TTS."""
        pass

    async def _init_edge_tts(self):
        """Initialize Edge TTS."""
        pass

    async def speak(
        self,
        text: str,
        interrupt: bool = True
    ) -> None:
        """Speak text using TTS.

        Args:
            text: Text to speak
            interrupt: Whether to interrupt current speech
        """
        if interrupt and self._current_task:
            self._current_task.cancel()
            self.is_speaking = False

        self._current_task = asyncio.create_task(self._speak_async(text))

    async def _speak_async(self, text: str) -> None:
        """Internal async speak method."""
        self.is_speaking = True
        logger.info(f"[VoiceOutputHandler] Speaking: {text[:50]}...")

        try:
            if self.config.tts_provider == TTSProvider.GTTS:
                await self._speak_gtts(text)
            elif self.config.tts_provider == TTSProvider.EDGE_TTS:
                await self._speak_edge(text)
            else:
                await self._speak_mock(text)
        except Exception as e:
            logger.error(f"[VoiceOutputHandler] Speech failed: {e}")
        finally:
            self.is_speaking = False

    async def _speak_gtts(self, text: str):
        """Speak using Google TTS."""
        try:
            from gtts import gTTS
            import io

            tts = gTTS(text=text, lang=self.config.language)
            audio_bytes = io.BytesIO()
            tts.write_to_fp(audio_bytes)
            audio_bytes.seek(0)

            await self._play_audio(audio_bytes.read())
        except ImportError:
            logger.warning("[VoiceOutputHandler] gTTS not installed")

    async def _speak_edge(self, text: str):
        """Speak using Edge TTS."""
        logger.warning("[VoiceOutputHandler] Edge TTS not fully implemented")

    async def _speak_mock(self, text: str):
        """Mock speak for testing."""
        await asyncio.sleep(0.5)

    async def _play_audio(self, audio_data: bytes) -> None:
        """Play audio data."""
        logger.debug(f"[VoiceOutputHandler] Playing {len(audio_data)} bytes of audio")

    async def speak_with_voice(
        self,
        text: str,
        voice: str,
        interrupt: bool = True
    ) -> None:
        """Speak with specific voice.

        Args:
            text: Text to speak
            voice: Voice name
            interrupt: Whether to interrupt
        """
        original_voice = self.config.voice
        self.config.voice = voice
        await self.speak(text, interrupt)
        self.config.voice = original_voice

    def stop(self) -> None:
        """Stop current speech."""
        if self._current_task:
            self._current_task.cancel()
        self.is_speaking = False
        logger.info("[VoiceOutputHandler] Stopped speaking")


async def create_voice_handlers(config: Optional[VoiceConfig] = None) -> tuple:
    """Create and initialize voice handlers.

    Args:
        config: Voice configuration

    Returns:
        Tuple of (input_handler, output_handler)
    """
    config = config or VoiceConfig()

    input_handler = VoiceInputHandler(config)
    output_handler = VoiceOutputHandler(config)

    await input_handler.initialize()
    await output_handler.initialize()

    return input_handler, output_handler
