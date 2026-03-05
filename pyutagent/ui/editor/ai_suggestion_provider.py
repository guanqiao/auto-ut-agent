"""AI suggestion provider for inline code completion.

Generates code suggestions using LLM and provides them to the editor
for ghost text display and inline diff.
"""

import logging
import asyncio
from typing import Optional, List, Callable, Any
from dataclasses import dataclass
from enum import Enum, auto
from concurrent.futures import ThreadPoolExecutor
import time

from PyQt6.QtCore import QObject, pyqtSignal, QTimer

logger = logging.getLogger(__name__)


class SuggestionType(Enum):
    """Type of AI suggestion."""
    COMPLETION = auto()     # Continue/complete current line
    INSERTION = auto()      # Insert new code at cursor
    REPLACEMENT = auto()    # Replace selected/ surrounding code


@dataclass
class CodeSuggestion:
    """Represents a code suggestion from AI."""
    text: str
    suggestion_type: SuggestionType
    start_line: int
    start_column: int
    end_line: Optional[int] = None
    end_column: Optional[int] = None
    confidence: float = 0.0
    explanation: str = ""


@dataclass
class SuggestionContext:
    """Context for generating suggestions."""
    file_path: Optional[str]
    language: str
    cursor_line: int
    cursor_column: int
    text_before_cursor: str
    text_after_cursor: str
    selected_text: str
    full_text: str


class SuggestionCache:
    """Cache for suggestion results to avoid redundant API calls."""
    
    def __init__(self, max_size: int = 100, ttl_seconds: float = 30.0):
        self._cache: dict = {}
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        
    def _make_key(self, context: SuggestionContext) -> str:
        """Create a cache key from context."""
        # Use relevant parts of context for key
        return hash((
            context.file_path,
            context.language,
            context.cursor_line,
            context.cursor_column,
            context.text_before_cursor[-200:],  # Last 200 chars before cursor
            context.text_after_cursor[:50],     # First 50 chars after cursor
        ))
        
    def get(self, context: SuggestionContext) -> Optional[CodeSuggestion]:
        """Get cached suggestion if available and not expired."""
        key = self._make_key(context)
        if key in self._cache:
            suggestion, timestamp = self._cache[key]
            if time.time() - timestamp < self._ttl_seconds:
                logger.debug("Cache hit for suggestion")
                return suggestion
            else:
                # Expired
                del self._cache[key]
        return None
        
    def set(self, context: SuggestionContext, suggestion: CodeSuggestion):
        """Cache a suggestion."""
        key = self._make_key(context)
        
        # Evict oldest if at capacity
        if len(self._cache) >= self._max_size:
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][1])
            del self._cache[oldest_key]
            
        self._cache[key] = (suggestion, time.time())
        logger.debug("Suggestion cached")
        
    def clear(self):
        """Clear all cached suggestions."""
        self._cache.clear()
        logger.debug("Suggestion cache cleared")


class AISuggestionProvider(QObject):
    """Provider for AI-powered code suggestions.
    
    Features:
    - Async suggestion generation
    - Suggestion caching
    - Debounced requests
    - Multiple suggestion types
    """
    
    # Signals
    suggestion_ready = pyqtSignal(CodeSuggestion)
    suggestion_error = pyqtSignal(str)
    suggestion_started = pyqtSignal()
    
    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        
        self._cache = SuggestionCache()
        self._debounce_timer: Optional[QTimer] = None
        self._debounce_delay_ms = 300  # 300ms debounce
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._pending_context: Optional[SuggestionContext] = None
        self._is_generating = False
        
        # LLM configuration (placeholder for actual LLM integration)
        self._llm_config = {
            'model': 'default',
            'max_tokens': 150,
            'temperature': 0.2,
        }
        
        self._setup_debounce_timer()
        
    def _setup_debounce_timer(self):
        """Setup the debounce timer."""
        self._debounce_timer = QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.timeout.connect(self._generate_suggestion_async)
        
    def request_suggestion(self, context: SuggestionContext, 
                          suggestion_type: SuggestionType = SuggestionType.COMPLETION):
        """Request a code suggestion.
        
        Args:
            context: The context for suggestion generation
            suggestion_type: Type of suggestion to generate
        """
        # Store context for later use
        self._pending_context = context
        self._pending_suggestion_type = suggestion_type
        
        # Check cache first
        cached = self._cache.get(context)
        if cached:
            self.suggestion_ready.emit(cached)
            return
            
        # Debounce the request
        if self._debounce_timer:
            self._debounce_timer.stop()
            self._debounce_timer.start(self._debounce_delay_ms)
            
    def _generate_suggestion_async(self):
        """Generate suggestion in background thread."""
        if not self._pending_context or self._is_generating:
            return
            
        self._is_generating = True
        self.suggestion_started.emit()
        
        context = self._pending_context
        suggestion_type = self._pending_suggestion_type
        
        # Run in thread pool
        self._executor.submit(self._generate_suggestion_worker, context, suggestion_type)
        
    def _generate_suggestion_worker(self, context: SuggestionContext, 
                                    suggestion_type: SuggestionType):
        """Worker function for suggestion generation."""
        try:
            suggestion = self._generate_suggestion(context, suggestion_type)
            
            # Cache the result
            self._cache.set(context, suggestion)
            
            # Emit result (thread-safe via Qt's queued connection)
            self.suggestion_ready.emit(suggestion)
            
        except Exception as e:
            logger.exception("Failed to generate suggestion")
            self.suggestion_error.emit(str(e))
        finally:
            self._is_generating = False
            
    def _generate_suggestion(self, context: SuggestionContext,
                            suggestion_type: SuggestionType) -> CodeSuggestion:
        """Generate a code suggestion based on context.
        
        This is a placeholder implementation. In production, this would
        call an actual LLM API.
        
        Args:
            context: The context for suggestion generation
            suggestion_type: Type of suggestion to generate
            
        Returns:
            A CodeSuggestion object
        """
        # Build prompt based on suggestion type
        prompt = self._build_prompt(context, suggestion_type)
        
        # Call LLM (placeholder)
        response = self._call_llm(prompt)
        
        # Parse response
        suggestion_text = self._parse_suggestion(response, context, suggestion_type)
        
        return CodeSuggestion(
            text=suggestion_text,
            suggestion_type=suggestion_type,
            start_line=context.cursor_line,
            start_column=context.cursor_column,
            confidence=0.85,  # Placeholder confidence
            explanation="AI-generated code suggestion"
        )
        
    def _build_prompt(self, context: SuggestionContext, 
                     suggestion_type: SuggestionType) -> str:
        """Build the prompt for the LLM.
        
        Args:
            context: The suggestion context
            suggestion_type: Type of suggestion
            
        Returns:
            The prompt string
        """
        language = context.language or "code"
        
        if suggestion_type == SuggestionType.COMPLETION:
            prompt = f"""Complete the following {language} code. Only provide the completion, no explanations.

Context:
```{language}
{context.text_before_cursor[-500:]}
```

Continue from where the cursor is (after the above code)."""

        elif suggestion_type == SuggestionType.INSERTION:
            prompt = f"""Insert code at the cursor position in the following {language} file.

Before cursor:
```{language}
{context.text_before_cursor[-300:]}
```

After cursor:
```{language}
{context.text_after_cursor[:300]}
```

Provide only the code to insert, no explanations."""

        elif suggestion_type == SuggestionType.REPLACEMENT:
            prompt = f"""Replace the following {language} code with an improved version.

Original code:
```{language}
{context.selected_text or context.text_before_cursor[-300:]}
```

Provide only the replacement code, no explanations."""

        else:
            prompt = f"""Suggest code for the following {language} context:

```{language}
{context.text_before_cursor[-500:]}
```

Provide only the suggested code, no explanations."""

        return prompt
        
    def _call_llm(self, prompt: str) -> str:
        """Call the LLM with the given prompt.
        
        This is a placeholder implementation. In production, this would
        call an actual LLM API like OpenAI, Anthropic, or local models.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            The LLM response
        """
        # Placeholder: return mock suggestions based on context
        # In production, replace with actual API call
        
        # Simple heuristics for demo purposes
        if "def " in prompt and ":" not in prompt.split("def ")[-1][:50]:
            return "():\n    pass"
        elif "if " in prompt and ":" not in prompt.split("if ")[-1][:30]:
            return ":\n    pass"
        elif "for " in prompt and ":" not in prompt.split("for ")[-1][:30]:
            return ":\n    pass"
        elif "class " in prompt and ":" not in prompt.split("class ")[-1][:50]:
            return ":\n    pass"
        elif "import" in prompt[-50:]:
            return " os\nimport sys"
        elif "return" in prompt[-50:]:
            return " None"
        else:
            # Default suggestion
            return "  # TODO: Implement"
            
    def _parse_suggestion(self, response: str, context: SuggestionContext,
                         suggestion_type: SuggestionType) -> str:
        """Parse the LLM response into a suggestion text.
        
        Args:
            response: The LLM response
            context: The suggestion context
            suggestion_type: Type of suggestion
            
        Returns:
            Cleaned suggestion text
        """
        # Remove code block markers if present
        text = response.strip()
        
        # Remove markdown code blocks
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first line (```language)
            if lines[0].startswith("```"):
                lines = lines[1:]
            # Remove last line (```)
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)
            
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
        
    def cancel_pending(self):
        """Cancel any pending suggestion requests."""
        if self._debounce_timer:
            self._debounce_timer.stop()
        self._pending_context = None
        logger.debug("Pending suggestions cancelled")
        
    def clear_cache(self):
        """Clear the suggestion cache."""
        self._cache.clear()
        
    def set_debounce_delay(self, delay_ms: int):
        """Set the debounce delay for suggestion requests.
        
        Args:
            delay_ms: Delay in milliseconds
        """
        self._debounce_delay_ms = max(50, delay_ms)
        logger.debug(f"Debounce delay set to {delay_ms}ms")
        
    def is_generating(self) -> bool:
        """Check if a suggestion is currently being generated."""
        return self._is_generating


class MockAISuggestionProvider(AISuggestionProvider):
    """Mock provider for testing without actual LLM calls."""
    
    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._mock_suggestions: dict = {}
        
    def set_mock_suggestion(self, context_pattern: str, suggestion: str):
        """Set a mock suggestion for a context pattern."""
        self._mock_suggestions[context_pattern] = suggestion
        
    def _call_llm(self, prompt: str) -> str:
        """Return mock suggestion based on prompt."""
        # Check for matching patterns
        for pattern, suggestion in self._mock_suggestions.items():
            if pattern in prompt:
                return suggestion
                
        # Default mock suggestions
        if "def " in prompt:
            return "(self, x):\n    return x * 2"
        elif "if " in prompt:
            return " x > 0:\n    return x"
        elif "for " in prompt:
            return " i in range(10):\n    print(i)"
        elif "class " in prompt:
            return " MyClass:\n    def __init__(self):\n        pass"
        elif "import" in prompt:
            return " os"
        elif "return" in prompt:
            return " result"
        else:
            return "  # AI suggestion"


class SuggestionManager(QObject):
    """Manages suggestions and their lifecycle in the editor.
    
    Coordinates between the AI provider, ghost text display, and diff view.
    """
    
    suggestion_accepted = pyqtSignal(CodeSuggestion)
    suggestion_rejected = pyqtSignal(CodeSuggestion)
    
    def __init__(self, editor: 'CodeEditor', parent: Optional[QObject] = None):
        super().__init__(parent)
        
        self._editor = editor
        self._provider = AISuggestionProvider(self)
        self._current_suggestion: Optional[CodeSuggestion] = None
        
        # Connect provider signals
        self._provider.suggestion_ready.connect(self._on_suggestion_ready)
        self._provider.suggestion_error.connect(self._on_suggestion_error)
        
    def request_completion(self):
        """Request a completion suggestion at current cursor position."""
        context = self._build_context()
        self._provider.request_suggestion(context, SuggestionType.COMPLETION)
        
    def request_insertion(self):
        """Request an insertion suggestion at current cursor position."""
        context = self._build_context()
        self._provider.request_suggestion(context, SuggestionType.INSERTION)
        
    def request_replacement(self):
        """Request a replacement suggestion for selected text."""
        context = self._build_context()
        self._provider.request_suggestion(context, SuggestionType.REPLACEMENT)
        
    def _build_context(self) -> SuggestionContext:
        """Build suggestion context from current editor state."""
        cursor = self._editor.textCursor()
        
        # Get cursor position
        cursor_line = cursor.blockNumber() + 1
        cursor_column = cursor.columnNumber()
        
        # Get text before and after cursor
        full_text = self._editor.toPlainText()
        cursor_pos = cursor.position()
        text_before = full_text[:cursor_pos]
        text_after = full_text[cursor_pos:]
        
        # Get selected text
        selected_text = cursor.selectedText()
        
        # Get file info
        file_path = getattr(self._editor, '_file_path', None)
        language = getattr(self._editor, '_language', '')
        
        return SuggestionContext(
            file_path=file_path,
            language=language,
            cursor_line=cursor_line,
            cursor_column=cursor_column,
            text_before_cursor=text_before,
            text_after_cursor=text_after,
            selected_text=selected_text,
            full_text=full_text
        )
        
    def _on_suggestion_ready(self, suggestion: CodeSuggestion):
        """Handle new suggestion from provider."""
        self._current_suggestion = suggestion
        # The editor will handle displaying this
        logger.info(f"Suggestion ready: {len(suggestion.text)} chars")
        
    def _on_suggestion_error(self, error: str):
        """Handle suggestion generation error."""
        logger.error(f"Suggestion error: {error}")
        self._current_suggestion = None
        
    def get_current_suggestion(self) -> Optional[CodeSuggestion]:
        """Get the current suggestion."""
        return self._current_suggestion
        
    def accept_current_suggestion(self) -> bool:
        """Accept the current suggestion.
        
        Returns:
            True if suggestion was accepted
        """
        if not self._current_suggestion:
            return False
            
        self.suggestion_accepted.emit(self._current_suggestion)
        self._current_suggestion = None
        return True
        
    def reject_current_suggestion(self) -> bool:
        """Reject the current suggestion.
        
        Returns:
            True if suggestion was rejected
        """
        if not self._current_suggestion:
            return False
            
        self.suggestion_rejected.emit(self._current_suggestion)
        self._current_suggestion = None
        return True
        
    def clear_suggestion(self):
        """Clear the current suggestion."""
        self._current_suggestion = None
        self._provider.cancel_pending()
