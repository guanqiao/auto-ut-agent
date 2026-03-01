"""Code extraction utilities for LLM responses."""

import re
import logging

logger = logging.getLogger(__name__)


class CodeExtractor:
    """Extract code from LLM responses.
    
    This class provides methods to extract code blocks from various
    LLM response formats (markdown code blocks, raw text, etc.).
    """
    
    # Default patterns for Java code extraction
    JAVA_PATTERNS = [
        r'```java\s*\n(.*?)```',
        r'```\s*\n(.*?)```',
    ]
    
    # Default patterns for Python code extraction
    PYTHON_PATTERNS = [
        r'```python\s*\n(.*?)```',
        r'```py\s*\n(.*?)```',
        r'```\s*\n(.*?)```',
    ]
    
    # Default patterns for generic code extraction
    GENERIC_PATTERNS = [
        r'```(?:\w+)?\s*\n(.*?)```',
    ]
    
    @staticmethod
    def extract_code(
        response: str,
        patterns: list[str] | None = None,
        language: str | None = None
    ) -> str:
        """Extract code from response using specified patterns.
        
        Args:
            response: The LLM response text
            patterns: List of regex patterns to try (optional)
            language: Language hint for default patterns (java, python, etc.)
            
        Returns:
            Extracted code or original response if no patterns match
        """
        if not response:
            return ""
        
        # Select default patterns based on language
        if patterns is None:
            if language == "java":
                patterns = CodeExtractor.JAVA_PATTERNS
            elif language == "python":
                patterns = CodeExtractor.PYTHON_PATTERNS
            else:
                patterns = CodeExtractor.GENERIC_PATTERNS
        
        # Try each pattern
        for pattern in patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            if matches:
                extracted = matches[0].strip()
                logger.debug(f"[CodeExtractor] Extracted code using pattern: {pattern[:30]}...")
                return extracted
        
        # Return original if no patterns match
        logger.debug("[CodeExtractor] No patterns matched, returning original response")
        return response.strip()
    
    @staticmethod
    def extract_java_code(response: str) -> str:
        """Extract Java code from LLM response.
        
        Args:
            response: The LLM response text
            
        Returns:
            Extracted Java code or original response
        """
        return CodeExtractor.extract_code(response, language="java")
    
    @staticmethod
    def extract_python_code(response: str) -> str:
        """Extract Python code from LLM response.
        
        Args:
            response: The LLM response text
            
        Returns:
            Extracted Python code or original response
        """
        return CodeExtractor.extract_code(response, language="python")
    
    @staticmethod
    def extract_code_from_markdown(response: str, language: str = "") -> str:
        """Extract code from markdown code blocks.
        
        Args:
            response: The LLM response text
            language: Specific language to look for (optional)
            
        Returns:
            Extracted code or original response
        """
        if not response:
            return ""
        
        # Try language-specific pattern first
        if language:
            pattern = rf'```{language}\s*\n(.*?)```'
            matches = re.findall(pattern, response, re.DOTALL)
            if matches:
                return matches[0].strip()
        
        # Try generic code block pattern
        pattern = r'```(?:\w+)?\s*\n(.*?)```'
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            return matches[0].strip()
        
        return response.strip()
