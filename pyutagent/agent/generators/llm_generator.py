"""LLM-based test generator."""

import logging
from typing import Dict, Any, Optional

from .base_generator import BaseTestGenerator
from ...llm.client import LLMClient
from ...core.config import LLMConfig

logger = logging.getLogger(__name__)


class LLMTestGenerator(BaseTestGenerator):
    """Generates tests using LLM."""
    
    def __init__(
        self,
        project_path: str,
        llm_client: Optional[LLMClient] = None,
        llm_config: Optional[LLMConfig] = None
    ):
        """Initialize LLM test generator.
        
        Args:
            project_path: Path to the project
            llm_client: Optional LLM client
            llm_config: Optional LLM configuration
        """
        super().__init__(project_path)
        self._llm_client = llm_client
        self._llm_config = llm_config
    
    def _get_llm_client(self) -> LLMClient:
        """Get or create LLM client."""
        if self._llm_client is None:
            if self._llm_config:
                self._llm_client = LLMClient.from_config(self._llm_config)
            else:
                raise ValueError("LLM client or config required")
        return self._llm_client
    
    async def generate_initial_test(self, class_info: Dict[str, Any]) -> str:
        """Generate initial test code for a class.
        
        Args:
            class_info: Class information dictionary
            
        Returns:
            Generated test code
        """
        prompt = self._build_test_generation_prompt(class_info)
        
        system_prompt = """You are a Java unit test expert. Generate JUnit 5 tests following best practices:
- Use @Test annotation
- Use meaningful test method names
- Include assertions
- Mock external dependencies
- Cover edge cases

Return only the test code without explanations."""
        
        try:
            client = self._get_llm_client()
            response = await client.agenerate(prompt, system_prompt)
            return response
        except Exception as e:
            logger.exception(f"LLM generation failed: {e}")
            return self._generate_basic_test_template(class_info)
    
    async def generate_additional_tests(
        self,
        class_info: Dict[str, Any],
        uncovered_lines: list
    ) -> str:
        """Generate additional tests for uncovered lines.
        
        Args:
            class_info: Class information dictionary
            uncovered_lines: List of uncovered line numbers
            
        Returns:
            Additional test code
        """
        prompt = f"""Generate additional JUnit 5 tests for the following uncovered lines:
Class: {class_info.get('name', 'Unknown')}
Uncovered lines: {uncovered_lines}

Focus on covering these specific lines. Return only the test methods."""
        
        try:
            client = self._get_llm_client()
            response = await client.agenerate(prompt)
            return response
        except Exception as e:
            logger.exception(f"Additional test generation failed: {e}")
            return ""
    
    def _build_test_generation_prompt(self, class_info: Dict[str, Any]) -> str:
        """Build prompt for test generation.
        
        Args:
            class_info: Class information dictionary
            
        Returns:
            Prompt string
        """
        methods = class_info.get('methods', [])
        methods_str = "\n".join([
            f"- {m.get('name', 'unknown')}({', '.join(f'{t} {n}' for t, n in m.get('parameters', []))}): {m.get('return_type', 'void')}"
            for m in methods
        ])
        
        return f"""Generate JUnit 5 unit tests for the following Java class:

Package: {class_info.get('package', '')}
Class: {class_info.get('name', 'Unknown')}

Methods:
{methods_str}

Generate comprehensive tests covering:
1. Normal cases
2. Edge cases
3. Null/empty inputs
4. Exception handling

Use Mockito for mocking and AssertJ for assertions."""
