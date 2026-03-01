"""Aider-based test generator with iterative improvement."""

import logging
from typing import Dict, Any, Optional

from .base_generator import BaseTestGenerator
from ...tools.aider_integration import AiderTestGenerator as AiderTG, AiderCodeFixer
from ...llm.client import LLMClient
from ...core.config import LLMConfig

logger = logging.getLogger(__name__)


class AiderTestGenerator(BaseTestGenerator):
    """Generates tests using Aider-style iterative improvement."""
    
    def __init__(
        self,
        project_path: str,
        llm_client: Optional[LLMClient] = None,
        llm_config: Optional[LLMConfig] = None,
        max_fix_attempts: int = 3
    ):
        """Initialize Aider test generator.
        
        Args:
            project_path: Path to the project
            llm_client: Optional LLM client
            llm_config: Optional LLM configuration
            max_fix_attempts: Maximum attempts for fixing errors
        """
        super().__init__(project_path)
        self._llm_client = llm_client
        self._llm_config = llm_config
        self._max_fix_attempts = max_fix_attempts
        self._aider_generator: Optional[AiderTG] = None
        self._aider_fixer: Optional[AiderCodeFixer] = None
    
    def _get_llm_client(self) -> LLMClient:
        """Get or create LLM client."""
        if self._llm_client is None:
            if self._llm_config:
                self._llm_client = LLMClient.from_config(self._llm_config)
            else:
                raise ValueError("LLM client or config required")
        return self._llm_client
    
    def _get_aider_generator(self) -> AiderTG:
        """Get or create Aider test generator."""
        if self._aider_generator is None:
            self._aider_generator = AiderTG(self._get_llm_client())
        return self._aider_generator
    
    def _get_aider_fixer(self) -> AiderCodeFixer:
        """Get or create Aider code fixer."""
        if self._aider_fixer is None:
            self._aider_fixer = AiderCodeFixer(
                self._get_llm_client(),
                max_attempts=self._max_fix_attempts
            )
        return self._aider_fixer
    
    async def generate_initial_test(self, class_info: Dict[str, Any]) -> str:
        """Generate initial test code using Aider.
        
        Args:
            class_info: Class information dictionary
            
        Returns:
            Generated test code
        """
        try:
            return await self._get_aider_generator().generate_initial_test(class_info)
        except Exception as e:
            logger.exception(f"Aider initial test generation failed: {e}")
            return self._generate_basic_test_template(class_info)
    
    async def generate_additional_tests(
        self,
        class_info: Dict[str, Any],
        uncovered_lines: list
    ) -> str:
        """Generate additional tests using Aider.
        
        Args:
            class_info: Class information dictionary
            uncovered_lines: List of uncovered line numbers
            
        Returns:
            Additional test code
        """
        try:
            fix_result = await self._get_aider_fixer().improve_coverage(
                test_code="",  # Will be filled by caller
                uncovered_lines=uncovered_lines,
                class_info=class_info
            )
            
            if fix_result.success:
                return fix_result.fixed_code
            else:
                logger.warning(f"Aider coverage improvement failed: {fix_result.error_message}")
                return ""
        except Exception as e:
            logger.exception(f"Aider additional test generation failed: {e}")
            return ""
    
    async def fix_compilation_errors(
        self,
        test_code: str,
        compiler_output: str,
        error_analysis: Any
    ) -> Optional[str]:
        """Fix compilation errors using Aider.
        
        Args:
            test_code: Current test code
            compiler_output: Compiler error output
            error_analysis: Error analysis result
            
        Returns:
            Fixed code or None
        """
        try:
            fix_result = await self._get_aider_fixer().fix_compilation_errors(
                test_code=test_code,
                error_analysis=error_analysis
            )
            
            if fix_result.success:
                logger.info(f"Compilation errors fixed in {fix_result.attempts} attempts")
                return fix_result.fixed_code
            else:
                logger.warning(f"Failed to fix compilation errors: {fix_result.error_message}")
                return None
        except Exception as e:
            logger.exception(f"Aider compilation fix failed: {e}")
            return None
    
    async def fix_test_failures(
        self,
        test_code: str,
        failure_analysis: Any
    ) -> Optional[str]:
        """Fix test failures using Aider.
        
        Args:
            test_code: Current test code
            failure_analysis: Failure analysis result
            
        Returns:
            Fixed code or None
        """
        try:
            fix_result = await self._get_aider_fixer().fix_test_failures(
                test_code=test_code,
                failure_analysis=failure_analysis
            )
            
            if fix_result.success:
                logger.info(f"Test failures fixed in {fix_result.attempts} attempts")
                return fix_result.fixed_code
            else:
                logger.warning(f"Failed to fix test failures: {fix_result.error_message}")
                return None
        except Exception as e:
            logger.exception(f"Aider test failure fix failed: {e}")
            return None
    
    async def improve_coverage(
        self,
        test_code: str,
        uncovered_lines: list,
        class_info: Dict[str, Any]
    ) -> Optional[str]:
        """Improve test coverage using Aider.
        
        Args:
            test_code: Current test code
            uncovered_lines: List of uncovered line numbers
            class_info: Class information dictionary
            
        Returns:
            Improved test code or None
        """
        try:
            fix_result = await self._get_aider_fixer().improve_coverage(
                test_code=test_code,
                uncovered_lines=uncovered_lines,
                class_info=class_info
            )
            
            if fix_result.success:
                return fix_result.fixed_code
            else:
                logger.warning(f"Coverage improvement failed: {fix_result.error_message}")
                return None
        except Exception as e:
            logger.exception(f"Aider coverage improvement failed: {e}")
            return None
