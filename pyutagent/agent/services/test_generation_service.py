"""Test generation service for UT generation."""

import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable

from ...core.protocols import AgentState
from ..base_agent import StepResult
from ...core.config import get_settings

logger = logging.getLogger(__name__)


class TestGenerationService:
    """Service for generating test code.
    
    Responsibilities:
    - Parse target Java files
    - Generate initial test cases
    - Generate additional tests for uncovered code
    - Extract Java code from LLM responses
    - Manage test file operations
    """
    
    def __init__(
        self,
        project_path: str,
        java_parser: Any,
        prompt_builder: Any,
        llm_client: Any,
        progress_callback: Optional[Callable[[AgentState, str], None]] = None
    ):
        """Initialize test generation service.
        
        Args:
            project_path: Path to the project
            java_parser: Java code parser
            prompt_builder: Prompt builder for LLM
            llm_client: LLM client for generation
            progress_callback: Optional callback for progress updates
        """
        self.project_path = Path(project_path)
        self.java_parser = java_parser
        self.prompt_builder = prompt_builder
        self.llm_client = llm_client
        self.progress_callback = progress_callback
        self._stop_requested = False
    
    def stop(self):
        """Stop generation."""
        self._stop_requested = True
    
    def reset(self):
        """Reset service state."""
        self._stop_requested = False
    
    def _update_state(self, state: AgentState, message: str):
        """Update state via callback."""
        if self.progress_callback:
            self.progress_callback(state, message)
    
    async def parse_target_file(self, target_file: str) -> StepResult:
        """Parse the target Java file.
        
        Args:
            target_file: Path to the target file relative to project
            
        Returns:
            StepResult with parsed class info
        """
        if self._stop_requested:
            return StepResult(
                success=False,
                state=AgentState.PAUSED,
                message="Parsing stopped by user"
            )
        
        logger.info(f"[TestGenerationService] 📖 Parsing target file - File: {target_file}")
        self._update_state(AgentState.PARSING, f"📖 Parsing {target_file}...")
        
        try:
            file_path = self.project_path / target_file
            if not file_path.exists():
                logger.error(f"[TestGenerationService] Target file not found - Path: {file_path}")
                return StepResult(
                    success=False,
                    state=AgentState.FAILED,
                    message=f"File not found: {target_file}"
                )
            
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            logger.debug(f"[TestGenerationService] Read file content - Length: {len(source_code)}")
            
            parsed_class = self.java_parser.parse(source_code.encode('utf-8'))
            
            class_info = {
                'name': parsed_class.name,
                'package': parsed_class.package,
                'methods': [
                    {
                        'name': m.name,
                        'return_type': m.return_type,
                        'parameters': m.parameters,
                        'modifiers': m.modifiers,
                        'annotations': m.annotations,
                    }
                    for m in parsed_class.methods
                ],
                'fields': parsed_class.fields,
                'imports': parsed_class.imports,
                'annotations': parsed_class.annotations,
                'source': source_code,
            }
            
            method_count = len(class_info.get('methods', []))
            logger.info(f"[TestGenerationService] ✅ Parsing complete - Class: {class_info.get('name', 'unknown')}, Methods: {method_count}")
            
            return StepResult(
                success=True,
                state=AgentState.PARSING,
                message=f"✅ Successfully parsed {class_info.get('name', 'unknown')} ({method_count} methods)",
                data={"class_info": class_info, "source_code": source_code}
            )
        except Exception as e:
            logger.exception(f"[TestGenerationService] ❌ Failed to parse file: {e}")
            return StepResult(
                success=False,
                state=AgentState.FAILED,
                message=f"❌ Error parsing file: {str(e)}"
            )
    
    async def generate_initial_tests(
        self,
        class_info: Dict[str, Any]
    ) -> StepResult:
        """Generate initial test cases.
        
        Args:
            class_info: Parsed class information
            
        Returns:
            StepResult with generated test file path
        """
        if self._stop_requested:
            return StepResult(
                success=False,
                state=AgentState.PAUSED,
                message="Generation stopped by user"
            )
        
        class_name = class_info.get('name', 'Unknown')
        method_count = len(class_info.get('methods', []))
        logger.info(f"[TestGenerationService] ✨ Generating initial tests for {class_name} ({method_count} methods)")
        logger.info(f"[TestGenerationService] 🤖 Calling LLM to generate tests...")
        self._update_state(AgentState.GENERATING, f"✨ Generating initial tests for {class_name}...")
        
        try:
            prompt = self.prompt_builder.build_initial_test_prompt(
                class_info=class_info,
                source_code=class_info.get("source", "")
            )
            
            logger.debug(f"[TestGenerationService] Initial test prompt - Length: {len(prompt)}")
            
            response = await self.llm_client.generate(prompt)
            test_code = self._extract_java_code(response)
            
            logger.debug(f"[TestGenerationService] Extracted test code - Length: {len(test_code)}")
            
            class_name = class_info.get("name", "Unknown")
            test_file_name = f"{class_name}Test.java"
            
            settings = get_settings()
            test_dir = self.project_path / settings.project_paths.src_test_java
            package_path = class_info.get("package", "").replace(".", "/")
            if package_path:
                test_dir = test_dir / package_path
            
            test_dir.mkdir(parents=True, exist_ok=True)
            test_file_path = test_dir / test_file_name
            
            with open(test_file_path, 'w', encoding='utf-8') as f:
                f.write(test_code)
            
            relative_path = str(test_file_path.relative_to(self.project_path))
            logger.info(f"[TestGenerationService] ✅ Initial test generation complete - TestFile: {relative_path}, CodeLength: {len(test_code)} chars")
            
            return StepResult(
                success=True,
                state=AgentState.GENERATING,
                message=f"✅ Generated initial tests: {relative_path}",
                data={"test_file": relative_path, "test_code": test_code}
            )
        except Exception as e:
            logger.exception(f"[TestGenerationService] ❌ Failed to generate initial tests: {e}")
            return StepResult(
                success=False,
                state=AgentState.FAILED,
                message=f"❌ Error generating tests: {str(e)}"
            )
    
    async def generate_additional_tests(
        self,
        class_info: Dict[str, Any],
        current_test_file: str,
        coverage_data: Dict[str, Any]
    ) -> StepResult:
        """Generate additional tests for uncovered code.
        
        Args:
            class_info: Parsed class information
            current_test_file: Path to current test file
            coverage_data: Coverage analysis data
            
        Returns:
            StepResult with additional test code
        """
        if self._stop_requested:
            return StepResult(
                success=False,
                state=AgentState.PAUSED,
                message="Generation stopped by user"
            )
        
        logger.info("[TestGenerationService] Generating additional tests")
        self._update_state(AgentState.OPTIMIZING, "Generating additional tests...")
        
        try:
            report = coverage_data.get("report")
            uncovered_info = self._get_uncovered_info(report)
            
            logger.debug(f"[TestGenerationService] Uncovered info - Lines: {len(uncovered_info.get('lines', []))}")
            
            test_file_path = self.project_path / current_test_file
            with open(test_file_path, 'r', encoding='utf-8') as f:
                current_test_code = f.read()
            
            prompt = self.prompt_builder.build_additional_tests_prompt(
                class_info=class_info,
                existing_tests=current_test_code,
                uncovered_info=uncovered_info,
                current_coverage=coverage_data.get("line_coverage", 0.0)
            )
            
            logger.debug(f"[TestGenerationService] Additional tests prompt - Length: {len(prompt)}")
            
            response = await self.llm_client.agenerate(prompt)
            additional_tests = self._extract_java_code(response)
            
            logger.debug(f"[TestGenerationService] Extracted additional test code - Length: {len(additional_tests)}")
            
            self._append_tests_to_file(test_file_path, additional_tests)
            
            logger.info("[TestGenerationService] Additional test generation complete")
            
            return StepResult(
                success=True,
                state=AgentState.OPTIMIZING,
                message="Generated additional tests for uncovered code",
                data={"additional_tests": additional_tests}
            )
        except Exception as e:
            logger.exception(f"[TestGenerationService] Failed to generate additional tests: {e}")
            return StepResult(
                success=False,
                state=AgentState.FAILED,
                message=f"Error generating additional tests: {str(e)}"
            )
    
    def write_test_file(self, test_file: str, code: str) -> bool:
        """Write test code to file.
        
        Args:
            test_file: Path to test file relative to project
            code: Test code to write
            
        Returns:
            True if successful
        """
        try:
            test_file_path = self.project_path / test_file
            test_file_path.write_text(code, encoding='utf-8')
            logger.info(f"[TestGenerationService] Wrote test file - Path: {test_file_path}, Length: {len(code)}")
            return True
        except PermissionError as e:
            logger.error(f"[TestGenerationService] Permission denied writing test file: {e}")
            return False
        except OSError as e:
            logger.error(f"[TestGenerationService] OS error writing test file: {e}")
            return False
        except Exception as e:
            logger.exception(f"[TestGenerationService] Failed to write test file: {e}")
            return False
    
    def _extract_java_code(self, response: str) -> str:
        """Extract Java code from LLM response."""
        code_block_pattern = r'```(?:java)?\s*\n(.*?)```'
        matches = re.findall(code_block_pattern, response, re.DOTALL)
        
        if matches:
            return matches[0].strip()
        
        return response.strip()
    
    def _get_uncovered_info(self, report) -> Dict[str, Any]:
        """Get information about uncovered code."""
        uncovered_info = {
            "methods": [],
            "lines": [],
            "branches": []
        }
        
        if report and hasattr(report, 'files') and report.files:
            for file_coverage in report.files:
                for line_num, is_covered in file_coverage.lines:
                    if not is_covered:
                        uncovered_info["lines"].append(line_num)
        
        logger.debug(f"[TestGenerationService] Uncovered info - Lines: {len(uncovered_info['lines'])}")
        return uncovered_info
    
    def _append_tests_to_file(self, test_file_path: Path, additional_tests: str):
        """Append additional tests to existing test file."""
        with open(test_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        last_brace = content.rfind('}')
        if last_brace > 0:
            new_content = content[:last_brace] + "\n" + additional_tests + "\n" + content[last_brace:]
        else:
            new_content = content + "\n" + additional_tests
        
        with open(test_file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        logger.debug(f"[TestGenerationService] Appended tests to file - Path: {test_file_path}, AddedLength: {len(additional_tests)}")
