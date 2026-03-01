"""Handler for test compilation operations."""

import logging
import asyncio
import subprocess
from pathlib import Path
from typing import List, Optional, Callable, Dict, Any

from ...core.protocols import AgentState
from ..base_agent import StepResult
from ...core.config import get_settings
from ...tools.maven_tools import MavenRunner

logger = logging.getLogger(__name__)


class CompilationHandler:
    """Handles test compilation operations."""
    
    def __init__(
        self,
        project_path: str,
        progress_callback: Optional[Callable[[AgentState, str], None]] = None,
        maven_runner: Optional[MavenRunner] = None
    ):
        """Initialize compilation handler.
        
        Args:
            project_path: Path to the project
            progress_callback: Optional callback for progress updates
            maven_runner: Optional MavenRunner instance for reuse
        """
        self.project_path = Path(project_path)
        self.progress_callback = progress_callback
        self._stop_requested = False
        self._maven_runner = maven_runner or MavenRunner(project_path)
    
    def stop(self):
        """Stop compilation."""
        self._stop_requested = True
    
    def reset(self):
        """Reset handler state."""
        self._stop_requested = False
    
    def _update_state(self, state: AgentState, message: str):
        """Update state via callback."""
        if self.progress_callback:
            self.progress_callback(state, message)
    
    async def compile_tests(
        self,
        test_file: str,
        attempt: int = 1
    ) -> StepResult:
        """Compile the generated tests.
        
        Args:
            test_file: Path to the test file relative to project
            attempt: Current attempt number
            
        Returns:
            StepResult with compilation outcome
        """
        if self._stop_requested:
            return StepResult(
                success=False,
                state=AgentState.PAUSED,
                message="Compilation stopped by user"
            )
        
        self._update_state(AgentState.COMPILING, f"Attempt {attempt}: Compiling tests...")
        logger.info(f"[CompilationHandler] Starting compilation - Attempt: {attempt}, File: {test_file}")
        
        try:
            classpath = await self._build_classpath()
            result = await self._run_javac(test_file, classpath)
            
            if result.returncode == 0:
                logger.info("[CompilationHandler] Compilation successful")
                self._update_state(AgentState.COMPILING, "Compilation successful")
                return StepResult(
                    success=True,
                    state=AgentState.COMPILING,
                    message="Tests compiled successfully"
                )
            else:
                errors = self._extract_errors(result)
                logger.warning(f"[CompilationHandler] Compilation failed - Errors: {len(errors)}")
                return StepResult(
                    success=False,
                    state=AgentState.FIXING,
                    message="Compilation failed",
                    data={"errors": errors, "stdout": result.stdout}
                )
        except Exception as e:
            logger.exception(f"[CompilationHandler] Compilation exception: {e}")
            return StepResult(
                success=False,
                state=AgentState.FAILED,
                message=f"Error compiling tests: {str(e)}"
            )
    
    async def _build_classpath(self) -> str:
        """Build the classpath for compilation.
        
        Returns:
            Classpath string
        """
        logger.debug("[CompilationHandler] Building classpath")
        
        try:
            classpath = await self._maven_runner.get_classpath_async()
        except Exception as e:
            logger.warning(f"[CompilationHandler] Failed to get Maven classpath: {e}")
            classpath = ""
        
        settings = get_settings()
        target_classes = self.project_path / settings.project_paths.target_classes
        target_test_classes = self.project_path / settings.project_paths.target_test_classes
        
        classpath_parts = [str(target_classes), str(target_test_classes)]
        if classpath:
            classpath_parts.append(classpath)
        
        full_classpath = ";".join(classpath_parts)
        logger.debug(f"[CompilationHandler] Full classpath length: {len(full_classpath)}")
        
        return full_classpath
    
    async def _run_javac(
        self,
        test_file: str,
        classpath: str
    ) -> subprocess.CompletedProcess:
        """Run javac command.
        
        Args:
            test_file: Path to test file
            classpath: Classpath string
            
        Returns:
            CompletedProcess result
        """
        test_file_path = self.project_path / test_file
        output_dir = self.project_path / "target" / "test-classes"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        compile_cmd = [
            "javac",
            "-cp", classpath,
            "-d", str(output_dir),
            str(test_file_path)
        ]
        
        logger.debug(f"[CompilationHandler] Running: javac -cp ... {test_file_path}")
        
        return subprocess.run(
            compile_cmd,
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
    
    def _extract_errors(
        self,
        result: subprocess.CompletedProcess
    ) -> List[str]:
        """Extract compilation errors from result.
        
        Args:
            result: CompletedProcess from javac
            
        Returns:
            List of error messages
        """
        errors = []
        if result.stderr:
            errors.append(result.stderr)
        if not errors:
            errors.append("Unknown compilation error")
        return errors
    
    def get_compiler_errors(self, test_file: str) -> List[str]:
        """Get detailed compiler errors for a test file.
        
        Args:
            test_file: Path to test file
            
        Returns:
            List of error details
        """
        logger.debug(f"[CompilationHandler] Getting compiler errors for: {test_file}")
        return []
