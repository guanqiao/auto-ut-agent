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
        self._javac_path: Optional[str] = None
    
    def _get_javac_path(self) -> str:
        """Get javac executable path.
        
        Priority:
        1. User configured java_home from settings
        2. Auto-detected javac path
        3. Fallback to "javac"
        """
        if self._javac_path is not None:
            return self._javac_path
        
        try:
            from ...tools.java_tools import get_configured_java_paths
            _, javac_path = get_configured_java_paths()
            self._javac_path = javac_path or "javac"
            logger.debug(f"[CompilationHandler] Using javac: {self._javac_path}")
        except Exception as e:
            logger.warning(f"[CompilationHandler] Failed to get javac path: {e}")
            self._javac_path = "javac"
        
        return self._javac_path
    
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
        """Compile the generated tests using Maven (preferred) or javac as fallback.
        
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
        
        self._update_state(AgentState.COMPILING, f"Attempt {attempt}: Compiling with Maven...")
        logger.info(f"[CompilationHandler] Starting Maven compilation - Attempt: {attempt}, File: {test_file}")
        
        try:
            # Step 1: Try Maven to compile both production and test code
            maven_success, maven_output = await self._compile_with_maven()
            
            if maven_success:
                logger.info("[CompilationHandler] Maven compilation successful")
                self._update_state(AgentState.COMPILING, "Compilation successful")
                return StepResult(
                    success=True,
                    state=AgentState.COMPILING,
                    message="Tests compiled successfully via Maven"
                )
            
            # Step 2: If Maven fails, try javac as fallback
            logger.warning(f"[CompilationHandler] Maven compilation failed, falling back to javac")
            self._update_state(AgentState.COMPILING, "Maven failed, trying javac...")
            
            classpath = await self._build_classpath()
            result = await self._run_javac(test_file, classpath)
            
            if result.returncode == 0:
                logger.info("[CompilationHandler] Javac compilation successful")
                self._update_state(AgentState.COMPILING, "Compilation successful (javac)")
                return StepResult(
                    success=True,
                    state=AgentState.COMPILING,
                    message="Tests compiled successfully via javac"
                )
            else:
                errors = self._extract_errors(result)
                logger.warning(f"[CompilationHandler] Javac compilation failed - Errors: {len(errors)}")
                return StepResult(
                    success=False,
                    state=AgentState.FIXING,
                    message="Compilation failed",
                    data={"errors": errors, "stdout": result.stdout, "maven_output": maven_output}
                )
        except Exception as e:
            logger.exception(f"[CompilationHandler] Compilation exception: {e}")
            return StepResult(
                success=False,
                state=AgentState.FAILED,
                message=f"Error compiling tests: {str(e)}"
            )
    
    async def _compile_with_maven(self) -> tuple[bool, str]:
        """Compile production and test code using Maven.
        
        Returns:
            Tuple of (success, output)
        """
        logger.info("[CompilationHandler] Running Maven compile and test-compile")
        
        try:
            # First compile production code, then test code
            compile_success = await self._maven_runner.compile_project_async()
            
            if not compile_success:
                logger.error("[CompilationHandler] Maven production code compilation failed")
                return False, "Failed to compile production code"
            
            # Run test-compile to compile test classes
            test_compile_success = await self._compile_test_classes_async()
            
            if test_compile_success:
                logger.info("[CompilationHandler] Maven test compilation successful")
                return True, "Compilation successful"
            else:
                logger.error("[CompilationHandler] Maven test compilation failed")
                return False, "Failed to compile test code"
                
        except Exception as e:
            logger.exception(f"[CompilationHandler] Maven compilation error: {e}")
            return False, str(e)
    
    async def _compile_test_classes_async(self) -> bool:
        """Run Maven test-compile goal to compile test classes.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            process = await asyncio.create_subprocess_exec(
                "mvn", "test-compile", "-q",
                cwd=str(self.project_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            return process.returncode == 0
        except Exception as e:
            logger.exception(f"[CompilationHandler] Error running test-compile: {e}")
            return False
    
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
        
        javac_path = self._get_javac_path()
        compile_cmd = [
            javac_path,
            "-cp", classpath,
            "-d", str(output_dir),
            str(test_file_path)
        ]
        
        logger.debug(f"[CompilationHandler] Running: {javac_path} -cp ... {test_file_path}")
        
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
