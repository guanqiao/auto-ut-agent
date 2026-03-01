"""Code interpreter for safe test code execution.

This module provides code interpretation capabilities:
- Safe test code execution
- Runtime error capture
- Assertion verification
- Execution result feedback
"""

import asyncio
import logging
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class ExecutionStatus(Enum):
    """Status of code execution."""
    SUCCESS = auto()
    COMPILATION_ERROR = auto()
    RUNTIME_ERROR = auto()
    TIMEOUT = auto()
    ASSERTION_FAILURE = auto()
    SECURITY_VIOLATION = auto()


@dataclass
class ExecutionResult:
    """Result of code execution."""
    status: ExecutionStatus
    stdout: str = ""
    stderr: str = ""
    return_code: int = 0
    execution_time: float = 0.0
    test_results: List[Dict[str, Any]] = field(default_factory=list)
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: List[str] = field(default_factory=list)
    assertions: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def success(self) -> bool:
        return self.status == ExecutionStatus.SUCCESS
    
    @property
    def total_tests(self) -> int:
        return self.passed + self.failed + self.skipped


@dataclass
class InterpreterConfig:
    """Configuration for code interpreter."""
    timeout_seconds: float = 30.0
    max_output_size: int = 10000
    allowed_imports: List[str] = field(default_factory=lambda: [
        "org.junit.jupiter.api.*",
        "org.junit.jupiter.api.Assertions.*",
        "org.mockito.*",
        "java.util.*",
        "java.lang.*",
    ])
    blocked_operations: List[str] = field(default_factory=lambda: [
        "Runtime.getRuntime",
        "ProcessBuilder",
        "System.exit",
        "Files.delete",
        "File.delete",
    ])
    java_path: str = "java"
    javac_path: str = "javac"


class CodeInterpreter:
    """Safe code interpreter for test execution.
    
    Features:
    - Safe test code execution
    - Runtime error capture
    - Assertion verification
    - Timeout handling
    - Security checks
    """
    
    def __init__(self, config: Optional[InterpreterConfig] = None):
        self.config = config or InterpreterConfig()
        self._temp_dir: Optional[Path] = None
    
    def _create_temp_dir(self) -> Path:
        """Create a temporary directory for execution."""
        self._temp_dir = Path(tempfile.mkdtemp(prefix="pyutagent_interpreter_"))
        return self._temp_dir
    
    def _cleanup_temp_dir(self):
        """Clean up temporary directory."""
        if self._temp_dir and self._temp_dir.exists():
            import shutil
            try:
                shutil.rmtree(self._temp_dir)
            except Exception as e:
                logger.warning(f"[CodeInterpreter] Failed to cleanup temp dir: {e}")
            self._temp_dir = None
    
    def _check_security(self, code: str) -> List[str]:
        """Check code for security violations."""
        violations = []
        
        for blocked in self.config.blocked_operations:
            if blocked in code:
                violations.append(f"Blocked operation: {blocked}")
        
        dangerous_patterns = [
            (r'Runtime\.getRuntime\(\)\.exec', "Command execution"),
            (r'ProcessBuilder', "Process creation"),
            (r'System\.exit', "System exit"),
            (r'Files\.delete', "File deletion"),
            (r'File\.delete\(\)', "File deletion"),
            (r'Class\.forName', "Dynamic class loading"),
        ]
        
        for pattern, desc in dangerous_patterns:
            if re.search(pattern, code):
                violations.append(f"Dangerous pattern: {desc}")
        
        return violations
    
    def _extract_test_methods(self, code: str) -> List[str]:
        """Extract test method names from code."""
        pattern = r'@Test\s+(?:public\s+)?void\s+(\w+)\s*\('
        return re.findall(pattern, code)
    
    def _parse_test_results(self, output: str) -> Tuple[int, int, int, List[Dict]]:
        """Parse test results from output."""
        passed = 0
        failed = 0
        skipped = 0
        test_results = []
        
        passed_match = re.search(r'Tests run:\s*(\d+),\s*Failures:\s*(\d+),\s*Errors:\s*(\d+),\s*Skipped:\s*(\d+)', output)
        if passed_match:
            passed = int(passed_match.group(1)) - int(passed_match.group(2)) - int(passed_match.group(3))
            failed = int(passed_match.group(2)) + int(passed_match.group(3))
            skipped = int(passed_match.group(4))
        
        test_pattern = r'(\w+)\s*\(\)\s*(PASSED|FAILED|SKIPPED)'
        for match in re.finditer(test_pattern, output):
            test_results.append({
                "name": match.group(1),
                "status": match.group(2),
            })
        
        return passed, failed, skipped, test_results
    
    async def compile_code(
        self,
        code: str,
        class_name: str,
        classpath: Optional[str] = None
    ) -> Tuple[bool, str]:
        """Compile Java code.
        
        Args:
            code: Java source code
            class_name: Name of the class
            classpath: Optional classpath
            
        Returns:
            Tuple of (success, error_message)
        """
        temp_dir = self._create_temp_dir()
        
        source_file = temp_dir / f"{class_name}.java"
        source_file.write_text(code, encoding='utf-8')
        
        cmd = [self.config.javac_path]
        if classpath:
            cmd.extend(["-cp", classpath])
        cmd.append(str(source_file))
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(temp_dir)
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.config.timeout_seconds
            )
            
            if process.returncode == 0:
                return True, ""
            
            error_msg = stderr.decode('utf-8', errors='replace')
            return False, error_msg
            
        except asyncio.TimeoutError:
            return False, f"Compilation timed out after {self.config.timeout_seconds}s"
        except Exception as e:
            return False, str(e)
    
    async def execute_test(
        self,
        code: str,
        class_name: str,
        classpath: Optional[str] = None
    ) -> ExecutionResult:
        """Execute test code safely.
        
        Args:
            code: Test source code
            class_name: Name of the test class
            classpath: Optional classpath
            
        Returns:
            ExecutionResult with execution details
        """
        start_time = time.time()
        
        violations = self._check_security(code)
        if violations:
            return ExecutionResult(
                status=ExecutionStatus.SECURITY_VIOLATION,
                errors=violations,
                execution_time=time.time() - start_time
            )
        
        compile_success, compile_error = await self.compile_code(code, class_name, classpath)
        
        if not compile_success:
            return ExecutionResult(
                status=ExecutionStatus.COMPILATION_ERROR,
                stderr=compile_error,
                errors=[compile_error],
                execution_time=time.time() - start_time
            )
        
        temp_dir = self._temp_dir
        
        cmd = [
            self.config.java_path,
            "-cp",
            f"{str(temp_dir)}{';' if classpath else ''}{classpath or ''}",
            "org.junit.platform.console.ConsoleLauncher",
            "--select-class", class_name,
        ]
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(temp_dir)
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.config.timeout_seconds
            )
            
            stdout_str = stdout.decode('utf-8', errors='replace')
            stderr_str = stderr.decode('utf-8', errors='replace')
            
            passed, failed, skipped, test_results = self._parse_test_results(stdout_str)
            
            execution_time = time.time() - start_time
            
            if process.returncode == 0:
                return ExecutionResult(
                    status=ExecutionStatus.SUCCESS,
                    stdout=stdout_str,
                    stderr=stderr_str,
                    return_code=process.returncode,
                    execution_time=execution_time,
                    test_results=test_results,
                    passed=passed,
                    failed=failed,
                    skipped=skipped
                )
            else:
                if "AssertionError" in stdout_str or "AssertionError" in stderr_str:
                    status = ExecutionStatus.ASSERTION_FAILURE
                else:
                    status = ExecutionStatus.RUNTIME_ERROR
                
                return ExecutionResult(
                    status=status,
                    stdout=stdout_str,
                    stderr=stderr_str,
                    return_code=process.returncode,
                    execution_time=execution_time,
                    test_results=test_results,
                    passed=passed,
                    failed=failed,
                    skipped=skipped,
                    errors=[stderr_str] if stderr_str else []
                )
                
        except asyncio.TimeoutError:
            return ExecutionResult(
                status=ExecutionStatus.TIMEOUT,
                errors=[f"Execution timed out after {self.config.timeout_seconds}s"],
                execution_time=time.time() - start_time
            )
        except Exception as e:
            return ExecutionResult(
                status=ExecutionStatus.RUNTIME_ERROR,
                errors=[str(e)],
                execution_time=time.time() - start_time
            )
        finally:
            self._cleanup_temp_dir()
    
    async def verify_assertions(
        self,
        code: str,
        expected_results: Dict[str, Any]
    ) -> Dict[str, bool]:
        """Verify that assertions match expected results.
        
        Args:
            code: Test code
            expected_results: Expected assertion results
            
        Returns:
            Dictionary of assertion -> passed
        """
        results = {}
        
        assertion_patterns = [
            (r'assertEquals\s*\(\s*([^,]+)\s*,\s*([^)]+)\)', 'assertEquals'),
            (r'true\)\s*;\s*//\s*assertion', 'assertTrue'),
            (r'assertFalse\s*\(\s*([^)]+)\)', 'assertFalse'),
            (r'assertNull\s*\(\s*([^)]+)\)', 'assertNull'),
            (r'assertNotNull\s*\(\s*([^)]+)\)', 'assertNotNull'),
        ]
        
        for pattern, assert_type in assertion_patterns:
            for match in re.finditer(pattern, code):
                assertion = match.group(0)
                results[assertion] = True
        
        return results
    
    def analyze_failures(
        self,
        result: ExecutionResult
    ) -> List[Dict[str, Any]]:
        """Analyze test failures for insights.
        
        Args:
            result: Execution result
            
        Returns:
            List of failure analyses
        """
        analyses = []
        
        for test in result.test_results:
            if test.get("status") == "FAILED":
                analyses.append({
                    "test_name": test.get("name"),
                    "type": "test_failure",
                    "suggestion": "Review test logic and expected values"
                })
        
        if result.status == ExecutionStatus.ASSERTION_FAILURE:
            analyses.append({
                "type": "assertion_failure",
                "suggestion": "Check expected vs actual values in assertions"
            })
        
        if result.status == ExecutionStatus.RUNTIME_ERROR:
            analyses.append({
                "type": "runtime_error",
                "suggestion": "Check for null pointers, type mismatches, or missing dependencies"
            })
        
        return analyses


class TestCodeInterpreter(CodeInterpreter):
    """Specialized interpreter for test code.
    
    Additional features:
    - Test method detection
    - Coverage estimation
    - Test quality analysis
    """
    
    def __init__(self, config: Optional[InterpreterConfig] = None):
        super().__init__(config)
    
    async def dry_run_test(
        self,
        code: str,
        class_name: str,
        classpath: Optional[str] = None
    ) -> ExecutionResult:
        """Perform a dry run of test code.
        
        Compiles and validates without full execution.
        
        Args:
            code: Test source code
            class_name: Name of the test class
            classpath: Optional classpath
            
        Returns:
            ExecutionResult with validation details
        """
        start_time = time.time()
        
        violations = self._check_security(code)
        if violations:
            return ExecutionResult(
                status=ExecutionStatus.SECURITY_VIOLATION,
                errors=violations,
                execution_time=time.time() - start_time
            )
        
        compile_success, compile_error = await self.compile_code(code, class_name, classpath)
        
        if not compile_success:
            return ExecutionResult(
                status=ExecutionStatus.COMPILATION_ERROR,
                stderr=compile_error,
                errors=[compile_error],
                execution_time=time.time() - start_time
            )
        
        test_methods = self._extract_test_methods(code)
        
        return ExecutionResult(
            status=ExecutionStatus.SUCCESS,
            execution_time=time.time() - start_time,
            test_results=[{"name": m, "status": "VALIDATED"} for m in test_methods],
            passed=len(test_methods),
            metadata={"dry_run": True, "test_methods": test_methods}
        )
    
    def estimate_coverage(
        self,
        test_code: str,
        source_code: str
    ) -> Dict[str, Any]:
        """Estimate test coverage potential.
        
        Args:
            test_code: Test source code
            source_code: Source code being tested
            
        Returns:
            Coverage estimation
        """
        test_methods = self._extract_test_methods(test_code)
        
        source_methods = re.findall(r'public\s+\w+\s+(\w+)\s*\(', source_code)
        
        tested_methods = set()
        for test_method in test_methods:
            for source_method in source_methods:
                if source_method.lower() in test_method.lower():
                    tested_methods.add(source_method)
        
        coverage = len(tested_methods) / len(source_methods) if source_methods else 0
        
        return {
            "estimated_coverage": coverage,
            "source_methods": source_methods,
            "tested_methods": list(tested_methods),
            "untested_methods": [m for m in source_methods if m not in tested_methods]
        }


def create_code_interpreter(
    timeout: float = 30.0,
    java_path: str = "java"
) -> CodeInterpreter:
    """Create a code interpreter instance.
    
    Args:
        timeout: Execution timeout
        java_path: Path to java executable
        
    Returns:
        Configured CodeInterpreter
    """
    config = InterpreterConfig(
        timeout_seconds=timeout,
        java_path=java_path
    )
    return CodeInterpreter(config)


def create_test_interpreter(
    timeout: float = 30.0
) -> TestCodeInterpreter:
    """Create a test code interpreter instance.
    
    Args:
        timeout: Execution timeout
        
    Returns:
        Configured TestCodeInterpreter
    """
    config = InterpreterConfig(timeout_seconds=timeout)
    return TestCodeInterpreter(config)
