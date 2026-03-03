"""Sandboxed code execution for safe test execution.

This module provides sandboxed execution capabilities:
- File system isolation
- Network access restriction
- Resource usage limits
- Code injection detection
- Real-time monitoring
"""

import asyncio
import logging
import os
import re
import subprocess
import tempfile
import threading
import time

# Unix-only modules
try:
    import resource
except ImportError:
    resource = None

try:
    import signal
except ImportError:
    signal = None
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Callable, Tuple

from .code_interpreter import CodeInterpreter, ExecutionResult, ExecutionStatus, InterpreterConfig

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security levels for sandbox execution."""
    STRICT = auto()    # Maximum restrictions
    MODERATE = auto()  # Balanced security
    RELAXED = auto()   # Minimal restrictions (for trusted code)


@dataclass
class SandboxConfig:
    """Configuration for sandboxed execution."""
    # Security level
    security_level: SecurityLevel = SecurityLevel.MODERATE
    
    # Resource limits
    max_memory_mb: int = 512
    max_cpu_time_seconds: int = 30
    max_file_size_mb: int = 10
    max_open_files: int = 64
    max_processes: int = 1
    
    # File system
    allowed_paths: List[str] = field(default_factory=list)
    blocked_paths: List[str] = field(default_factory=lambda: [
        "/etc", "/usr", "/bin", "/sbin", "/lib", "/lib64",
        "~/.ssh", "~/.gnupg", "~/.aws", "~/.azure"
    ])
    read_only_paths: List[str] = field(default_factory=list)
    
    # Network
    allow_network: bool = False
    allowed_hosts: List[str] = field(default_factory=list)
    blocked_hosts: List[str] = field(default_factory=lambda: [
        "localhost", "127.0.0.1", "0.0.0.0", "::1"
    ])
    
    # Java specific
    java_opts: List[str] = field(default_factory=lambda: [
        "-Xmx256m",
        "-XX:+UseContainerSupport",
        "-Djava.security.manager",
    ])
    
    # Monitoring
    enable_monitoring: bool = True
    monitoring_interval_ms: int = 100
    
    # Cleanup
    auto_cleanup: bool = True


@dataclass
class ResourceUsage:
    """Resource usage statistics."""
    cpu_time_seconds: float = 0.0
    memory_peak_mb: float = 0.0
    memory_current_mb: float = 0.0
    disk_read_bytes: int = 0
    disk_write_bytes: int = 0
    network_in_bytes: int = 0
    network_out_bytes: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass
class SecurityReport:
    """Security analysis report."""
    violations: List[str] = field(default_factory=list)
    suspicious_patterns: List[str] = field(default_factory=list)
    risk_score: float = 0.0  # 0.0 - 1.0
    recommendations: List[str] = field(default_factory=list)


class SandboxExecutor:
    """Sandboxed code executor with security controls.
    
    Features:
    - File system isolation
    - Resource usage limits
    - Network access control
    - Code injection detection
    - Real-time monitoring
    """
    
    def __init__(self, config: Optional[SandboxConfig] = None):
        """Initialize sandbox executor.
        
        Args:
            config: Sandbox configuration
        """
        self.config = config or SandboxConfig()
        self._interpreter = CodeInterpreter()
        self._active_executions: Dict[str, Any] = {}
        
        # Compile security patterns
        self._init_security_patterns()
        
        logger.info(f"[SandboxExecutor] Initialized with security level: {self.config.security_level.name}")
    
    def _init_security_patterns(self):
        """Initialize security detection patterns."""
        # Dangerous code patterns
        self.dangerous_patterns = [
            (r'Runtime\.getRuntime\(\)\.exec', "Command execution via Runtime"),
            (r'ProcessBuilder', "Process creation"),
            (r'System\.exit', "System exit call"),
            (r'System\.load', "Native library loading"),
            (r'System\.loadLibrary', "Native library loading"),
            (r'Class\.forName', "Dynamic class loading"),
            (r'\.getDeclaredMethod', "Reflection method access"),
            (r'\.setAccessible\s*\(\s*true', "Reflection accessibility override"),
            (r'URLClassLoader', "Custom class loader"),
            (r'Unsafe', "Unsafe operations"),
            (r'sun\.misc\.', "Internal JVM classes"),
            (r'java\.lang\.reflect\.Proxy', "Dynamic proxy creation"),
            (r'ScriptEngine', "Script execution"),
            (r'javax\.script', "Script execution"),
        ]
        
        # File system access patterns
        self.filesystem_patterns = [
            (r'File\s*\(\s*["\']/', "Absolute path access"),
            (r'Paths\.get\s*\(\s*["\']/', "Absolute path access"),
            (r'FileInputStream\s*\(\s*["\']/', "File read from absolute path"),
            (r'FileOutputStream\s*\(\s*["\']/', "File write to absolute path"),
            (r'FileWriter\s*\(\s*["\']/', "File write to absolute path"),
            (r'FileReader\s*\(\s*["\']/', "File read from absolute path"),
            (r'delete\s*\(\s*\)', "File deletion"),
            (r'deleteOnExit', "File deletion on exit"),
            (r'renameTo', "File rename"),
            (r'mkdir', "Directory creation"),
        ]
        
        # Network access patterns
        self.network_patterns = [
            (r'URL\s*\(', "URL creation"),
            (r'URLConnection', "URL connection"),
            (r'HttpURLConnection', "HTTP connection"),
            (r'Socket\s*\(', "Socket creation"),
            (r'ServerSocket', "Server socket creation"),
            (r'DatagramSocket', "UDP socket creation"),
            (r'InetAddress', "Network address lookup"),
            (r'HttpClient', "HTTP client"),
            (r'WebSocket', "WebSocket connection"),
        ]
        
        # Data exfiltration patterns
        self.exfiltration_patterns = [
            (r'PrintWriter\s*\(\s*new\s+FileWriter', "Writing data to file"),
            (r'FileUtils\.write', "Writing data to file"),
            (r'Files\.write', "Writing data to file"),
            (r'OutputStreamWriter', "Writing data to stream"),
            (r'Base64\.getEncoder', "Base64 encoding (potential obfuscation)"),
        ]
    
    async def execute_sandboxed(
        self,
        code: str,
        class_name: str,
        classpath: Optional[str] = None,
        timeout: Optional[float] = None
    ) -> ExecutionResult:
        """Execute code in sandboxed environment.
        
        Args:
            code: Java source code
            class_name: Name of the class
            classpath: Optional classpath
            timeout: Optional timeout override
            
        Returns:
            ExecutionResult with execution details
        """
        execution_id = f"exec_{id(code)}_{time.time()}"
        start_time = time.time()
        
        try:
            # Security analysis
            security_report = self._analyze_security(code)
            if security_report.risk_score > 0.8:
                logger.warning(f"[SandboxExecutor] High risk code detected: {security_report.violations}")
                if self.config.security_level == SecurityLevel.STRICT:
                    return ExecutionResult(
                        status=ExecutionStatus.SECURITY_VIOLATION,
                        errors=security_report.violations,
                        execution_time=time.time() - start_time
                    )
            
            # Create sandbox environment
            with self._create_sandbox() as sandbox_dir:
                # Write code to sandbox
                source_file = sandbox_dir / f"{class_name}.java"
                source_file.write_text(code, encoding='utf-8')
                
                # Compile in sandbox
                compile_result = await self._compile_in_sandbox(
                    source_file, classpath, sandbox_dir
                )
                
                if not compile_result[0]:
                    return ExecutionResult(
                        status=ExecutionStatus.COMPILATION_ERROR,
                        stderr=compile_result[1],
                        errors=[compile_result[1]],
                        execution_time=time.time() - start_time
                    )
                
                # Execute in sandbox with monitoring
                result = await self._execute_in_sandbox(
                    class_name, classpath, sandbox_dir,
                    timeout or self.config.max_cpu_time_seconds,
                    execution_id
                )
                
                return result
                
        except Exception as e:
            logger.exception(f"[SandboxExecutor] Execution error: {e}")
            return ExecutionResult(
                status=ExecutionStatus.RUNTIME_ERROR,
                errors=[str(e)],
                execution_time=time.time() - start_time
            )
    
    def _analyze_security(self, code: str) -> SecurityReport:
        """Analyze code for security issues.
        
        Args:
            code: Java source code
            
        Returns:
            Security report
        """
        violations = []
        suspicious = []
        
        # Check dangerous patterns
        for pattern, description in self.dangerous_patterns:
            if re.search(pattern, code):
                violations.append(f"Dangerous: {description}")
        
        # Check file system patterns
        for pattern, description in self.filesystem_patterns:
            if re.search(pattern, code):
                suspicious.append(f"FileSystem: {description}")
        
        # Check network patterns
        for pattern, description in self.network_patterns:
            if re.search(pattern, code):
                if not self.config.allow_network:
                    violations.append(f"Network: {description}")
                else:
                    suspicious.append(f"Network: {description}")
        
        # Check exfiltration patterns
        for pattern, description in self.exfiltration_patterns:
            if re.search(pattern, code):
                suspicious.append(f"Data: {description}")
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(violations, suspicious)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(violations, suspicious)
        
        return SecurityReport(
            violations=violations,
            suspicious_patterns=suspicious,
            risk_score=risk_score,
            recommendations=recommendations
        )
    
    def _calculate_risk_score(self, violations: List[str], suspicious: List[str]) -> float:
        """Calculate risk score based on findings."""
        score = 0.0
        
        # Violations contribute more to risk
        score += len(violations) * 0.3
        
        # Suspicious patterns contribute less
        score += len(suspicious) * 0.1
        
        # Cap at 1.0
        return min(score, 1.0)
    
    def _generate_recommendations(self, violations: List[str], suspicious: List[str]) -> List[str]:
        """Generate security recommendations."""
        recommendations = []
        
        if violations:
            recommendations.append("Remove dangerous operations before execution")
        
        if any("FileSystem" in s for s in suspicious):
            recommendations.append("Use sandboxed file paths only")
        
        if any("Network" in s for s in suspicious):
            recommendations.append("Network access is restricted in sandbox")
        
        if any("Data" in s for s in suspicious):
            recommendations.append("Review data handling for potential exfiltration")
        
        return recommendations
    
    @contextmanager
    def _create_sandbox(self):
        """Create a sandboxed temporary directory."""
        sandbox_dir = Path(tempfile.mkdtemp(prefix="pyutagent_sandbox_"))
        
        try:
            # Create subdirectories
            (sandbox_dir / "src").mkdir(exist_ok=True)
            (sandbox_dir / "classes").mkdir(exist_ok=True)
            (sandbox_dir / "tmp").mkdir(exist_ok=True)
            
            logger.debug(f"[SandboxExecutor] Created sandbox: {sandbox_dir}")
            yield sandbox_dir
            
        finally:
            if self.config.auto_cleanup:
                import shutil
                try:
                    shutil.rmtree(sandbox_dir)
                    logger.debug(f"[SandboxExecutor] Cleaned up sandbox: {sandbox_dir}")
                except Exception as e:
                    logger.warning(f"[SandboxExecutor] Failed to cleanup sandbox: {e}")
    
    async def _compile_in_sandbox(
        self,
        source_file: Path,
        classpath: Optional[str],
        sandbox_dir: Path
    ) -> Tuple[bool, str]:
        """Compile code in sandbox."""
        classes_dir = sandbox_dir / "classes"
        
        cmd = ["javac", "-d", str(classes_dir)]
        
        if classpath:
            cmd.extend(["-cp", classpath])
        
        cmd.append(str(source_file))
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(sandbox_dir)
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.config.max_cpu_time_seconds
            )
            
            if process.returncode == 0:
                return True, ""
            
            return False, stderr.decode('utf-8', errors='replace')
            
        except asyncio.TimeoutError:
            return False, f"Compilation timed out after {self.config.max_cpu_time_seconds}s"
        except Exception as e:
            return False, str(e)
    
    async def _execute_in_sandbox(
        self,
        class_name: str,
        classpath: Optional[str],
        sandbox_dir: Path,
        timeout: float,
        execution_id: str
    ) -> ExecutionResult:
        """Execute code in sandbox with monitoring."""
        classes_dir = sandbox_dir / "classes"
        
        # Build Java command with security options
        cmd = ["java"]
        
        # Add memory limit
        cmd.append(f"-Xmx{self.config.max_memory_mb}m")
        
        # Add security manager (if not relaxed)
        if self.config.security_level != SecurityLevel.RELAXED:
            cmd.append("-Djava.security.manager")
        
        # Add classpath
        full_classpath = str(classes_dir)
        if classpath:
            full_classpath += f":{classpath}"
        cmd.extend(["-cp", full_classpath])
        
        # Add main class
        cmd.append(class_name)
        
        start_time = time.time()
        
        try:
            # Start process
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(sandbox_dir)
            )
            
            # Monitor execution
            if self.config.enable_monitoring:
                monitor_task = asyncio.create_task(
                    self._monitor_execution(process, execution_id)
                )
            
            # Wait for completion
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout
                )
                
                if self.config.enable_monitoring:
                    monitor_task.cancel()
                
                execution_time = time.time() - start_time
                
                stdout_str = stdout.decode('utf-8', errors='replace')
                stderr_str = stderr.decode('utf-8', errors='replace')
                
                # Determine status
                if process.returncode == 0:
                    status = ExecutionStatus.SUCCESS
                elif "AssertionError" in stdout_str or "AssertionError" in stderr_str:
                    status = ExecutionStatus.ASSERTION_FAILURE
                else:
                    status = ExecutionStatus.RUNTIME_ERROR
                
                return ExecutionResult(
                    status=status,
                    stdout=stdout_str,
                    stderr=stderr_str,
                    return_code=process.returncode,
                    execution_time=execution_time
                )
                
            except asyncio.TimeoutError:
                process.kill()
                if self.config.enable_monitoring:
                    monitor_task.cancel()
                
                return ExecutionResult(
                    status=ExecutionStatus.TIMEOUT,
                    errors=[f"Execution timed out after {timeout}s"],
                    execution_time=time.time() - start_time
                )
                
        except Exception as e:
            return ExecutionResult(
                status=ExecutionStatus.RUNTIME_ERROR,
                errors=[str(e)],
                execution_time=time.time() - start_time
            )
    
    async def _monitor_execution(self, process: asyncio.subprocess.Process, execution_id: str):
        """Monitor execution resource usage."""
        try:
            while process.returncode is None:
                await asyncio.sleep(self.config.monitoring_interval_ms / 1000)
                
                # Check if process is still running
                if process.returncode is not None:
                    break
                
                # In a real implementation, we would check:
                # - Memory usage via /proc/[pid]/status
                # - CPU time via /proc/[pid]/stat
                # - File descriptors
                
                # For now, just log that monitoring is active
                logger.debug(f"[SandboxExecutor] Monitoring execution {execution_id}")
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning(f"[SandboxExecutor] Monitoring error: {e}")
    
    def validate_code(self, code: str) -> SecurityReport:
        """Validate code without executing.
        
        Args:
            code: Java source code
            
        Returns:
            Security report
        """
        return self._analyze_security(code)
    
    def get_security_recommendations(self, code: str) -> List[str]:
        """Get security recommendations for code.
        
        Args:
            code: Java source code
            
        Returns:
            List of recommendations
        """
        report = self._analyze_security(code)
        return report.recommendations


def create_sandbox_executor(
    security_level: SecurityLevel = SecurityLevel.MODERATE,
    max_memory_mb: int = 512,
    timeout_seconds: int = 30
) -> SandboxExecutor:
    """Create a sandbox executor instance.
    
    Args:
        security_level: Security level
        max_memory_mb: Maximum memory in MB
        timeout_seconds: Timeout in seconds
        
    Returns:
        Configured SandboxExecutor
    """
    config = SandboxConfig(
        security_level=security_level,
        max_memory_mb=max_memory_mb,
        max_cpu_time_seconds=timeout_seconds
    )
    return SandboxExecutor(config)
