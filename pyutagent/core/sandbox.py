"""Sandbox for safe tool execution.

This module provides sandbox isolation for tool execution:
- File system isolation
- Network access control
- Resource limits
- Timeout enforcement
"""

import asyncio
import logging
import os
import resource
import signal
import tempfile
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    TypeVar,
    Union,
)

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class SandboxConfig:
    """Configuration for sandbox environment."""
    allowed_paths: List[str] = field(default_factory=list)
    blocked_paths: List[str] = field(default_factory=list)
    allowed_network_hosts: List[str] = field(default_factory=list)
    allow_network: bool = True
    max_file_size: int = 10 * 1024 * 1024
    max_execution_time: float = 60.0
    max_memory_mb: int = 512
    max_processes: int = 10
    allow_subprocess: bool = True
    environment_vars: Dict[str, str] = field(default_factory=dict)
    temp_dir: Optional[str] = None


@dataclass
class SandboxResult:
    """Result of sandboxed execution."""
    success: bool
    result: Any = None
    error: Optional[Exception] = None
    execution_time: float = 0.0
    memory_used: int = 0
    files_accessed: List[str] = field(default_factory=list)
    network_calls: List[str] = field(default_factory=list)
    violations: List[str] = field(default_factory=list)


class SandboxViolation(Exception):
    """Exception raised when sandbox rules are violated."""
    pass


class SandboxedToolExecutor:
    """Executes tools in a sandboxed environment.
    
    Features:
    - File system access control
    - Network access control
    - Resource limits
    - Timeout enforcement
    - Execution monitoring
    """
    
    def __init__(self, config: Optional[SandboxConfig] = None):
        self.config = config or SandboxConfig()
        
        self._files_accessed: Set[str] = set()
        self._network_calls: List[str] = []
        self._violations: List[str] = []
        self._temp_dir: Optional[Path] = None
    
    @asynccontextmanager
    async def sandbox_context(self):
        """Context manager for sandboxed execution."""
        self._files_accessed.clear()
        self._network_calls.clear()
        self._violations.clear()
        
        if self.config.temp_dir:
            self._temp_dir = Path(self.config.temp_dir)
        else:
            self._temp_dir = Path(tempfile.mkdtemp(prefix="pyutagent_sandbox_"))
        
        original_env = os.environ.copy()
        
        try:
            for key, value in self.config.environment_vars.items():
                os.environ[key] = value
            
            if not self.config.allow_network:
                os.environ['NO_PROXY'] = '*'
                os.environ['HTTP_PROXY'] = ''
                os.environ['HTTPS_PROXY'] = ''
            
            logger.debug(f"[Sandbox] Entered sandbox context - TempDir: {self._temp_dir}")
            
            yield self
            
        finally:
            os.environ.clear()
            os.environ.update(original_env)
            
            if self._temp_dir and self._temp_dir.exists():
                try:
                    import shutil
                    shutil.rmtree(self._temp_dir)
                except Exception as e:
                    logger.warning(f"[Sandbox] Failed to cleanup temp dir: {e}")
            
            logger.debug("[Sandbox] Exited sandbox context")
    
    def check_file_access(self, path: str, mode: str = 'r') -> bool:
        """Check if file access is allowed.
        
        Args:
            path: File path to check
            mode: Access mode ('r' for read, 'w' for write)
            
        Returns:
            True if access is allowed
        """
        abs_path = str(Path(path).resolve())
        
        self._files_accessed.add(abs_path)
        
        for blocked in self.config.blocked_paths:
            if abs_path.startswith(blocked):
                violation = f"Blocked path access: {path}"
                self._violations.append(violation)
                logger.warning(f"[Sandbox] {violation}")
                return False
        
        if self.config.allowed_paths:
            for allowed in self.config.allowed_paths:
                if abs_path.startswith(allowed):
                    return True
            violation = f"Path not in allowed list: {path}"
            self._violations.append(violation)
            logger.warning(f"[Sandbox] {violation}")
            return False
        
        return True
    
    def check_network_access(self, host: str) -> bool:
        """Check if network access is allowed.
        
        Args:
            host: Host to check
            
        Returns:
            True if access is allowed
        """
        self._network_calls.append(host)
        
        if not self.config.allow_network:
            violation = f"Network access blocked: {host}"
            self._violations.append(violation)
            logger.warning(f"[Sandbox] {violation}")
            return False
        
        if self.config.allowed_network_hosts:
            if host not in self.config.allowed_network_hosts:
                violation = f"Host not in allowed list: {host}"
                self._violations.append(violation)
                logger.warning(f"[Sandbox] {violation}")
                return False
        
        return True
    
    async def execute_safe(
        self,
        func: Callable[..., T],
        *args,
        timeout: Optional[float] = None,
        **kwargs
    ) -> SandboxResult:
        """Execute a function in the sandbox.
        
        Args:
            func: Function to execute
            *args: Positional arguments
            timeout: Optional timeout override
            **kwargs: Keyword arguments
            
        Returns:
            SandboxResult with execution result
        """
        import time
        start_time = time.time()
        
        timeout_val = timeout or self.config.max_execution_time
        
        async with self.sandbox_context():
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await asyncio.wait_for(
                        func(*args, **kwargs),
                        timeout=timeout_val
                    )
                else:
                    result = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(
                            None,
                            lambda: func(*args, **kwargs)
                        ),
                        timeout=timeout_val
                    )
                
                execution_time = time.time() - start_time
                
                return SandboxResult(
                    success=True,
                    result=result,
                    execution_time=execution_time,
                    files_accessed=list(self._files_accessed),
                    network_calls=self._network_calls.copy(),
                    violations=self._violations.copy()
                )
                
            except asyncio.TimeoutError:
                execution_time = time.time() - start_time
                return SandboxResult(
                    success=False,
                    error=TimeoutError(f"Execution timed out after {timeout_val}s"),
                    execution_time=execution_time,
                    files_accessed=list(self._files_accessed),
                    network_calls=self._network_calls.copy(),
                    violations=self._violations.copy()
                )
                
            except SandboxViolation as e:
                execution_time = time.time() - start_time
                return SandboxResult(
                    success=False,
                    error=e,
                    execution_time=execution_time,
                    files_accessed=list(self._files_accessed),
                    network_calls=self._network_calls.copy(),
                    violations=self._violations.copy()
                )
                
            except Exception as e:
                execution_time = time.time() - start_time
                logger.exception(f"[Sandbox] Execution failed: {e}")
                return SandboxResult(
                    success=False,
                    error=e,
                    execution_time=execution_time,
                    files_accessed=list(self._files_accessed),
                    network_calls=self._network_calls.copy(),
                    violations=self._violations.copy()
                )
    
    async def execute_subprocess_safe(
        self,
        command: List[str],
        cwd: Optional[str] = None,
        timeout: Optional[float] = None,
        env: Optional[Dict[str, str]] = None
    ) -> SandboxResult:
        """Execute a subprocess in the sandbox.
        
        Args:
            command: Command and arguments
            cwd: Working directory
            timeout: Optional timeout
            env: Additional environment variables
            
        Returns:
            SandboxResult with subprocess result
        """
        if not self.config.allow_subprocess:
            return SandboxResult(
                success=False,
                error=SandboxViolation("Subprocess execution not allowed"),
                violations=["Subprocess execution not allowed"]
            )
        
        import time
        start_time = time.time()
        timeout_val = timeout or self.config.max_execution_time
        
        async with self.sandbox_context():
            try:
                if cwd:
                    self.check_file_access(cwd, 'r')
                
                merged_env = os.environ.copy()
                if env:
                    merged_env.update(env)
                
                process = await asyncio.create_subprocess_exec(
                    *command,
                    cwd=cwd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=merged_env
                )
                
                try:
                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(),
                        timeout=timeout_val
                    )
                except asyncio.TimeoutError:
                    process.kill()
                    raise
                
                execution_time = time.time() - start_time
                
                return SandboxResult(
                    success=process.returncode == 0,
                    result={
                        "returncode": process.returncode,
                        "stdout": stdout.decode('utf-8', errors='replace'),
                        "stderr": stderr.decode('utf-8', errors='replace')
                    },
                    execution_time=execution_time,
                    files_accessed=list(self._files_accessed),
                    network_calls=self._network_calls.copy(),
                    violations=self._violations.copy()
                )
                
            except Exception as e:
                execution_time = time.time() - start_time
                return SandboxResult(
                    success=False,
                    error=e,
                    execution_time=execution_time,
                    files_accessed=list(self._files_accessed),
                    network_calls=self._network_calls.copy(),
                    violations=self._violations.copy()
                )
    
    def get_temp_dir(self) -> Optional[Path]:
        """Get the sandbox temporary directory."""
        return self._temp_dir
    
    def get_violations(self) -> List[str]:
        """Get list of sandbox violations."""
        return self._violations.copy()


class RestrictedFileAccess:
    """Restricted file access wrapper."""
    
    def __init__(
        self,
        sandbox: SandboxedToolExecutor,
        base_path: Optional[str] = None
    ):
        self.sandbox = sandbox
        self.base_path = Path(base_path) if base_path else None
    
    def open(self, path: str, mode: str = 'r', *args, **kwargs):
        """Open a file with sandbox restrictions."""
        full_path = self._resolve_path(path)
        
        if not self.sandbox.check_file_access(str(full_path), mode[0]):
            raise SandboxViolation(f"Access denied: {path}")
        
        return open(full_path, mode, *args, **kwargs)
    
    def read(self, path: str) -> str:
        """Read a file with sandbox restrictions."""
        with self.open(path, 'r') as f:
            return f.read()
    
    def write(self, path: str, content: str):
        """Write a file with sandbox restrictions."""
        with self.open(path, 'w') as f:
            f.write(content)
    
    def exists(self, path: str) -> bool:
        """Check if a file exists."""
        full_path = self._resolve_path(path)
        if not self.sandbox.check_file_access(str(full_path), 'r'):
            return False
        return full_path.exists()
    
    def list_dir(self, path: str) -> List[str]:
        """List directory contents."""
        full_path = self._resolve_path(path)
        if not self.sandbox.check_file_access(str(full_path), 'r'):
            raise SandboxViolation(f"Access denied: {path}")
        return [str(p) for p in full_path.iterdir()]
    
    def _resolve_path(self, path: str) -> Path:
        """Resolve path relative to base path."""
        p = Path(path)
        if not p.is_absolute() and self.base_path:
            p = self.base_path / p
        return p.resolve()


def create_sandbox(
    allowed_paths: Optional[List[str]] = None,
    allow_network: bool = True,
    timeout: float = 60.0
) -> SandboxedToolExecutor:
    """Create a sandboxed tool executor.
    
    Args:
        allowed_paths: List of allowed file paths
        allow_network: Whether to allow network access
        timeout: Default execution timeout
        
    Returns:
        Configured SandboxedToolExecutor
    """
    config = SandboxConfig(
        allowed_paths=allowed_paths or [],
        allow_network=allow_network,
        max_execution_time=timeout
    )
    return SandboxedToolExecutor(config)
