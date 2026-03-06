"""Incremental Compiler.

This module provides incremental compilation support to avoid
recompiling unchanged files.
"""

import asyncio
import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ...tools.maven_tools import MavenRunner

logger = logging.getLogger(__name__)


@dataclass
class CompileResult:
    """Result of a compilation operation."""
    success: bool
    cached: bool = False
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    output: str = ""
    time_elapsed: float = 0.0
    file_path: Optional[str] = None


class IncrementalCompiler:
    """Incremental compiler with caching.
    
    This compiler:
    - Tracks file hashes to detect changes
    - Caches compilation results
    - Supports forced recompilation
    - Provides compilation statistics
    
    Example:
        compiler = IncrementalCompiler(project_path="/path/to/project")
        result = await compiler.compile_if_changed("src/test/java/MyTest.java")
        if result.cached:
            print("Compilation skipped (no changes)")
    """
    
    def __init__(
        self,
        project_path: str,
        maven_runner: Optional["MavenRunner"] = None,
        cache_enabled: bool = True,
        force_recompile: bool = False
    ):
        """Initialize incremental compiler.
        
        Args:
            project_path: Path to the project
            maven_runner: Optional Maven runner instance
            cache_enabled: Whether to enable caching
            force_recompile: Force recompilation even if cached
        """
        self._project_path = Path(project_path)
        self._maven_runner = maven_runner
        self._cache_enabled = cache_enabled
        self._force_recompile = force_recompile
        
        self._file_hashes: Dict[str, str] = {}
        self._compile_cache: Dict[str, CompileResult] = {}
        self._stats = {
            "total_compiles": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_time": 0.0
        }
    
    def _compute_hash(self, file_path: str) -> str:
        """Compute hash of a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            SHA256 hash of the file content
        """
        full_path = self._project_path / file_path
        if not full_path.exists():
            return ""
        
        content = full_path.read_bytes()
        return hashlib.sha256(content).hexdigest()
    
    def _compute_content_hash(self, content: str) -> str:
        """Compute hash of content.
        
        Args:
            content: Content to hash
            
        Returns:
            SHA256 hash of the content
        """
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def _is_changed(self, file_path: str) -> bool:
        """Check if file has changed since last compilation.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if file has changed
        """
        if self._force_recompile:
            return True
        
        current_hash = self._compute_hash(file_path)
        cached_hash = self._file_hashes.get(file_path)
        
        return current_hash != cached_hash
    
    async def compile_if_changed(
        self,
        test_file: str,
        force: bool = False
    ) -> CompileResult:
        """Compile file only if it has changed.
        
        Args:
            test_file: Path to the test file
            force: Force recompilation
            
        Returns:
            CompileResult with compilation status
        """
        import time
        start_time = time.time()
        
        if not force and self._cache_enabled and not self._is_changed(test_file):
            cached_result = self._compile_cache.get(test_file)
            if cached_result:
                self._stats["cache_hits"] += 1
                logger.info(f"[IncrementalCompiler] Cache hit for {test_file}")
                return CompileResult(
                    success=True,
                    cached=True,
                    file_path=test_file
                )
        
        self._stats["cache_misses"] += 1
        self._stats["total_compiles"] += 1
        
        result = await self._compile(test_file)
        result.time_elapsed = time.time() - start_time
        self._stats["total_time"] += result.time_elapsed
        
        if self._cache_enabled and result.success:
            self._file_hashes[test_file] = self._compute_hash(test_file)
            self._compile_cache[test_file] = result
        
        return result
    
    async def _compile(self, test_file: str) -> CompileResult:
        """Perform actual compilation.
        
        Args:
            test_file: Path to the test file
            
        Returns:
            CompileResult
        """
        logger.info(f"[IncrementalCompiler] Compiling {test_file}")
        
        if self._maven_runner:
            try:
                success = self._maven_runner.compile_tests()
                return CompileResult(
                    success=success,
                    cached=False,
                    file_path=test_file
                )
            except Exception as e:
                return CompileResult(
                    success=False,
                    cached=False,
                    errors=[str(e)],
                    file_path=test_file
                )
        
        try:
            from ...tools.maven_tools import find_maven_executable
            from ...tools.java_tools import find_javac_executable
            
            mvn_exe = find_maven_executable()
            if mvn_exe:
                process = await asyncio.create_subprocess_exec(
                    mvn_exe, "test-compile", "-q",
                    cwd=str(self._project_path),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                stdout, stderr = await process.communicate()
                
                errors = []
                if stderr:
                    errors = [line.strip() for line in stderr.decode('utf-8', errors='replace').split('\n') if line.strip()]
                
                return CompileResult(
                    success=process.returncode == 0,
                    cached=False,
                    errors=errors,
                    output=stdout.decode('utf-8', errors='replace') if stdout else "",
                    file_path=test_file
                )
            
            javac_exe = find_javac_executable()
            if javac_exe:
                return await self._compile_with_javac(test_file, javac_exe)
            
            return CompileResult(
                success=False,
                cached=False,
                errors=["No compiler found (Maven or javac)"],
                file_path=test_file
            )
            
        except Exception as e:
            logger.exception(f"[IncrementalCompiler] Compilation failed: {e}")
            return CompileResult(
                success=False,
                cached=False,
                errors=[str(e)],
                file_path=test_file
            )
    
    async def _compile_with_javac(
        self,
        test_file: str,
        javac_exe: str
    ) -> CompileResult:
        """Compile using javac directly.
        
        Args:
            test_file: Path to the test file
            javac_exe: Path to javac executable
            
        Returns:
            CompileResult
        """
        full_path = self._project_path / test_file
        
        classpath = await self._get_classpath()
        
        output_dir = self._project_path / "target" / "test-classes"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        process = await asyncio.create_subprocess_exec(
            javac_exe,
            "-cp", classpath,
            "-d", str(output_dir),
            str(full_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        errors = []
        if stderr:
            errors = [line.strip() for line in stderr.decode('utf-8', errors='replace').split('\n') if line.strip()]
        
        return CompileResult(
            success=process.returncode == 0,
            cached=False,
            errors=errors,
            output=stdout.decode('utf-8', errors='replace') if stdout else "",
            file_path=test_file
        )
    
    async def _get_classpath(self) -> str:
        """Get the classpath for compilation.
        
        Returns:
            Classpath string
        """
        try:
            from ...tools.maven_tools import find_maven_executable
            
            mvn_exe = find_maven_executable()
            if mvn_exe:
                process = await asyncio.create_subprocess_exec(
                    mvn_exe, "dependency:build-classpath",
                    "-Dmdep.outputFile=cp.txt", "-q",
                    cwd=str(self._project_path),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                await process.communicate()
                
                cp_file = self._project_path / "cp.txt"
                if cp_file.exists():
                    return cp_file.read_text(encoding='utf-8').strip()
        except Exception as e:
            logger.warning(f"[IncrementalCompiler] Failed to get classpath: {e}")
        
        target_classes = self._project_path / "target" / "classes"
        target_test_classes = self._project_path / "target" / "test-classes"
        
        parts = []
        if target_classes.exists():
            parts.append(str(target_classes))
        if target_test_classes.exists():
            parts.append(str(target_test_classes))
        
        return ";".join(parts) if parts else "."
    
    def invalidate_cache(self, file_path: Optional[str] = None) -> None:
        """Invalidate compilation cache.
        
        Args:
            file_path: Optional specific file to invalidate, or None for all
        """
        if file_path:
            self._file_hashes.pop(file_path, None)
            self._compile_cache.pop(file_path, None)
            logger.info(f"[IncrementalCompiler] Invalidated cache for {file_path}")
        else:
            self._file_hashes.clear()
            self._compile_cache.clear()
            logger.info("[IncrementalCompiler] Invalidated all caches")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get compilation statistics.
        
        Returns:
            Statistics dictionary
        """
        total = self._stats["cache_hits"] + self._stats["cache_misses"]
        hit_rate = self._stats["cache_hits"] / total if total > 0 else 0.0
        
        return {
            **self._stats,
            "cache_hit_rate": hit_rate,
            "cached_files": len(self._file_hashes)
        }
    
    def set_force_recompile(self, force: bool) -> None:
        """Set force recompile flag.
        
        Args:
            force: Whether to force recompilation
        """
        self._force_recompile = force
        logger.info(f"[IncrementalCompiler] Force recompile set to {force}")
