"""Dependency installer for Maven projects.

This module handles the installation of Maven dependencies, including
adding dependencies to pom.xml, running mvn clean install, and verifying
the installation.
"""

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .pom_editor import PomEditor

logger = logging.getLogger(__name__)


@dataclass
class InstallResult:
    """Result of dependency installation.
    
    Attributes:
        success: Whether installation was successful
        message: Status message
        installed_deps: List of successfully installed dependencies
        failed_deps: List of failed dependencies
        backup_path: Path to pom.xml backup
    """
    success: bool
    message: str
    installed_deps: List[Dict[str, str]]
    failed_deps: List[Dict[str, str]]
    backup_path: Optional[str] = None


class DependencyInstaller:
    """Maven dependency installer.
    
    Handles the complete dependency installation workflow:
    1. Backup pom.xml
    2. Add dependencies to pom.xml
    3. Run mvn clean install
    4. Verify installation
    5. Rollback on failure
    
    Example:
        >>> installer = DependencyInstaller("/path/to/project", maven_runner)
        >>> result = await installer.install_dependencies([
        ...     {
        ...         "group_id": "org.junit.jupiter",
        ...         "artifact_id": "junit-jupiter",
        ...         "version": "5.10.0",
        ...         "scope": "test"
        ...     }
        ... ])
        >>> print(result.success)
    """
    
    def __init__(
        self,
        project_path: str,
        maven_runner: Optional[Any] = None,
        timeout: int = 600,
        progress_callback: Optional[Callable[[str, str], None]] = None
    ):
        """Initialize dependency installer.
        
        Args:
            project_path: Path to Maven project root
            maven_runner: MavenRunner instance for executing Maven commands
            timeout: Timeout for Maven operations in seconds
            progress_callback: Callback for progress updates
        """
        self.project_path = Path(project_path).resolve()
        self.maven_runner = maven_runner
        self.timeout = timeout
        self.progress_callback = progress_callback
        
        self.pom_editor = PomEditor(project_path)
        
        logger.debug(f"[DependencyInstaller] Initialized for project: {self.project_path}")
    
    async def install_dependencies(
        self,
        dependencies: List[Dict[str, str]],
        skip_tests: bool = True,
        backup: bool = True
    ) -> InstallResult:
        """Install dependencies to the project.
        
        Args:
            dependencies: List of dependency dictionaries
            skip_tests: Whether to skip tests during installation
            backup: Whether to backup pom.xml before modification
            
        Returns:
            InstallResult object
        """
        if not dependencies:
            return InstallResult(
                success=True,
                message="No dependencies to install",
                installed_deps=[],
                failed_deps=[]
            )
        
        backup_path = None
        
        try:
            if backup:
                backup_path = self.pom_editor.backup_pom(label="before_install")
                logger.info(f"[DependencyInstaller] Created backup: {backup_path}")
            
            if self.progress_callback:
                self.progress_callback("ADDING_DEPS", f"Adding {len(dependencies)} dependencies to pom.xml...")
            
            add_success, add_messages = self.pom_editor.add_dependencies(
                dependencies,
                backup=False
            )
            
            if not add_success:
                logger.warning(f"[DependencyInstaller] Some dependencies were not added: {add_messages}")
            
            if self.progress_callback:
                self.progress_callback("INSTALLING", "Running mvn clean install...")
            
            install_success, install_message = await self._run_maven_install(skip_tests)
            
            if install_success:
                installed_deps = [dep for dep in dependencies if self._is_dependency_installed(dep)]
                failed_deps = [dep for dep in dependencies if not self._is_dependency_installed(dep)]
                
                return InstallResult(
                    success=len(failed_deps) == 0,
                    message=f"Successfully installed {len(installed_deps)} dependencies" if len(failed_deps) == 0 else f"Failed to install {len(failed_deps)} dependencies",
                    installed_deps=installed_deps,
                    failed_deps=failed_deps,
                    backup_path=backup_path
                )
            else:
                if backup_path:
                    logger.warning("[DependencyInstaller] Installation failed, rolling back pom.xml")
                    self.pom_editor.restore_pom(backup_path)
                
                return InstallResult(
                    success=False,
                    message=f"Maven install failed: {install_message}",
                    installed_deps=[],
                    failed_deps=dependencies,
                    backup_path=backup_path
                )
                
        except Exception as e:
            logger.exception(f"[DependencyInstaller] Installation failed: {e}")
            
            if backup_path:
                self.pom_editor.restore_pom(backup_path)
            
            return InstallResult(
                success=False,
                message=f"Installation failed: {e}",
                installed_deps=[],
                failed_deps=dependencies,
                backup_path=backup_path
            )
    
    async def _run_maven_install(self, skip_tests: bool = True) -> Tuple[bool, str]:
        """Run mvn clean install.
        
        Args:
            skip_tests: Whether to skip tests
            
        Returns:
            Tuple of (success, message)
        """
        try:
            if self.maven_runner:
                return await self._run_with_maven_runner(skip_tests)
            else:
                return await self._run_with_subprocess(skip_tests)
                
        except Exception as e:
            logger.exception(f"[DependencyInstaller] Maven install failed: {e}")
            return False, str(e)
    
    async def _run_with_maven_runner(self, skip_tests: bool) -> Tuple[bool, str]:
        """Run Maven install using MavenRunner.
        
        Args:
            skip_tests: Whether to skip tests
            
        Returns:
            Tuple of (success, message)
        """
        try:
            if hasattr(self.maven_runner, 'compile_project_async'):
                success = await self.maven_runner.compile_project_async()
            elif hasattr(self.maven_runner, 'compile_project'):
                success = self.maven_runner.compile_project()
            else:
                return await self._run_with_subprocess(skip_tests)
            
            if success:
                return True, "Maven install successful"
            else:
                return False, "Maven install failed"
                
        except Exception as e:
            logger.error(f"[DependencyInstaller] MavenRunner failed: {e}")
            return False, str(e)
    
    async def _run_with_subprocess(self, skip_tests: bool) -> Tuple[bool, str]:
        """Run Maven install using subprocess.
        
        Args:
            skip_tests: Whether to skip tests
            
        Returns:
            Tuple of (success, message)
        """
        try:
            cmd = ["mvn", "clean", "install", "-q"]
            
            if skip_tests:
                cmd.append("-DskipTests")
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(self.project_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                return False, f"Timeout after {self.timeout} seconds"
            
            if process.returncode == 0:
                return True, "Maven install successful"
            else:
                error_msg = stderr.decode() if stderr else stdout.decode() if stdout else "Unknown error"
                return False, f"Maven install failed: {error_msg[:200]}"
                
        except FileNotFoundError:
            return False, "Maven executable not found"
        except Exception as e:
            return False, str(e)
    
    def _is_dependency_installed(self, dependency: Dict[str, str]) -> bool:
        """Check if a dependency is installed in the project.
        
        Args:
            dependency: Dependency information dictionary
            
        Returns:
            True if dependency is installed
        """
        group_id = dependency.get('group_id', '')
        artifact_id = dependency.get('artifact_id', '')
        
        return self.pom_editor.has_dependency(group_id, artifact_id)
    
    async def verify_dependencies(
        self,
        dependencies: List[Dict[str, str]]
    ) -> Tuple[bool, List[str]]:
        """Verify that dependencies are available.
        
        Args:
            dependencies: List of dependency dictionaries
            
        Returns:
            Tuple of (all_available, list of missing dependencies)
        """
        missing = []
        
        for dep in dependencies:
            if not self._is_dependency_installed(dep):
                missing.append(f"{dep.get('group_id')}:{dep.get('artifact_id')}")
        
        return len(missing) == 0, missing
    
    async def rollback(self, backup_path: str) -> bool:
        """Rollback to a previous pom.xml backup.
        
        Args:
            backup_path: Path to the backup file
            
        Returns:
            True if rollback successful
        """
        return self.pom_editor.restore_pom(backup_path)
    
    async def resolve_dependencies(self) -> Tuple[bool, str]:
        """Resolve all dependencies using mvn dependency:resolve.
        
        Returns:
            Tuple of (success, message)
        """
        try:
            if self.maven_runner and hasattr(self.maven_runner, 'resolve_dependencies_async'):
                success, output = await self.maven_runner.resolve_dependencies_async()
                return success, output
            else:
                return await self._run_dependency_resolve()
                
        except Exception as e:
            logger.exception(f"[DependencyInstaller] Dependency resolution failed: {e}")
            return False, str(e)
    
    async def _run_dependency_resolve(self) -> Tuple[bool, str]:
        """Run mvn dependency:resolve using subprocess.
        
        Returns:
            Tuple of (success, message)
        """
        try:
            process = await asyncio.create_subprocess_exec(
                "mvn", "dependency:resolve", "-q",
                cwd=str(self.project_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                return False, f"Timeout after {self.timeout} seconds"
            
            output = stderr.decode() if stderr else stdout.decode() if stdout else ""
            return process.returncode == 0, output
            
        except FileNotFoundError:
            return False, "Maven executable not found"
        except Exception as e:
            return False, str(e)
    
    async def download_sources(self) -> bool:
        """Download source jars for dependencies.
        
        Returns:
            True if successful
        """
        try:
            process = await asyncio.create_subprocess_exec(
                "mvn", "dependency:sources", "-q",
                cwd=str(self.project_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            await process.communicate()
            return process.returncode == 0
            
        except Exception as e:
            logger.warning(f"[DependencyInstaller] Failed to download sources: {e}")
            return False
    
    def get_installed_dependencies(self) -> List[Dict[str, str]]:
        """Get list of installed dependencies.
        
        Returns:
            List of dependency dictionaries
        """
        return self.pom_editor.get_dependencies()
    
    def cleanup_backups(self, keep_count: int = 10) -> int:
        """Clean up old backup files.
        
        Args:
            keep_count: Number of recent backups to keep
            
        Returns:
            Number of backups removed
        """
        return self.pom_editor.cleanup_old_backups(keep_count)
