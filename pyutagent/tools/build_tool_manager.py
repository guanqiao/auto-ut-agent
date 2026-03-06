"""Build tool management for multi-build-system support.

This module provides automatic detection and unified interface for
Maven, Gradle, and Bazel build systems.
"""

import logging
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)


class BuildToolType(Enum):
    """Supported build tool types."""
    MAVEN = auto()
    GRADLE = auto()
    BAZEL = auto()
    UNKNOWN = auto()


@dataclass
class BuildToolInfo:
    """Information about detected build tool."""
    tool_type: BuildToolType
    version: Optional[str] = None
    config_file: Optional[Path] = None
    wrapper_available: bool = False
    executable_path: Optional[str] = None


@dataclass
class TestResult:
    """Unified test result format."""
    success: bool
    stdout: str = ""
    stderr: str = ""
    test_count: int = 0
    failure_count: int = 0
    error_count: int = 0
    skipped_count: int = 0
    execution_time_ms: int = 0
    report_path: Optional[Path] = None
    raw_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CoverageResult:
    """Unified coverage result format."""
    success: bool
    line_coverage: float = 0.0
    branch_coverage: float = 0.0
    method_coverage: float = 0.0
    class_coverage: float = 0.0
    report_path: Optional[Path] = None
    uncovered_lines: Dict[str, List[int]] = field(default_factory=dict)
    raw_data: Dict[str, Any] = field(default_factory=dict)


class BuildToolRunner(ABC):
    """Abstract base class for build tool runners."""
    
    def __init__(self, project_path: Path, tool_info: BuildToolInfo):
        self.project_path = project_path
        self.tool_info = tool_info
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    async def run_tests(
        self,
        test_class: Optional[str] = None,
        test_method: Optional[str] = None,
        additional_args: Optional[List[str]] = None
    ) -> TestResult:
        """Run tests."""
        pass
    
    @abstractmethod
    async def compile_project(self, clean: bool = False) -> Tuple[bool, str]:
        """Compile the project."""
        pass
    
    @abstractmethod
    async def generate_coverage(
        self,
        test_class: Optional[str] = None
    ) -> CoverageResult:
        """Generate coverage report."""
        pass
    
    @abstractmethod
    def get_classpath(self) -> str:
        """Get project classpath."""
        pass
    
    @abstractmethod
    def get_test_report_path(self) -> Path:
        """Get path to test reports."""
        pass
    
    @abstractmethod
    def get_coverage_report_path(self) -> Path:
        """Get path to coverage reports."""
        pass
    
    def _run_command(
        self,
        command: List[str],
        cwd: Optional[Path] = None,
        timeout: int = 300
    ) -> Tuple[int, str, str]:
        """Run a shell command.
        
        Args:
            command: Command and arguments
            cwd: Working directory
            timeout: Timeout in seconds
            
        Returns:
            Tuple of (returncode, stdout, stderr)
        """
        try:
            self.logger.debug(f"Running command: {' '.join(command)}")
            
            result = subprocess.run(
                command,
                cwd=cwd or self.project_path,
                capture_output=True,
                text=True,
                timeout=timeout,
                encoding='utf-8',
                errors='replace'
            )
            
            return result.returncode, result.stdout, result.stderr
            
        except subprocess.TimeoutExpired:
            self.logger.error(f"Command timed out after {timeout}s: {' '.join(command)}")
            return -1, "", f"Timeout after {timeout} seconds"
        except Exception as e:
            self.logger.exception(f"Failed to run command: {e}")
            return -1, "", str(e)


class MavenRunner(BuildToolRunner):
    """Maven build tool runner."""
    
    def __init__(self, project_path: Path, tool_info: BuildToolInfo):
        super().__init__(project_path, tool_info)
        self._executable = self._get_executable()
    
    def _get_executable(self) -> str:
        """Get Maven executable."""
        if self.tool_info.wrapper_available:
            wrapper = self.project_path / "mvnw"
            if wrapper.exists():
                return str(wrapper)

        if self.tool_info.executable_path:
            return self.tool_info.executable_path

        # Fallback: try to find mvn in PATH
        import shutil
        mvn_path = shutil.which("mvn")
        if mvn_path:
            return mvn_path

        raise FileNotFoundError(
            "Maven executable not found. Please install Maven or configure 'maven_path' in settings. "
            "See: https://maven.apache.org/install.html"
        )
    
    async def run_tests(
        self,
        test_class: Optional[str] = None,
        test_method: Optional[str] = None,
        additional_args: Optional[List[str]] = None
    ) -> TestResult:
        """Run Maven tests."""
        cmd = [self._executable, "test", "-q"]
        
        if test_class:
            cmd.extend(["-Dtest=" + test_class])
            if test_method:
                cmd[-1] += "#" + test_method
        
        if additional_args:
            cmd.extend(additional_args)
        
        returncode, stdout, stderr = self._run_command(cmd)
        
        # Parse test results
        test_count, failure_count, error_count, skipped_count = self._parse_test_summary(stdout + stderr)
        
        return TestResult(
            success=returncode == 0,
            stdout=stdout,
            stderr=stderr,
            test_count=test_count,
            failure_count=failure_count,
            error_count=error_count,
            skipped_count=skipped_count,
            report_path=self.get_test_report_path(),
            raw_data={"returncode": returncode}
        )
    
    def _parse_test_summary(self, output: str) -> Tuple[int, int, int, int]:
        """Parse Maven test summary from output."""
        test_count = failure_count = error_count = skipped_count = 0
        
        # Pattern: Tests run: X, Failures: Y, Errors: Z, Skipped: W
        import re
        match = re.search(r'Tests run:\s*(\d+),\s*Failures:\s*(\d+),\s*Errors:\s*(\d+),\s*Skipped:\s*(\d+)', output)
        if match:
            test_count = int(match.group(1))
            failure_count = int(match.group(2))
            error_count = int(match.group(3))
            skipped_count = int(match.group(4))
        
        return test_count, failure_count, error_count, skipped_count
    
    async def compile_project(self, clean: bool = False) -> Tuple[bool, str]:
        """Compile Maven project."""
        cmd = [self._executable]
        if clean:
            cmd.append("clean")
        cmd.extend(["compile", "-q"])
        
        returncode, stdout, stderr = self._run_command(cmd)
        return returncode == 0, stdout + stderr
    
    async def compile_tests_async(self) -> Tuple[bool, str]:
        """Compile test classes asynchronously.
        
        Returns:
            Tuple of (success, output)
        """
        cmd = [self._executable, "test-compile", "-q"]
        returncode, stdout, stderr = self._run_command(cmd)
        output = stderr if stderr else stdout
        return returncode == 0, output
    
    async def generate_coverage(
        self,
        test_class: Optional[str] = None
    ) -> CoverageResult:
        """Generate JaCoCo coverage report."""
        cmd = [self._executable, "jacoco:report", "-q"]
        
        if test_class:
            cmd.insert(1, f"-Dtest={test_class}")
        
        returncode, stdout, stderr = self._run_command(cmd)
        
        # Parse coverage report
        coverage = self._parse_coverage_report()
        
        return CoverageResult(
            success=returncode == 0 and coverage.get("line_coverage", 0) > 0,
            line_coverage=coverage.get("line_coverage", 0.0),
            branch_coverage=coverage.get("branch_coverage", 0.0),
            method_coverage=coverage.get("method_coverage", 0.0),
            class_coverage=coverage.get("class_coverage", 0.0),
            report_path=self.get_coverage_report_path(),
            uncovered_lines=coverage.get("uncovered_lines", {}),
            raw_data=coverage.get("raw_data", {})
        )
    
    def _parse_coverage_report(self) -> Dict[str, Any]:
        """Parse JaCoCo XML report."""
        report_path = self.get_coverage_report_path()
        if not report_path or not report_path.exists():
            return {}
        
        try:
            tree = ET.parse(report_path)
            root = tree.getroot()
            
            # Extract coverage metrics
            coverage = {"raw_data": {}}
            
            for counter in root.findall('.//counter'):
                counter_type = counter.get('type', '').lower()
                missed = int(counter.get('missed', 0))
                covered = int(counter.get('covered', 0))
                total = missed + covered
                
                if total > 0:
                    rate = covered / total
                    if counter_type == 'line':
                        coverage["line_coverage"] = rate
                    elif counter_type == 'branch':
                        coverage["branch_coverage"] = rate
                    elif counter_type == 'method':
                        coverage["method_coverage"] = rate
                    elif counter_type == 'class':
                        coverage["class_coverage"] = rate
            
            return coverage
            
        except Exception as e:
            self.logger.error(f"Failed to parse coverage report: {e}")
            return {}
    
    def get_classpath(self) -> str:
        """Get Maven classpath."""
        cmd = [self._executable, "dependency:build-classpath", "-q", "-Dmdep.outputFile=/dev/stdout"]
        returncode, stdout, stderr = self._run_command(cmd)
        
        if returncode == 0:
            return stdout.strip()
        return ""
    
    def get_test_report_path(self) -> Path:
        """Get Surefire report path."""
        return self.project_path / "target" / "surefire-reports"
    
    def get_coverage_report_path(self) -> Path:
        """Get JaCoCo report path."""
        return self.project_path / "target" / "site" / "jacoco" / "jacoco.xml"


class GradleRunner(BuildToolRunner):
    """Gradle build tool runner."""
    
    def __init__(self, project_path: Path, tool_info: BuildToolInfo):
        super().__init__(project_path, tool_info)
        self._executable = self._get_executable()
    
    def _get_executable(self) -> str:
        """Get Gradle executable."""
        if self.tool_info.wrapper_available:
            wrapper = self.project_path / "gradlew"
            if wrapper.exists():
                return str(wrapper)
        
        if self.tool_info.executable_path:
            return self.tool_info.executable_path
        
        return "gradle"
    
    async def run_tests(
        self,
        test_class: Optional[str] = None,
        test_method: Optional[str] = None,
        additional_args: Optional[List[str]] = None
    ) -> TestResult:
        """Run Gradle tests."""
        cmd = [self._executable, "test", "--quiet"]
        
        if test_class:
            # Gradle test filtering: --tests "ClassName.methodName"
            test_filter = test_class
            if test_method:
                test_filter += f".{test_method}"
            cmd.extend(["--tests", test_filter])
        
        if additional_args:
            cmd.extend(additional_args)
        
        returncode, stdout, stderr = self._run_command(cmd)
        
        # Parse test results
        test_count, failure_count, error_count, skipped_count = self._parse_test_summary(stdout + stderr)
        
        return TestResult(
            success=returncode == 0,
            stdout=stdout,
            stderr=stderr,
            test_count=test_count,
            failure_count=failure_count,
            error_count=error_count,
            skipped_count=skipped_count,
            report_path=self.get_test_report_path(),
            raw_data={"returncode": returncode}
        )
    
    def _parse_test_summary(self, output: str) -> Tuple[int, int, int, int]:
        """Parse Gradle test summary from output."""
        test_count = failure_count = error_count = skipped_count = 0
        
        import re
        # Pattern: X tests completed, Y failed, Z skipped
        match = re.search(r'(\d+)\s+tests?\s+completed[,;]?\s*(\d+)\s+failed[,;]?\s*(\d+)\s+skipped', output, re.IGNORECASE)
        if match:
            test_count = int(match.group(1))
            failure_count = int(match.group(2))
            skipped_count = int(match.group(3))
        
        return test_count, failure_count, error_count, skipped_count
    
    async def compile_project(self, clean: bool = False) -> Tuple[bool, str]:
        """Compile Gradle project."""
        cmd = [self._executable]
        if clean:
            cmd.append("clean")
        cmd.extend(["compileJava", "--quiet"])
        
        returncode, stdout, stderr = self._run_command(cmd)
        return returncode == 0, stdout + stderr
    
    async def compile_tests_async(self) -> Tuple[bool, str]:
        """Compile test classes asynchronously.
        
        Returns:
            Tuple of (success, output)
        """
        cmd = [self._executable, "compileTestJava", "--quiet"]
        returncode, stdout, stderr = self._run_command(cmd)
        output = stderr if stderr else stdout
        return returncode == 0, output
    
    async def generate_coverage(
        self,
        test_class: Optional[str] = None
    ) -> CoverageResult:
        """Generate JaCoCo coverage report with Gradle."""
        cmd = [self._executable, "jacocoTestReport", "--quiet"]
        
        if test_class:
            cmd.extend(["--tests", test_class])
        
        returncode, stdout, stderr = self._run_command(cmd)
        
        # Parse coverage report
        coverage = self._parse_coverage_report()
        
        return CoverageResult(
            success=returncode == 0 and coverage.get("line_coverage", 0) > 0,
            line_coverage=coverage.get("line_coverage", 0.0),
            branch_coverage=coverage.get("branch_coverage", 0.0),
            method_coverage=coverage.get("method_coverage", 0.0),
            class_coverage=coverage.get("class_coverage", 0.0),
            report_path=self.get_coverage_report_path(),
            uncovered_lines=coverage.get("uncovered_lines", {}),
            raw_data=coverage.get("raw_data", {})
        )
    
    def _parse_coverage_report(self) -> Dict[str, Any]:
        """Parse Gradle JaCoCo XML report."""
        report_path = self.get_coverage_report_path()
        if not report_path or not report_path.exists():
            return {}
        
        try:
            tree = ET.parse(report_path)
            root = tree.getroot()
            
            coverage = {"raw_data": {}}
            
            for counter in root.findall('.//counter'):
                counter_type = counter.get('type', '').lower()
                missed = int(counter.get('missed', 0))
                covered = int(counter.get('covered', 0))
                total = missed + covered
                
                if total > 0:
                    rate = covered / total
                    if counter_type == 'line':
                        coverage["line_coverage"] = rate
                    elif counter_type == 'branch':
                        coverage["branch_coverage"] = rate
                    elif counter_type == 'method':
                        coverage["method_coverage"] = rate
                    elif counter_type == 'class':
                        coverage["class_coverage"] = rate
            
            return coverage
            
        except Exception as e:
            self.logger.error(f"Failed to parse coverage report: {e}")
            return {}
    
    def get_classpath(self) -> str:
        """Get Gradle classpath."""
        cmd = [self._executable, "printClasspath", "--quiet"]
        returncode, stdout, stderr = self._run_command(cmd)
        
        if returncode == 0:
            return stdout.strip()
        return ""
    
    def get_test_report_path(self) -> Path:
        """Get Gradle test report path."""
        return self.project_path / "build" / "test-results" / "test"
    
    def get_coverage_report_path(self) -> Path:
        """Get Gradle JaCoCo report path."""
        return self.project_path / "build" / "reports" / "jacoco" / "test" / "jacocoTestReport.xml"


class BazelRunner(BuildToolRunner):
    """Bazel build tool runner."""
    
    def __init__(self, project_path: Path, tool_info: BuildToolInfo):
        super().__init__(project_path, tool_info)
        self._executable = self.tool_info.executable_path or "bazel"
    
    async def run_tests(
        self,
        test_class: Optional[str] = None,
        test_method: Optional[str] = None,
        additional_args: Optional[List[str]] = None
    ) -> TestResult:
        """Run Bazel tests."""
        cmd = [self._executable, "test", "//..."]
        
        if test_class:
            # Bazel test filtering
            cmd = [self._executable, "test", f"//... --test_filter={test_class}"]
            if test_method:
                cmd[-1] += f"#{test_method}"
        
        if additional_args:
            cmd.extend(additional_args)
        
        returncode, stdout, stderr = self._run_command(cmd)
        
        return TestResult(
            success=returncode == 0,
            stdout=stdout,
            stderr=stderr,
            report_path=self.get_test_report_path(),
            raw_data={"returncode": returncode}
        )
    
    async def compile_project(self, clean: bool = False) -> Tuple[bool, str]:
        """Compile Bazel project."""
        if clean:
            self._run_command([self._executable, "clean"])
        
        cmd = [self._executable, "build", "//..."]
        returncode, stdout, stderr = self._run_command(cmd)
        
        return returncode == 0, stdout + stderr
    
    async def compile_tests_async(self) -> Tuple[bool, str]:
        """Compile test classes asynchronously.
        
        Returns:
            Tuple of (success, output)
        """
        cmd = [self._executable, "build", "//...", "--javacopts=-source 11 -target 11"]
        returncode, stdout, stderr = self._run_command(cmd)
        output = stderr if stderr else stdout
        return returncode == 0, output
    
    async def generate_coverage(
        self,
        test_class: Optional[str] = None
    ) -> CoverageResult:
        """Generate coverage report with Bazel."""
        cmd = [self._executable, "coverage", "//..."]
        
        if test_class:
            cmd = [self._executable, "coverage", f"--test_filter={test_class}", "//..."]
        
        returncode, stdout, stderr = self._run_command(cmd)
        
        return CoverageResult(
            success=returncode == 0,
            report_path=self.get_coverage_report_path()
        )
    
    def get_classpath(self) -> str:
        """Get Bazel classpath."""
        # Bazel doesn't have a simple classpath command
        return ""
    
    def get_test_report_path(self) -> Path:
        """Get Bazel test report path."""
        return self.project_path / "bazel-testlogs"
    
    def get_coverage_report_path(self) -> Path:
        """Get Bazel coverage report path."""
        return self.project_path / "bazel-out" / "_coverage"


class BuildToolManager:
    """Manager for automatic build tool detection and unified interface."""
    
    def __init__(self, project_path: str):
        """Initialize build tool manager.
        
        Args:
            project_path: Path to the project root
        """
        self.project_path = Path(project_path).resolve()
        self._detected_tool: Optional[BuildToolInfo] = None
        self._runner: Optional[BuildToolRunner] = None
        
        logger.info(f"[BuildToolManager] Initialized for project: {self.project_path}")
    
    def detect_build_tool(self) -> BuildToolInfo:
        """Automatically detect the build tool used in the project.
        
        Returns:
            Detected build tool information
        """
        if self._detected_tool:
            return self._detected_tool
        
        # Check for build files in order of priority
        detection_order = [
            (self._detect_maven, BuildToolType.MAVEN),
            (self._detect_gradle, BuildToolType.GRADLE),
            (self._detect_bazel, BuildToolType.BAZEL),
        ]
        
        for detect_func, tool_type in detection_order:
            tool_info = detect_func()
            if tool_info:
                self._detected_tool = tool_info
                logger.info(f"[BuildToolManager] Detected {tool_type.name} build tool")
                return tool_info
        
        # No build tool detected
        self._detected_tool = BuildToolInfo(
            tool_type=BuildToolType.UNKNOWN,
            config_file=None
        )
        logger.warning("[BuildToolManager] No build tool detected")
        return self._detected_tool
    
    def _detect_maven(self) -> Optional[BuildToolInfo]:
        """Detect Maven build tool."""
        pom_xml = self.project_path / "pom.xml"

        if not pom_xml.exists():
            return None

        # Check for Maven wrapper
        wrapper = self.project_path / "mvnw"
        wrapper_available = wrapper.exists()

        # Get configured Maven path
        configured_maven_path = None
        try:
            from ..core.config import get_settings
            settings = get_settings()
            if settings.maven.maven_path and settings.maven.maven_path.strip():
                configured_path = settings.maven.maven_path.strip()
                if Path(configured_path).exists():
                    configured_maven_path = configured_path
                    logger.info(f"[BuildToolManager] Using configured Maven path: {configured_path}")
        except Exception as e:
            logger.debug(f"[BuildToolManager] Failed to get configured Maven path: {e}")

        # Determine executable path
        if wrapper_available:
            executable_path = str(wrapper)
        elif configured_maven_path:
            executable_path = configured_maven_path
        else:
            # Check if mvn is available in PATH
            mvn_path = self._find_executable_in_path("mvn")
            if mvn_path:
                executable_path = mvn_path
            else:
                logger.warning("[BuildToolManager] Maven not found in PATH. Please install Maven or configure maven_path in settings.")
                return BuildToolInfo(
                    tool_type=BuildToolType.MAVEN,
                    version=None,
                    config_file=pom_xml,
                    wrapper_available=False,
                    executable_path=None
                )

        # Try to get version
        version = None
        try:
            result = subprocess.run(
                [executable_path, "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                # Parse version from output
                import re
                match = re.search(r'Apache Maven (\d+\.\d+\.\d+)', result.stdout)
                if match:
                    version = match.group(1)
        except Exception:
            pass

        return BuildToolInfo(
            tool_type=BuildToolType.MAVEN,
            version=version,
            config_file=pom_xml,
            wrapper_available=wrapper_available,
            executable_path=executable_path
        )

    def _find_executable_in_path(self, executable: str) -> Optional[str]:
        """Find executable in system PATH.

        Args:
            executable: Executable name (e.g., 'mvn', 'gradle')

        Returns:
            Full path to executable if found, None otherwise
        """
        import shutil
        return shutil.which(executable)
    
    def _detect_gradle(self) -> Optional[BuildToolInfo]:
        """Detect Gradle build tool."""
        build_gradle = self.project_path / "build.gradle"
        build_gradle_kts = self.project_path / "build.gradle.kts"
        
        config_file = None
        if build_gradle.exists():
            config_file = build_gradle
        elif build_gradle_kts.exists():
            config_file = build_gradle_kts
        
        if not config_file:
            return None
        
        # Check for Gradle wrapper
        wrapper = self.project_path / "gradlew"
        wrapper_available = wrapper.exists()
        
        # Try to get version
        version = None
        try:
            result = subprocess.run(
                ["gradle", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                import re
                match = re.search(r'Gradle (\d+\.\d+\.\d+)', result.stdout)
                if match:
                    version = match.group(1)
        except Exception:
            pass
        
        return BuildToolInfo(
            tool_type=BuildToolType.GRADLE,
            version=version,
            config_file=config_file,
            wrapper_available=wrapper_available,
            executable_path="gradle" if not wrapper_available else str(wrapper)
        )
    
    def _detect_bazel(self) -> Optional[BuildToolInfo]:
        """Detect Bazel build tool."""
        workspace = self.project_path / "WORKSPACE"
        workspace_bazel = self.project_path / "WORKSPACE.bazel"
        
        config_file = None
        if workspace.exists():
            config_file = workspace
        elif workspace_bazel.exists():
            config_file = workspace_bazel
        
        if not config_file:
            return None
        
        # Try to get version
        version = None
        try:
            result = subprocess.run(
                ["bazel", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                import re
                match = re.search(r'bazel (\d+\.\d+\.\d+)', result.stdout)
                if match:
                    version = match.group(1)
        except Exception:
            pass
        
        return BuildToolInfo(
            tool_type=BuildToolType.BAZEL,
            version=version,
            config_file=config_file,
            wrapper_available=False,
            executable_path="bazel"
        )
    
    def get_runner(self) -> Optional[BuildToolRunner]:
        """Get the appropriate build tool runner.
        
        Returns:
            Build tool runner instance
        """
        if self._runner:
            return self._runner
        
        tool_info = self.detect_build_tool()
        
        if tool_info.tool_type == BuildToolType.MAVEN:
            self._runner = MavenRunner(self.project_path, tool_info)
        elif tool_info.tool_type == BuildToolType.GRADLE:
            self._runner = GradleRunner(self.project_path, tool_info)
        elif tool_info.tool_type == BuildToolType.BAZEL:
            self._runner = BazelRunner(self.project_path, tool_info)
        else:
            logger.error("[BuildToolManager] Cannot create runner for unknown build tool")
            return None
        
        return self._runner
    
    def is_supported(self) -> bool:
        """Check if the project uses a supported build tool.
        
        Returns:
            True if a supported build tool is detected
        """
        tool_info = self.detect_build_tool()
        return tool_info.tool_type != BuildToolType.UNKNOWN
    
    def get_tool_type(self) -> BuildToolType:
        """Get the detected build tool type.
        
        Returns:
            Build tool type enum
        """
        tool_info = self.detect_build_tool()
        return tool_info.tool_type


# Convenience functions

def detect_project_build_tool(project_path: str) -> BuildToolType:
    """Quick detection of project build tool.
    
    Args:
        project_path: Path to project root
        
    Returns:
        Detected build tool type
    """
    manager = BuildToolManager(project_path)
    return manager.get_tool_type()


def get_project_runner(project_path: str) -> Optional[BuildToolRunner]:
    """Get build tool runner for project.
    
    Args:
        project_path: Path to project root
        
    Returns:
        Build tool runner or None
    """
    manager = BuildToolManager(project_path)
    return manager.get_runner()
