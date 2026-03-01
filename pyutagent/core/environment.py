"""Environment detection and validation module.

This module provides:
- Environment detector for Java, Maven, etc.
- Project environment validation
- Tool availability checking
- Automatic configuration based on environment
"""

import asyncio
import logging
import os
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from enum import Enum, auto

logger = logging.getLogger(__name__)


class EnvironmentStatus(Enum):
    """Status of an environment check."""
    AVAILABLE = auto()
    NOT_FOUND = auto()
    VERSION_MISMATCH = auto()
    ERROR = auto()


@dataclass
class ToolInfo:
    """Information about a detected tool."""
    name: str
    path: str
    version: str
    status: EnvironmentStatus
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class JavaEnvironment:
    """Java environment information."""
    java_home: str
    java_version: str
    vendor: str
    available_jdks: List[str] = field(default_factory=list)


@dataclass
class ProjectEnvironment:
    """Project environment information."""
    root_path: Path
    build_system: str
    language: str
    language_version: str
    has_tests: bool
    test_framework: Optional[str] = None


class EnvironmentDetector:
    """Detects and validates environment tools.

    This class provides automatic detection of:
    - Java SDK (version, vendor, path)
    - Maven/Gradle
    - Project structure
    - Test frameworks
    """

    def __init__(self, project_path: Optional[str] = None):
        """Initialize environment detector.

        Args:
            project_path: Project root path
        """
        self.project_path = Path(project_path) if project_path else Path.cwd()
        self._cache: Dict[str, ToolInfo] = {}

    async def detect_java(self) -> JavaEnvironment:
        """Detect Java environment.

        Returns:
            JavaEnvironment with detected information
        """
        java_home = os.environ.get("JAVA_HOME") or shutil.which("java")

        if not java_home:
            return JavaEnvironment(
                java_home="",
                java_version="",
                vendor="",
                available_jdks=[]
            )

        version_result = await self._run_command("java", ["-version"])

        version_match = re.search(r'version "(\d+\.\d+\.\d+)"', version_result.stderr or "")
        version = version_match.group(1) if version_match else "unknown"

        vendor = "unknown"
        if "Oracle" in version_result.stderr or "Java(TM)" in version_result.stderr:
            vendor = "Oracle"
        elif "OpenJDK" in version_result.stderr:
            vendor = "OpenJDK"
        elif "Eclipse" in version_result.stderr:
            vendor = "Eclipse"

        return JavaEnvironment(
            java_home=java_home,
            java_version=version,
            vendor=vendor,
            available_jdks=[java_home]
        )

    async def check_tool(self, tool_name: str) -> ToolInfo:
        """Check if a tool is available.

        Args:
            tool_name: Name of the tool (e.g., "java", "mvn", "gradle")

        Returns:
            ToolInfo with detection results
        """
        if tool_name in self._cache:
            return self._cache[tool_name]

        tool_path = shutil.which(tool_name)

        if not tool_path:
            info = ToolInfo(
                name=tool_name,
                path="",
                version="",
                status=EnvironmentStatus.NOT_FOUND
            )
            self._cache[tool_name] = info
            return info

        version_result = await self._run_command(tool_name, ["--version"])

        version = "unknown"
        if version_result.stdout:
            first_line = version_result.stdout.split("\n")[0]
            version_match = re.search(r'(\d+\.\d+\.\d+[\w.-]*)', first_line)
            if version_match:
                version = version_match.group(1)

        info = ToolInfo(
            name=tool_name,
            path=tool_path,
            version=version,
            status=EnvironmentStatus.AVAILABLE,
            details={
                "full_path": tool_path,
                "version_output": version_result.stdout or version_result.stderr
            }
        )

        self._cache[tool_name] = info
        logger.info(f"[EnvironmentDetector] Detected {tool_name}: {version} at {tool_path}")

        return info

    async def check_maven(self) -> ToolInfo:
        """Check Maven availability and version."""
        return await self.check_tool("mvn")

    async def check_gradle(self) -> ToolInfo:
        """Check Gradle availability and version."""
        return await self.check_tool("gradle")

    async def detect_project_environment(self) -> ProjectEnvironment:
        """Detect project environment.

        Returns:
            ProjectEnvironment with detected information
        """
        root = self.project_path
        build_system = "unknown"
        language = "unknown"
        language_version = "unknown"
        has_tests = False
        test_framework = None

        if (root / "pom.xml").exists():
            build_system = "maven"
            language = "java"

            try:
                import xml.etree.ElementTree as ET
                tree = ET.parse(root / "pom.xml")
                root_elem = tree.getroot()

                ns = {"m": "http://maven.apache.org/POM/4.0.0"}
                props = root_elem.find(".//m:properties", ns)
                if props is not None:
                    java_ver = props.find("m:java.version", ns)
                    if java_ver is not None:
                        language_version = java_ver.text

                for dep in root_elem.iter("{http://maven.apache.org/POM/4.0.0}dependency"):
                    artifact = dep.find("m:artifactId", ns)
                    if artifact is not None:
                        if "junit" in artifact.text.lower():
                            test_framework = "junit"
                            break

            except Exception as e:
                logger.warning(f"[EnvironmentDetector] Failed to parse pom.xml: {e}")

        elif (root / "build.gradle").exists() or (root / "build.gradle.kts").exists():
            build_system = "gradle"
            language = "java"

        test_dirs = [
            root / "src" / "test",
            root / "test",
            root / "tests"
        ]

        for test_dir in test_dirs:
            if test_dir.exists() and any(test_dir.iterdir()):
                has_tests = True
                break

        return ProjectEnvironment(
            root_path=root,
            build_system=build_system,
            language=language,
            language_version=language_version,
            has_tests=has_tests,
            test_framework=test_framework
        )

    async def validate_environment(self) -> Dict[str, Any]:
        """Validate the complete environment.

        Returns:
            Dictionary with validation results
        """
        results = {
            "status": "ok",
            "checks": {},
            "warnings": [],
            "errors": []
        }

        java = await self.detect_java()
        results["checks"]["java"] = {
            "status": java.java_version,
            "vendor": java.vendor,
            "java_home": java.java_home
        }

        if not java.java_version:
            results["errors"].append("Java not found. Please install JDK 8 or higher.")
            results["status"] = "error"

        mvn = await self.check_maven()
        results["checks"]["maven"] = {
            "status": mvn.status.name,
            "version": mvn.version,
            "path": mvn.path
        }

        if mvn.status != EnvironmentStatus.AVAILABLE:
            results["warnings"].append("Maven not found. Some features may not work.")

        project = await self.detect_project_environment()
        results["checks"]["project"] = {
            "build_system": project.build_system,
            "language": project.language,
            "has_tests": project.has_tests,
            "test_framework": project.test_framework
        }

        if project.build_system == "unknown":
            results["warnings"].append("No build system detected (Maven/Gradle).")

        return results

    async def get_recommendations(self) -> List[str]:
        """Get environment-specific recommendations.

        Returns:
            List of recommendation strings
        """
        recommendations = []

        java = await self.detect_java()
        if not java.java_version:
            recommendations.append("Install JDK 11 or higher for best compatibility")

        mvn = await self.check_maven()
        if mvn.status != EnvironmentStatus.AVAILABLE:
            recommendations.append("Install Maven 3.6+ for building Java projects")

        project = await self.detect_project_environment()
        if project.build_system == "maven" and not project.has_tests:
            recommendations.append("Add test dependencies to enable unit testing")

        return recommendations

    async def _run_command(
        self,
        cmd: str,
        args: List[str],
        timeout: int = 10
    ) -> asyncio.subprocess.Process:
        """Run a command asynchronously.

        Args:
            cmd: Command to run
            args: Command arguments
            timeout: Timeout in seconds

        Returns:
            Completed process
        """
        try:
            process = await asyncio.create_subprocess_exec(
                cmd, *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout
                )

                class Result:
                    def __init__(self, stdout, stderr, returncode):
                        self.stdout = stdout.decode("utf-8", errors="ignore") if stdout else ""
                        self.stderr = stderr.decode("utf-8", errors="ignore") if stderr else ""
                        self.returncode = returncode

                return Result(stdout, stderr, process.returncode)

            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                raise

        except FileNotFoundError:
            class Result:
                stdout = ""
                stderr = f"Command not found: {cmd}"
                returncode = -1

            return Result("", f"Command not found: {cmd}", -1)

    def clear_cache(self):
        """Clear cached tool information."""
        self._cache.clear()


async def check_environment(project_path: Optional[str] = None) -> Dict[str, Any]:
    """Quick environment check.

    Args:
        project_path: Optional project path

    Returns:
        Environment validation results
    """
    detector = EnvironmentDetector(project_path)
    return await detector.validate_environment()


def is_java_available() -> bool:
    """Quick check if Java is available.

    Returns:
        True if Java is found
    """
    return shutil.which("java") is not None


def is_maven_available() -> bool:
    """Quick check if Maven is available.

    Returns:
        True if Maven is found
    """
    return shutil.which("mvn") is not None
