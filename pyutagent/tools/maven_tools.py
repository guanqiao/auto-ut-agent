"""Maven project tools for running tests and analyzing coverage."""

import asyncio
import logging
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import os

logger = logging.getLogger(__name__)


@dataclass
class CoverageReport:
    """Coverage report data."""
    line_coverage: float
    branch_coverage: float
    instruction_coverage: float
    complexity_coverage: float
    method_coverage: float
    class_coverage: float
    files: List['FileCoverage']


@dataclass
class FileCoverage:
    """Coverage data for a single file."""
    path: str
    line_coverage: float
    branch_coverage: float
    lines: List[Tuple[int, bool]]  # (line_number, is_covered)


class MavenRunner:
    """Runs Maven commands for the project.
    
    Supports both synchronous and asynchronous operations.
    """
    
    def __init__(self, project_path: str):
        """Initialize Maven runner.
        
        Args:
            project_path: Path to Maven project root
        """
        self.project_path = Path(project_path)
        self.pom_path = self.project_path / "pom.xml"
        self._classpath_cache: Optional[str] = None
    
    def is_maven_project(self) -> bool:
        """Check if the project is a Maven project."""
        return self.pom_path.exists()
    
    def run_tests(self, clean: bool = False) -> bool:
        """Run Maven tests synchronously.
        
        Args:
            clean: Whether to run clean first
            
        Returns:
            True if tests passed, False otherwise
        """
        cmd = ["mvn"]
        if clean:
            cmd.append("clean")
        cmd.extend(["test", "-q"])
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_path,
                capture_output=True,
                text=True,
                check=False
            )
            return result.returncode == 0
        except FileNotFoundError:
            logger.error("Maven not found. Please ensure mvn is in PATH.")
            return False
        except Exception as e:
            logger.exception("Error running tests")
            return False
    
    async def run_tests_async(self, clean: bool = False) -> bool:
        """Run Maven tests asynchronously.
        
        Args:
            clean: Whether to run clean first
            
        Returns:
            True if tests passed, False otherwise
        """
        cmd = ["mvn"]
        if clean:
            cmd.append("clean")
        cmd.extend(["test", "-q"])
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(self.project_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            return process.returncode == 0
        except FileNotFoundError:
            logger.error("Maven not found. Please ensure mvn is in PATH.")
            return False
        except Exception as e:
            logger.exception("Error running tests asynchronously")
            return False
    
    def generate_coverage(self, clean: bool = False) -> bool:
        """Generate JaCoCo coverage report synchronously.
        
        Args:
            clean: Whether to run clean first
            
        Returns:
            True if successful, False otherwise
        """
        cmd = ["mvn"]
        if clean:
            cmd.append("clean")
        cmd.extend(["test", "jacoco:report", "-q"])
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_path,
                capture_output=True,
                text=True,
                check=False
            )
            return result.returncode == 0
        except FileNotFoundError:
            logger.error("Maven not found. Please ensure mvn is in PATH.")
            return False
        except Exception as e:
            logger.exception("Error generating coverage")
            return False
    
    async def generate_coverage_async(self, clean: bool = False) -> bool:
        """Generate JaCoCo coverage report asynchronously.
        
        Args:
            clean: Whether to run clean first
            
        Returns:
            True if successful, False otherwise
        """
        cmd = ["mvn"]
        if clean:
            cmd.append("clean")
        cmd.extend(["test", "jacoco:report", "-q"])
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(self.project_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            return process.returncode == 0
        except FileNotFoundError:
            logger.error("Maven not found. Please ensure mvn is in PATH.")
            return False
        except Exception as e:
            logger.exception("Error generating coverage asynchronously")
            return False
    
    def compile_project(self) -> bool:
        """Compile the project synchronously.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            result = subprocess.run(
                ["mvn", "compile", "-q"],
                cwd=self.project_path,
                capture_output=True,
                text=True,
                check=False
            )
            return result.returncode == 0
        except Exception as e:
            logger.exception("Error compiling project")
            return False
    
    async def compile_project_async(self) -> bool:
        """Compile the project asynchronously.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            process = await asyncio.create_subprocess_exec(
                "mvn", "compile", "-q",
                cwd=str(self.project_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            return process.returncode == 0
        except Exception as e:
            logger.exception("Error compiling project asynchronously")
            return False
    
    def get_classpath(self, force_refresh: bool = False) -> Optional[str]:
        """Get Maven classpath (with caching).
        
        Args:
            force_refresh: Force refresh the classpath cache
            
        Returns:
            Classpath string or None if failed
        """
        if not force_refresh and self._classpath_cache:
            return self._classpath_cache
        
        try:
            result = subprocess.run(
                ["mvn", "dependency:build-classpath", "-Dmdep.outputFile=cp.txt", "-q"],
                cwd=self.project_path,
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode != 0:
                return None
            
            cp_file = self.project_path / "cp.txt"
            if cp_file.exists():
                classpath = cp_file.read_text().strip()
                self._classpath_cache = classpath
                return classpath
            
            return None
        except Exception as e:
            logger.exception("Error getting classpath")
            return None
    
    async def get_classpath_async(self, force_refresh: bool = False) -> Optional[str]:
        """Get Maven classpath asynchronously (with caching).
        
        Args:
            force_refresh: Force refresh the classpath cache
            
        Returns:
            Classpath string or None if failed
        """
        if not force_refresh and self._classpath_cache:
            return self._classpath_cache
        
        try:
            process = await asyncio.create_subprocess_exec(
                "mvn", "dependency:build-classpath", "-Dmdep.outputFile=cp.txt", "-q",
                cwd=str(self.project_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                return None
            
            cp_file = self.project_path / "cp.txt"
            if cp_file.exists():
                classpath = cp_file.read_text().strip()
                self._classpath_cache = classpath
                return classpath
            
            return None
        except Exception as e:
            logger.exception("Error getting classpath asynchronously")
            return None
    
    def invalidate_cache(self):
        """Invalidate the classpath cache."""
        self._classpath_cache = None
        logger.debug("[MavenRunner] Classpath cache invalidated")


class CoverageAnalyzer:
    """Analyzes JaCoCo coverage reports."""
    
    def __init__(self, project_path: str):
        """Initialize coverage analyzer.
        
        Args:
            project_path: Path to project root
        """
        self.project_path = Path(project_path)
        self.jacoco_xml_path = (
            self.project_path / "target" / "site" / "jacoco" / "jacoco.xml"
        )
    
    def parse_report(self) -> Optional[CoverageReport]:
        """Parse JaCoCo XML report.
        
        Returns:
            CoverageReport if successful, None otherwise
        """
        if not self.jacoco_xml_path.exists():
            return None
        
        try:
            tree = ET.parse(self.jacoco_xml_path)
            root = tree.getroot()
            
            # Parse overall counters
            counters = {}
            for counter in root.findall('counter'):
                counter_type = counter.get('type')
                missed = int(counter.get('missed', 0))
                covered = int(counter.get('covered', 0))
                total = missed + covered
                counters[counter_type] = {
                    'missed': missed,
                    'covered': covered,
                    'ratio': covered / total if total > 0 else 0.0
                }
            
            # Parse file-level coverage
            files = self._parse_file_coverage(root)
            
            return CoverageReport(
                line_coverage=counters.get('LINE', {}).get('ratio', 0.0),
                branch_coverage=counters.get('BRANCH', {}).get('ratio', 0.0),
                instruction_coverage=counters.get('INSTRUCTION', {}).get('ratio', 0.0),
                complexity_coverage=counters.get('COMPLEXITY', {}).get('ratio', 0.0),
                method_coverage=counters.get('METHOD', {}).get('ratio', 0.0),
                class_coverage=counters.get('CLASS', {}).get('ratio', 0.0),
                files=files
            )
        except Exception as e:
            logger.exception("Error parsing coverage report")
            return None
    
    def _parse_file_coverage(self, root: ET.Element) -> List[FileCoverage]:
        """Parse file-level coverage from JaCoCo report."""
        files = []
        
        for package in root.findall('.//package'):
            package_name = package.get('name', '').replace('/', '.')
            
            for sourcefile in package.findall('sourcefile'):
                filename = sourcefile.get('name')
                file_path = f"{package_name}/{filename}"
                
                # Parse line coverage
                lines = []
                covered_lines = 0
                total_lines = 0
                
                for line in sourcefile.findall('line'):
                    line_num = int(line.get('nr'))
                    mi = int(line.get('mi', 0))  # missed instructions
                    ci = int(line.get('ci', 0))  # covered instructions
                    
                    is_covered = ci > 0
                    if mi + ci > 0:
                        total_lines += 1
                        if is_covered:
                            covered_lines += 1
                    
                    lines.append((line_num, is_covered))
                
                line_ratio = covered_lines / total_lines if total_lines > 0 else 0.0
                
                files.append(FileCoverage(
                    path=file_path,
                    line_coverage=line_ratio,
                    branch_coverage=0.0,  # TODO: parse branch coverage
                    lines=lines
                ))
        
        return files
    
    def get_uncovered_lines(self, file_path: str) -> List[int]:
        """Get uncovered line numbers for a specific file.
        
        Args:
            file_path: Path to source file
            
        Returns:
            List of uncovered line numbers
        """
        report = self.parse_report()
        if not report:
            return []
        
        for file_coverage in report.files:
            if file_path in file_coverage.path or file_coverage.path.endswith(file_path):
                return [line_num for line_num, is_covered in file_coverage.lines if not is_covered]
        
        return []
    
    def get_file_coverage(self, filename: str) -> Optional[FileCoverage]:
        """Get coverage for a specific file.
        
        Args:
            filename: Name of the file
            
        Returns:
            FileCoverage if found, None otherwise
        """
        report = self.parse_report()
        if not report:
            return None
        
        for file_coverage in report.files:
            if filename in file_coverage.path:
                return file_coverage
        
        return None


class ProjectScanner:
    """Scans Maven project for Java files."""
    
    def __init__(self, project_path: str):
        """Initialize project scanner.
        
        Args:
            project_path: Path to project root
        """
        self.project_path = Path(project_path)
    
    def scan_java_files(self) -> List[str]:
        """Scan for Java source files.
        
        Returns:
            List of Java file paths
        """
        java_files = []
        src_dir = self.project_path / "src" / "main" / "java"
        
        if src_dir.exists():
            for java_file in src_dir.rglob("*.java"):
                java_files.append(str(java_file.relative_to(self.project_path)))
        
        return java_files
    
    def scan_test_files(self) -> List[str]:
        """Scan for Java test files.
        
        Returns:
            List of test file paths
        """
        test_files = []
        test_dir = self.project_path / "src" / "test" / "java"
        
        if test_dir.exists():
            for test_file in test_dir.rglob("*.java"):
                test_files.append(str(test_file.relative_to(self.project_path)))
        
        return test_files
    
    def get_source_directories(self) -> List[str]:
        """Get all source directories.
        
        Returns:
            List of source directory paths
        """
        dirs = []
        
        main_java = self.project_path / "src" / "main" / "java"
        if main_java.exists():
            dirs.append(str(main_java))
        
        test_java = self.project_path / "src" / "test" / "java"
        if test_java.exists():
            dirs.append(str(test_java))
        
        return dirs
    
    def find_test_for_class(self, class_name: str) -> Optional[str]:
        """Find test file for a given class.
        
        Args:
            class_name: Name of the class
            
        Returns:
            Path to test file if found, None otherwise
        """
        test_files = self.scan_test_files()
        
        # Common naming patterns
        possible_names = [
            f"{class_name}Test.java",
            f"{class_name}Tests.java",
            f"Test{class_name}.java",
        ]
        
        for test_file in test_files:
            if any(name in test_file for name in possible_names):
                return test_file
        
        return None
