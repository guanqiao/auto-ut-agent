"""Maven project tools for running tests and analyzing coverage."""

import asyncio
import logging
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
import os
import shutil
import platform

logger = logging.getLogger(__name__)


def find_maven_executable() -> Optional[str]:
    """Find Maven executable with smart search strategy.
    
    Search order:
    1. Check PATH using shutil.which
    2. Check M2_HOME and M3_HOME environment variables
    3. Check MAVEN_HOME environment variable
    4. Windows-specific locations (Program Files, etc.)
    5. Common installation directories
    
    Returns:
        Path to mvn executable if found, None otherwise
    """
    # Step 1: Check PATH
    mvn_path = shutil.which("mvn")
    if mvn_path:
        logger.debug(f"[MavenFinder] Found mvn in PATH: {mvn_path}")
        return mvn_path
    
    # Step 2: Check M2_HOME / M3_HOME
    for env_var in ["M3_HOME", "M2_HOME"]:
        maven_home = os.environ.get(env_var)
        if maven_home:
            mvn_path = _check_maven_bin(maven_home)
            if mvn_path:
                logger.debug(f"[MavenFinder] Found mvn via {env_var}: {mvn_path}")
                return mvn_path
    
    # Step 3: Check MAVEN_HOME
    maven_home = os.environ.get("MAVEN_HOME")
    if maven_home:
        mvn_path = _check_maven_bin(maven_home)
        if mvn_path:
            logger.debug(f"[MavenFinder] Found mvn via MAVEN_HOME: {mvn_path}")
            return mvn_path
    
    # Step 4: Platform-specific search
    if platform.system() == "Windows":
        mvn_path = _find_maven_windows()
        if mvn_path:
            return mvn_path
    else:
        mvn_path = _find_maven_unix()
        if mvn_path:
            return mvn_path
    
    logger.warning("[MavenFinder] Maven not found in any standard location")
    return None


def _check_maven_bin(maven_home: str) -> Optional[str]:
    """Check if mvn exists in the bin directory of maven home."""
    home_path = Path(maven_home)
    
    if platform.system() == "Windows":
        mvn_exe = home_path / "bin" / "mvn.cmd"
        if mvn_exe.exists():
            return str(mvn_exe)
        mvn_exe = home_path / "bin" / "mvn.bat"
        if mvn_exe.exists():
            return str(mvn_exe)
    
    mvn_bin = home_path / "bin" / "mvn"
    if mvn_bin.exists():
        return str(mvn_bin)
    
    return None


def _find_maven_windows() -> Optional[str]:
    """Search for Maven in Windows-specific locations."""
    search_paths = [
        # Program Files locations
        Path(os.environ.get("ProgramFiles", "C:\\Program Files")) / "Apache" / "maven" / "bin",
        Path(os.environ.get("ProgramFiles", "C:\\Program Files")) / "Maven" / "bin",
        Path(os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)")) / "Apache" / "maven" / "bin",
        Path(os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)")) / "Maven" / "bin",
        
        # Chocolatey installation
        Path(os.environ.get("ProgramData", "C:\\ProgramData")) / "chocolatey" / "bin",
        
        # Scoop installation
        Path(os.environ.get("USERPROFILE", "C:\\Users\\Default")) / "scoop" / "shims",
        
        # Common user installations
        Path(os.environ.get("USERPROFILE", "C:\\Users\\Default")) / "apache-maven" / "bin",
        Path("C:\\") / "apache-maven" / "bin",
        Path("D:\\") / "apache-maven" / "bin",
    ]
    
    for search_path in search_paths:
        mvn_cmd = search_path / "mvn.cmd"
        if mvn_cmd.exists():
            logger.debug(f"[MavenFinder] Found mvn.cmd at {mvn_cmd}")
            return str(mvn_cmd)
        
        mvn_bat = search_path / "mvn.bat"
        if mvn_bat.exists():
            logger.debug(f"[MavenFinder] Found mvn.bat at {mvn_bat}")
            return str(mvn_bat)
    
    return None


def _find_maven_unix() -> Optional[str]:
    """Search for Maven in Unix-specific locations."""
    search_paths = [
        "/usr/share/maven/bin",
        "/usr/local/share/maven/bin",
        "/opt/maven/bin",
        "/usr/local/opt/maven/bin",  # Homebrew on macOS
        "/opt/homebrew/opt/maven/bin",  # Homebrew on Apple Silicon
        "/snap/bin",  # Snap packages
    ]
    
    for search_path in search_paths:
        mvn = Path(search_path) / "mvn"
        if mvn.exists():
            logger.debug(f"[MavenFinder] Found mvn at {mvn}")
            return str(mvn)
    
    return None


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
        self._maven_executable: Optional[str] = None
    
    def _get_maven_executable(self) -> str:
        """Get Maven executable path with configured path priority and smart fallback.
        
        Priority:
        1. Use configured Maven path from settings if available
        2. Auto-detect Maven using find_maven_executable()
        3. Fall back to "mvn" command
        
        Returns:
            Path to Maven executable, defaults to "mvn"
        """
        if self._maven_executable is None:
            try:
                from ..core.config import get_settings
                settings = get_settings()
                configured_path = settings.maven.maven_path
                
                if configured_path and configured_path.strip():
                    if Path(configured_path).exists():
                        self._maven_executable = configured_path.strip()
                        logger.info(f"[MavenRunner] Using configured Maven executable: {self._maven_executable}")
                    else:
                        logger.warning(f"[MavenRunner] Configured Maven path not found: {configured_path}, will auto-detect")
                        self._maven_executable = find_maven_executable()
                        if self._maven_executable:
                            logger.info(f"[MavenRunner] Auto-detected Maven executable: {self._maven_executable}")
                        else:
                            logger.warning("[MavenRunner] Maven executable not found, will use 'mvn' command")
                            self._maven_executable = "mvn"
                else:
                    self._maven_executable = find_maven_executable()
                    if self._maven_executable:
                        logger.info(f"[MavenRunner] Auto-detected Maven executable: {self._maven_executable}")
                    else:
                        logger.warning("[MavenRunner] Maven executable not found, will use 'mvn' command")
                        self._maven_executable = "mvn"
            except Exception as e:
                logger.exception(f"[MavenRunner] Failed to get configured Maven path, using auto-detect: {e}")
                self._maven_executable = find_maven_executable() or "mvn"
        
        return self._maven_executable
    
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
        mvn = self._get_maven_executable()
        cmd = [mvn]
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
            logger.error(f"Maven executable not found at {mvn}. Please ensure mvn is in PATH.")
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
        mvn = self._get_maven_executable()
        cmd = [mvn]
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
            logger.error(f"Maven executable not found at {mvn}. Please ensure mvn is in PATH.")
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
        mvn = self._get_maven_executable()
        cmd = [mvn]
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
            logger.error(f"Maven executable not found at {mvn}. Please ensure mvn is in PATH.")
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
        mvn = self._get_maven_executable()
        cmd = [mvn]
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
            logger.error(f"Maven executable not found at {mvn}. Please ensure mvn is in PATH.")
            return False
        except Exception as e:
            logger.exception("Error generating coverage asynchronously")
            return False
    
    def compile_project(self) -> bool:
        """Compile the project synchronously.
        
        Returns:
            True if successful, False otherwise
        """
        mvn = self._get_maven_executable()
        try:
            result = subprocess.run(
                [mvn, "compile", "-q"],
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
        mvn = self._get_maven_executable()
        try:
            process = await asyncio.create_subprocess_exec(
                mvn, "compile", "-q",
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
        
        mvn = self._get_maven_executable()
        try:
            result = subprocess.run(
                [mvn, "dependency:build-classpath", "-Dmdep.outputFile=cp.txt", "-q"],
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
        
        mvn = self._get_maven_executable()
        try:
            process = await asyncio.create_subprocess_exec(
                mvn, "dependency:build-classpath", "-Dmdep.outputFile=cp.txt", "-q",
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
    
    def resolve_dependencies(self) -> Tuple[bool, str]:
        """解析并下载所有依赖
        
        Returns:
            (success, output) 元组
        """
        mvn = self._get_maven_executable()
        try:
            result = subprocess.run(
                [mvn, "dependency:resolve", "-q"],
                cwd=self.project_path,
                capture_output=True,
                text=True,
                check=False,
                timeout=300
            )
            output = result.stderr if result.stderr else result.stdout
            return result.returncode == 0, output
        except subprocess.TimeoutExpired:
            return False, "Timeout after 300 seconds"
        except FileNotFoundError:
            return False, f"Maven executable not found at {mvn}"
        except Exception as e:
            return False, str(e)
    
    async def resolve_dependencies_async(self) -> Tuple[bool, str]:
        """解析并下载所有依赖（异步版本）
        
        Returns:
            (success, output) 元组
        """
        mvn = self._get_maven_executable()
        try:
            process = await asyncio.create_subprocess_exec(
                mvn, "dependency:resolve", "-q",
                cwd=str(self.project_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=300
                )
            except asyncio.TimeoutError:
                process.kill()
                return False, "Timeout after 300 seconds"
            
            output = stderr.decode() if stderr else stdout.decode() if stdout else ""
            return process.returncode == 0, output
        except FileNotFoundError:
            return False, f"Maven executable not found at {mvn}"
        except Exception as e:
            return False, str(e)
    
    def resolve_test_dependencies(self) -> Tuple[bool, str]:
        """解析并下载测试依赖
        
        运行 mvn test-compile -DskipTests 来下载测试依赖
        
        Returns:
            (success, output) 元组
        """
        mvn = self._get_maven_executable()
        try:
            result = subprocess.run(
                [mvn, "test-compile", "-DskipTests", "-q"],
                cwd=self.project_path,
                capture_output=True,
                text=True,
                check=False,
                timeout=300
            )
            output = result.stderr if result.stderr else result.stdout
            return result.returncode == 0, output
        except subprocess.TimeoutExpired:
            return False, "Timeout after 300 seconds"
        except FileNotFoundError:
            return False, f"Maven executable not found at {mvn}"
        except Exception as e:
            return False, str(e)
    
    async def resolve_test_dependencies_async(self) -> Tuple[bool, str]:
        """解析并下载测试依赖（异步版本）
        
        Returns:
            (success, output) 元组
        """
        mvn = self._get_maven_executable()
        try:
            process = await asyncio.create_subprocess_exec(
                mvn, "test-compile", "-DskipTests", "-q",
                cwd=str(self.project_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=300
                )
            except asyncio.TimeoutError:
                process.kill()
                return False, "Timeout after 300 seconds"
            
            output = stderr.decode() if stderr else stdout.decode() if stdout else ""
            return process.returncode == 0, output
        except FileNotFoundError:
            return False, f"Maven executable not found at {mvn}"
        except Exception as e:
            return False, str(e)
    
    def check_pom_has_test_dependencies(self) -> Dict[str, bool]:
        """检查 pom.xml 是否包含常见测试依赖
        
        Returns:
            字典，键为依赖名，值为是否包含
        """
        pom_path = self.project_path / "pom.xml"
        if not pom_path.exists():
            return {}
        
        try:
            content = pom_path.read_text(encoding='utf-8')
            return {
                "junit_jupiter": "junit-jupiter" in content,
                "mockito": "mockito" in content,
                "assertj": "assertj" in content,
                "hamcrest": "hamcrest" in content,
            }
        except Exception as e:
            logger.warning(f"[MavenRunner] Failed to read pom.xml: {e}")
            return {}
    
    def download_sources(self) -> bool:
        """下载依赖源码（可选）
        
        Returns:
            True 如果成功
        """
        mvn = self._get_maven_executable()
        try:
            result = subprocess.run(
                [mvn, "dependency:sources", "-q"],
                cwd=self.project_path,
                capture_output=True,
                text=True,
                check=False,
                timeout=300
            )
            return result.returncode == 0
        except Exception as e:
            logger.warning(f"[MavenRunner] Failed to download sources: {e}")
            return False


class CoverageAnalyzer:
    """Analyzes JaCoCo coverage reports.
    
    Supports multiple report locations and formats for robust parsing.
    """
    
    def __init__(self, project_path: str):
        """Initialize coverage analyzer.
        
        Args:
            project_path: Path to project root
        """
        self.project_path = Path(project_path)
        self.possible_report_paths = [
            self.project_path / "target" / "site" / "jacoco" / "jacoco.xml",
            self.project_path / "target" / "jacoco" / "jacoco.xml",
            self.project_path / "build" / "reports" / "jacoco" / "test" / "jacocoTestReport.xml",
            self.project_path / "target" / "site" / "jacoco" / "index.html",
        ]
    
    def _find_report_path(self) -> Optional[Path]:
        """Find JaCoCo XML report from possible locations.
        
        Returns:
            Path to report if found, None otherwise
        """
        for report_path in self.possible_report_paths:
            if report_path.exists() and report_path.suffix == '.xml':
                logger.debug(f"[CoverageAnalyzer] Found report at {report_path}")
                return report_path
        
        logger.warning("[CoverageAnalyzer] No JaCoCo XML report found in standard locations")
        return None
    
    def _get_diagnostic_info(self) -> Dict[str, Any]:
        """Get diagnostic information about coverage reports.
        
        Returns:
            Dictionary with diagnostic information
        """
        info = {
            "searched_paths": [str(p) for p in self.possible_report_paths],
            "existing_paths": [],
            "target_dir_exists": (self.project_path / "target").exists(),
            "jacoco_dir_exists": False,
        }
        
        for path in self.possible_report_paths:
            if path.exists():
                info["existing_paths"].append(str(path))
                if "jacoco" in str(path):
                    info["jacoco_dir_exists"] = True
        
        return info
    
    def parse_report(self) -> Optional[CoverageReport]:
        """Parse JaCoCo XML report with enhanced error handling.
        
        Returns:
            CoverageReport if successful, None otherwise
        """
        report_path = self._find_report_path()
        if not report_path:
            diag_info = self._get_diagnostic_info()
            logger.warning(f"[CoverageAnalyzer] Report not found. Target exists: {diag_info['target_dir_exists']}, "
                          f"Jacoco dir exists: {diag_info['jacoco_dir_exists']}")
            logger.debug(f"[CoverageAnalyzer] Searched paths: {diag_info['searched_paths']}")
            logger.debug(f"[CoverageAnalyzer] Existing paths: {diag_info['existing_paths']}")
            return None
        
        try:
            logger.info(f"[CoverageAnalyzer] Parsing report from {report_path}")
            tree = ET.parse(report_path)
            root = tree.getroot()
            
            if root.tag not in ['report', 'session']:
                logger.warning(f"[CoverageAnalyzer] Unexpected root tag: {root.tag}")
                return None
            
            counters = self._parse_counters(root)
            files = self._parse_file_coverage(root)
            
            logger.info(f"[CoverageAnalyzer] Parsed {len(files)} files, "
                       f"line coverage: {counters.get('LINE', {}).get('ratio', 0.0):.1%}")
            
            return CoverageReport(
                line_coverage=counters.get('LINE', {}).get('ratio', 0.0),
                branch_coverage=counters.get('BRANCH', {}).get('ratio', 0.0),
                instruction_coverage=counters.get('INSTRUCTION', {}).get('ratio', 0.0),
                complexity_coverage=counters.get('COMPLEXITY', {}).get('ratio', 0.0),
                method_coverage=counters.get('METHOD', {}).get('ratio', 0.0),
                class_coverage=counters.get('CLASS', {}).get('ratio', 0.0),
                files=files
            )
        except ET.ParseError as e:
            logger.error(f"[CoverageAnalyzer] XML parsing error: {e}")
            return None
        except Exception as e:
            logger.exception(f"[CoverageAnalyzer] Error parsing coverage report: {e}")
            return None
    
    def _parse_counters(self, root: ET.Element) -> Dict[str, Dict[str, Any]]:
        """Parse coverage counters from report.
        
        Args:
            root: XML root element
            
        Returns:
            Dictionary of counter types to their metrics
        """
        counters = {}
        
        for counter in root.findall('counter'):
            counter_type = counter.get('type')
            if not counter_type:
                continue
            
            try:
                missed = int(counter.get('missed', 0))
                covered = int(counter.get('covered', 0))
                total = missed + covered
                ratio = covered / total if total > 0 else 0.0
                
                counters[counter_type] = {
                    'missed': missed,
                    'covered': covered,
                    'total': total,
                    'ratio': ratio
                }
            except (ValueError, TypeError) as e:
                logger.warning(f"[CoverageAnalyzer] Failed to parse counter {counter_type}: {e}")
                continue
        
        if not counters:
            logger.debug("[CoverageAnalyzer] No counters found in report")
        
        return counters
    
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
