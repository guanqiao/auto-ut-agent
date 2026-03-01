"""Static analysis tool integration for code quality assessment.

This module provides integration with SpotBugs and PMD for deep
static analysis of Java code to identify potential bugs and
quality issues before test generation.
"""

import logging
import subprocess
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
import json

logger = logging.getLogger(__name__)


class AnalysisToolType(Enum):
    """Supported static analysis tools."""
    SPOTBUGS = auto()
    PMD = auto()
    CHECKSTYLE = auto()


class BugSeverity(Enum):
    """Bug severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class BugInstance:
    """Represents a bug or issue found by static analysis."""
    bug_type: str
    category: str
    severity: BugSeverity
    message: str
    class_name: str
    method_name: Optional[str] = None
    line_number: Optional[int] = None
    source_line: Optional[str] = None
    confidence: str = "medium"  # high, medium, low
    pattern: Optional[str] = None  # Bug pattern code
    explanation: Optional[str] = None
    suggestion: Optional[str] = None


@dataclass
class AnalysisResult:
    """Result of static analysis."""
    tool_type: AnalysisToolType
    success: bool
    bug_count: int = 0
    bugs: List[BugInstance] = field(default_factory=list)
    files_analyzed: int = 0
    analysis_time_ms: int = 0
    report_path: Optional[Path] = None
    raw_output: str = ""
    error_message: Optional[str] = None
    
    def get_bugs_by_severity(self, severity: BugSeverity) -> List[BugInstance]:
        """Get bugs filtered by severity."""
        return [b for b in self.bugs if b.severity == severity]
    
    def get_bugs_by_category(self, category: str) -> List[BugInstance]:
        """Get bugs filtered by category."""
        return [b for b in self.bugs if b.category == category]
    
    def get_unique_bug_types(self) -> Set[str]:
        """Get set of unique bug types."""
        return set(b.bug_type for b in self.bugs)


@dataclass
class TestQualityAssessment:
    """Assessment of test code quality."""
    test_class: str
    issues: List[BugInstance] = field(default_factory=list)
    assertions_count: int = 0
    test_methods_count: int = 0
    coverage_potential: float = 0.0
    quality_score: float = 0.0
    recommendations: List[str] = field(default_factory=list)


class StaticAnalysisRunner(ABC):
    """Abstract base class for static analysis tool runners."""
    
    def __init__(self, project_path: Path):
        self.project_path = project_path
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    async def analyze(
        self,
        target_files: Optional[List[str]] = None,
        include_tests: bool = False
    ) -> AnalysisResult:
        """Run static analysis."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the tool is available."""
        pass
    
    def _run_command(
        self,
        command: List[str],
        cwd: Optional[Path] = None,
        timeout: int = 300
    ) -> Tuple[int, str, str]:
        """Run a shell command."""
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
            self.logger.error(f"Command timed out after {timeout}s")
            return -1, "", f"Timeout after {timeout} seconds"
        except Exception as e:
            self.logger.exception(f"Failed to run command: {e}")
            return -1, "", str(e)


class SpotBugsRunner(StaticAnalysisRunner):
    """SpotBugs static analysis runner."""
    
    # Bug category to severity mapping
    CATEGORY_SEVERITY = {
        "CORRECTNESS": BugSeverity.HIGH,
        "MT_CORRECTNESS": BugSeverity.HIGH,
        "BAD_PRACTICE": BugSeverity.MEDIUM,
        "PERFORMANCE": BugSeverity.MEDIUM,
        "SECURITY": BugSeverity.CRITICAL,
        "STYLE": BugSeverity.LOW,
        "EXPERIMENTAL": BugSeverity.LOW,
    }
    
    def __init__(self, project_path: Path):
        super().__init__(project_path)
        self._executable = self._find_executable()
    
    def _find_executable(self) -> Optional[str]:
        """Find SpotBugs executable."""
        # Check for Maven plugin
        if (self.project_path / "pom.xml").exists():
            return "mvn"
        
        # Check for Gradle plugin
        if (self.project_path / "build.gradle").exists() or \
           (self.project_path / "build.gradle.kts").exists():
            return "gradle"
        
        # Check for standalone SpotBugs
        for cmd in ["spotbugs", "spotbugs-cli"]:
            try:
                result = subprocess.run(
                    [cmd, "-version"],
                    capture_output=True,
                    timeout=5
                )
                if result.returncode == 0:
                    return cmd
            except:
                pass
        
        return None
    
    def is_available(self) -> bool:
        """Check if SpotBugs is available."""
        return self._executable is not None
    
    async def analyze(
        self,
        target_files: Optional[List[str]] = None,
        include_tests: bool = False
    ) -> AnalysisResult:
        """Run SpotBugs analysis."""
        if not self.is_available():
            return AnalysisResult(
                tool_type=AnalysisToolType.SPOTBUGS,
                success=False,
                error_message="SpotBugs not available"
            )
        
        report_path = self.project_path / "target" / "spotbugs.xml"
        if self._executable == "gradle":
            report_path = self.project_path / "build" / "reports" / "spotbugs" / "main.xml"
        
        # Run analysis
        if self._executable == "mvn":
            cmd = ["mvn", "spotbugs:spotbugs", "-q"]
        elif self._executable == "gradle":
            cmd = ["gradle", "spotbugsMain", "--quiet"]
        else:
            # Standalone SpotBugs
            cmd = [self._executable, "-textui", "-xml", str(report_path), str(self.project_path)]
        
        returncode, stdout, stderr = self._run_command(cmd)
        
        # Parse results
        bugs = []
        if report_path.exists():
            bugs = self._parse_report(report_path)
        
        return AnalysisResult(
            tool_type=AnalysisToolType.SPOTBUGS,
            success=returncode == 0 or len(bugs) > 0,
            bug_count=len(bugs),
            bugs=bugs,
            report_path=report_path,
            raw_output=stdout + stderr
        )
    
    def _parse_report(self, report_path: Path) -> List[BugInstance]:
        """Parse SpotBugs XML report."""
        bugs = []
        
        try:
            tree = ET.parse(report_path)
            root = tree.getroot()
            
            for bug in root.findall('.//BugInstance'):
                bug_type = bug.get('type', 'Unknown')
                category = bug.get('category', 'UNKNOWN')
                priority = int(bug.get('priority', 2))
                
                # Map priority to severity
                severity = self._priority_to_severity(priority, category)
                
                # Get class and method info
                class_elem = bug.find('Class')
                class_name = class_elem.get('classname', 'Unknown') if class_elem else 'Unknown'
                
                method_elem = bug.find('Method')
                method_name = method_elem.get('name') if method_elem else None
                
                source_elem = bug.find('SourceLine')
                line_number = int(source_elem.get('start')) if source_elem else None
                
                # Get message
                message_elem = bug.find('ShortMessage')
                message = message_elem.text if message_elem else bug_type
                
                # Get detailed explanation
                long_msg = bug.find('LongMessage')
                explanation = long_msg.text if long_msg else None
                
                bugs.append(BugInstance(
                    bug_type=bug_type,
                    category=category,
                    severity=severity,
                    message=message,
                    class_name=class_name,
                    method_name=method_name,
                    line_number=line_number,
                    pattern=bug_type,
                    explanation=explanation,
                    suggestion=self._get_suggestion(bug_type)
                ))
                
        except Exception as e:
            self.logger.error(f"Failed to parse SpotBugs report: {e}")
        
        return bugs
    
    def _priority_to_severity(self, priority: int, category: str) -> BugSeverity:
        """Convert SpotBugs priority to severity."""
        # Priority: 1=high, 2=medium, 3=low
        if category == "SECURITY":
            return BugSeverity.CRITICAL
        elif priority == 1:
            return BugSeverity.HIGH
        elif priority == 2:
            return BugSeverity.MEDIUM
        else:
            return BugSeverity.LOW
    
    def _get_suggestion(self, bug_type: str) -> Optional[str]:
        """Get suggestion for bug type."""
        suggestions = {
            "NP_NULL_ON_SOME_PATH": "Add null check before using the variable",
            "URF_UNREAD_FIELD": "Remove unused field or use it",
            "UUF_UNUSED_FIELD": "Remove unused field",
            "DLS_DEAD_LOCAL_STORE": "Remove unnecessary assignment",
            "ICAST_INTEGER_MULTIPLY_CAST_TO_LONG": "Cast to long before multiplication to avoid overflow",
            "EI_EXPOSE_REP": "Return a copy of the mutable object instead of the original",
            "EI_EXPOSE_REP2": "Store a copy of the mutable object instead of the original",
        }
        return suggestions.get(bug_type)


class PMDRunner(StaticAnalysisRunner):
    """PMD static analysis runner."""
    
    # Rule priority to severity mapping
    PRIORITY_SEVERITY = {
        1: BugSeverity.CRITICAL,
        2: BugSeverity.HIGH,
        3: BugSeverity.MEDIUM,
        4: BugSeverity.LOW,
        5: BugSeverity.INFO,
    }
    
    def __init__(self, project_path: Path):
        super().__init__(project_path)
        self._executable = self._find_executable()
    
    def _find_executable(self) -> Optional[str]:
        """Find PMD executable."""
        # Check for Maven plugin
        if (self.project_path / "pom.xml").exists():
            return "mvn"
        
        # Check for Gradle plugin
        if (self.project_path / "build.gradle").exists() or \
           (self.project_path / "build.gradle.kts").exists():
            return "gradle"
        
        # Check for standalone PMD
        for cmd in ["pmd", "pmd-bin"]:
            try:
                result = subprocess.run(
                    [cmd, "--version"],
                    capture_output=True,
                    timeout=5
                )
                if result.returncode == 0:
                    return cmd
            except:
                pass
        
        return None
    
    def is_available(self) -> bool:
        """Check if PMD is available."""
        return self._executable is not None
    
    async def analyze(
        self,
        target_files: Optional[List[str]] = None,
        include_tests: bool = False
    ) -> AnalysisResult:
        """Run PMD analysis."""
        if not self.is_available():
            return AnalysisResult(
                tool_type=AnalysisToolType.PMD,
                success=False,
                error_message="PMD not available"
            )
        
        report_path = self.project_path / "target" / "pmd.xml"
        if self._executable == "gradle":
            report_path = self.project_path / "build" / "reports" / "pmd" / "main.xml"
        
        # Run analysis
        if self._executable == "mvn":
            cmd = ["mvn", "pmd:pmd", "-q"]
        elif self._executable == "gradle":
            cmd = ["gradle", "pmdMain", "--quiet"]
        else:
            # Standalone PMD
            cmd = [
                self._executable,
                "check",
                "-d", str(self.project_path / "src" / "main" / "java"),
                "-R", "rulesets/java/quickstart.xml",
                "-f", "xml",
                "-r", str(report_path)
            ]
        
        returncode, stdout, stderr = self._run_command(cmd)
        
        # Parse results
        bugs = []
        if report_path.exists():
            bugs = self._parse_report(report_path)
        
        return AnalysisResult(
            tool_type=AnalysisToolType.PMD,
            success=True,  # PMD returns non-zero when bugs found
            bug_count=len(bugs),
            bugs=bugs,
            report_path=report_path,
            raw_output=stdout + stderr
        )
    
    def _parse_report(self, report_path: Path) -> List[BugInstance]:
        """Parse PMD XML report."""
        bugs = []
        
        try:
            tree = ET.parse(report_path)
            root = tree.getroot()
            
            for file_elem in root.findall('.//file'):
                filename = file_elem.get('name', 'Unknown')
                class_name = Path(filename).stem
                
                for violation in file_elem.findall('violation'):
                    rule = violation.get('rule', 'Unknown')
                    priority = int(violation.get('priority', 3))
                    beginline = int(violation.get('beginline', 0))
                    
                    severity = self.PRIORITY_SEVERITY.get(priority, BugSeverity.MEDIUM)
                    message = violation.text.strip() if violation.text else rule
                    
                    bugs.append(BugInstance(
                        bug_type=rule,
                        category="CODE_STYLE",
                        severity=severity,
                        message=message,
                        class_name=class_name,
                        line_number=beginline,
                        pattern=rule,
                        explanation=message,
                        suggestion=self._get_pmd_suggestion(rule)
                    ))
                    
        except Exception as e:
            self.logger.error(f"Failed to parse PMD report: {e}")
        
        return bugs
    
    def _get_pmd_suggestion(self, rule: str) -> Optional[str]:
        """Get suggestion for PMD rule."""
        suggestions = {
            "UnusedImports": "Remove unused import statements",
            "UnusedLocalVariable": "Remove unused variable or use it",
            "UnusedPrivateField": "Remove unused private field",
            "UnusedPrivateMethod": "Remove unused private method",
            "SystemPrintln": "Use a logger instead of System.out.println",
            "AvoidPrintStackTrace": "Use a logger to log exceptions",
            "EmptyCatchBlock": "Handle the exception or log it",
            "EmptyIfStmt": "Remove empty if statement or add logic",
            "EmptyWhileStmt": "Remove empty while statement or add logic",
        }
        return suggestions.get(rule)


class StaticAnalysisManager:
    """Manager for static analysis tools."""
    
    def __init__(self, project_path: str):
        """Initialize static analysis manager.
        
        Args:
            project_path: Path to project root
        """
        self.project_path = Path(project_path).resolve()
        self.spotbugs = SpotBugsRunner(self.project_path)
        self.pmd = PMDRunner(self.project_path)
        
        logger.info(f"[StaticAnalysisManager] Initialized for {self.project_path}")
    
    async def run_all_analysis(
        self,
        target_files: Optional[List[str]] = None,
        include_tests: bool = False
    ) -> Dict[AnalysisToolType, AnalysisResult]:
        """Run all available static analysis tools.
        
        Args:
            target_files: Specific files to analyze
            include_tests: Whether to include test files
            
        Returns:
            Dictionary of tool type to analysis result
        """
        results = {}
        
        # Run SpotBugs if available
        if self.spotbugs.is_available():
            logger.info("[StaticAnalysisManager] Running SpotBugs analysis")
            results[AnalysisToolType.SPOTBUGS] = await self.spotbugs.analyze(
                target_files, include_tests
            )
        
        # Run PMD if available
        if self.pmd.is_available():
            logger.info("[StaticAnalysisManager] Running PMD analysis")
            results[AnalysisToolType.PMD] = await self.pmd.analyze(
                target_files, include_tests
            )
        
        return results
    
    async def analyze_test_quality(
        self,
        test_file_path: str,
        test_code: str
    ) -> TestQualityAssessment:
        """Analyze quality of generated test code.
        
        Args:
            test_file_path: Path to test file
            test_code: Test code content
            
        Returns:
            Test quality assessment
        """
        assessment = TestQualityAssessment(
            test_class=Path(test_file_path).stem
        )
        
        # Count test methods
        import re
        test_methods = re.findall(r'@Test[\s\S]*?void\s+(\w+)\s*\(', test_code)
        assessment.test_methods_count = len(test_methods)
        
        # Count assertions
        assertions = re.findall(r'assert\w+\s*\(', test_code)
        assessment.assertions_count = len(assertions)
        
        # Check for common test quality issues
        issues = []
        
        # Check for empty test methods
        for method_match in re.finditer(r'@Test[\s\S]*?void\s+(\w+)\s*\([^)]*\)\s*\{([^}]*)\}', test_code):
            method_name = method_match.group(1)
            method_body = method_match.group(2).strip()
            
            if not method_body or method_body == "":
                issues.append(BugInstance(
                    bug_type="EMPTY_TEST_METHOD",
                    category="TEST_QUALITY",
                    severity=BugSeverity.HIGH,
                    message=f"Test method '{method_name}' is empty",
                    class_name=assessment.test_class,
                    method_name=method_name,
                    suggestion="Add test logic or remove the empty test"
                ))
        
        # Check for tests without assertions
        for method_name in test_methods:
            # Find the method body
            pattern = rf'@Test[\s\S]*?void\s+{method_name}\s*\([^)]*\)\s*\{{([\s\S]*?)\}}'
            match = re.search(pattern, test_code)
            if match:
                method_body = match.group(1)
                if 'assert' not in method_body and 'verify' not in method_body:
                    issues.append(BugInstance(
                        bug_type="NO_ASSERTION",
                        category="TEST_QUALITY",
                        severity=BugSeverity.HIGH,
                        message=f"Test method '{method_name}' has no assertions",
                        class_name=assessment.test_class,
                        method_name=method_name,
                        suggestion="Add assertions to verify test outcomes"
                    ))
        
        # Check for proper exception testing
        if 'try {' in test_code and 'catch' in test_code:
            if 'fail(' not in test_code:
                issues.append(BugInstance(
                    bug_type="IMPROPER_EXCEPTION_TEST",
                    category="TEST_QUALITY",
                    severity=BugSeverity.MEDIUM,
                    message="Exception test without fail() call",
                    class_name=assessment.test_class,
                    suggestion="Use assertThrows() instead of try-catch, or add fail() after the expected exception line"
                ))
        
        assessment.issues = issues
        
        # Calculate quality score
        if assessment.test_methods_count > 0:
            assertions_per_test = assessment.assertions_count / assessment.test_methods_count
            issue_penalty = len([i for i in issues if i.severity in (BugSeverity.HIGH, BugSeverity.CRITICAL)]) * 0.2
            
            assessment.quality_score = min(1.0, max(0.0, 
                (assertions_per_test / 3.0) - issue_penalty
            ))
        
        # Generate recommendations
        if assessment.assertions_count < assessment.test_methods_count:
            assessment.recommendations.append(
                "Add more assertions to improve test coverage"
            )
        
        if len(test_methods) > 10:
            assessment.recommendations.append(
                "Consider splitting large test class into smaller focused test classes"
            )
        
        return assessment
    
    def get_available_tools(self) -> List[AnalysisToolType]:
        """Get list of available analysis tools.
        
        Returns:
            List of available tool types
        """
        available = []
        
        if self.spotbugs.is_available():
            available.append(AnalysisToolType.SPOTBUGS)
        
        if self.pmd.is_available():
            available.append(AnalysisToolType.PMD)
        
        return available
    
    def merge_analysis_results(
        self,
        results: Dict[AnalysisToolType, AnalysisResult]
    ) -> List[BugInstance]:
        """Merge results from multiple tools, removing duplicates.
        
        Args:
            results: Dictionary of analysis results
            
        Returns:
            Merged list of unique bugs
        """
        all_bugs = []
        seen = set()
        
        for tool_type, result in results.items():
            for bug in result.bugs:
                # Create unique key for deduplication
                key = (bug.class_name, bug.method_name, bug.line_number, bug.bug_type)
                
                if key not in seen:
                    seen.add(key)
                    all_bugs.append(bug)
        
        # Sort by severity
        severity_order = {
            BugSeverity.CRITICAL: 0,
            BugSeverity.HIGH: 1,
            BugSeverity.MEDIUM: 2,
            BugSeverity.LOW: 3,
            BugSeverity.INFO: 4
        }
        all_bugs.sort(key=lambda b: severity_order.get(b.severity, 5))
        
        return all_bugs


# Convenience functions

async def analyze_project(
    project_path: str,
    tools: Optional[List[AnalysisToolType]] = None
) -> Dict[AnalysisToolType, AnalysisResult]:
    """Quick project analysis with all or specified tools.
    
    Args:
        project_path: Path to project
        tools: Specific tools to run (None = all available)
        
    Returns:
        Analysis results
    """
    manager = StaticAnalysisManager(project_path)
    
    if tools:
        results = {}
        for tool in tools:
            if tool == AnalysisToolType.SPOTBUGS and manager.spotbugs.is_available():
                results[tool] = await manager.spotbugs.analyze()
            elif tool == AnalysisToolType.PMD and manager.pmd.is_available():
                results[tool] = await manager.pmd.analyze()
        return results
    
    return await manager.run_all_analysis()


def check_test_quality(test_code: str, test_class_name: str) -> TestQualityAssessment:
    """Quick check of test code quality.
    
    Args:
        test_code: Test code content
        test_class_name: Name of test class
        
    Returns:
        Quality assessment
    """
    manager = StaticAnalysisManager(".")
    # Note: This is a synchronous wrapper for the async method
    import asyncio
    return asyncio.run(manager.analyze_test_quality(
        f"{test_class_name}.java",
        test_code
    ))
