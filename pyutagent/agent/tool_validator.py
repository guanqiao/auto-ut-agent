"""Tool validation for safe and effective execution.

This module provides tool validation capabilities:
- Pre-execution validation
- Input validation
- Dependency checking
- Safety verification
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Levels of validation."""
    NONE = auto()
    BASIC = auto()
    STANDARD = auto()
    STRICT = auto()


class ValidationSeverity(Enum):
    """Severity of validation issues."""
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()


@dataclass
class ValidationIssue:
    """A validation issue."""
    severity: ValidationSeverity
    message: str
    field: Optional[str] = None
    suggestion: Optional[str] = None
    code: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of tool validation."""
    valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    sanitized_args: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def has_errors(self) -> bool:
        return any(i.severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL) 
                   for i in self.issues)
    
    @property
    def has_warnings(self) -> bool:
        return any(i.severity == ValidationSeverity.WARNING for i in self.issues)


@dataclass
class ToolConstraints:
    """Constraints for tool execution."""
    max_input_length: int = 10000
    max_file_size: int = 10 * 1024 * 1024
    allowed_file_extensions: List[str] = field(default_factory=lambda: ['.java', '.xml', '.properties'])
    blocked_paths: List[str] = field(default_factory=lambda: ['/etc', '/sys', '/proc'])
    required_params: List[str] = field(default_factory=list)
    optional_params: List[str] = field(default_factory=list)
    timeout_seconds: float = 60.0


class InputValidator:
    """Validates tool inputs."""
    
    def __init__(self, constraints: Optional[ToolConstraints] = None):
        self.constraints = constraints or ToolConstraints()
    
    def validate_string(
        self,
        value: str,
        field_name: str,
        min_length: int = 0,
        max_length: Optional[int] = None,
        pattern: Optional[str] = None
    ) -> List[ValidationIssue]:
        """Validate a string input."""
        issues = []
        
        max_len = max_length or self.constraints.max_input_length
        
        if not isinstance(value, str):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message=f"{field_name} must be a string",
                field=field_name
            ))
            return issues
        
        if len(value) < min_length:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message=f"{field_name} must be at least {min_length} characters",
                field=field_name
            ))
        
        if len(value) > max_len:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message=f"{field_name} exceeds recommended length ({len(value)} > {max_len})",
                field=field_name,
                suggestion="Consider truncating or splitting the input"
            ))
        
        if pattern and not re.match(pattern, value):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message=f"{field_name} does not match required pattern",
                field=field_name,
                suggestion=f"Expected pattern: {pattern}"
            ))
        
        return issues
    
    def validate_file_path(
        self,
        path: str,
        must_exist: bool = True,
        check_readable: bool = True
    ) -> List[ValidationIssue]:
        """Validate a file path."""
        issues = []
        
        for blocked in self.constraints.blocked_paths:
            if path.startswith(blocked):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    message=f"Access to path is blocked: {path}",
                    suggestion="Use a different path"
                ))
                return issues
        
        file_path = Path(path)
        
        ext = file_path.suffix.lower()
        if ext and ext not in self.constraints.allowed_file_extensions:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message=f"File extension '{ext}' is not in allowed list",
                suggestion=f"Allowed extensions: {self.constraints.allowed_file_extensions}"
            ))
        
        if must_exist and not file_path.exists():
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message=f"File does not exist: {path}",
                suggestion="Check the file path"
            ))
        
        if check_readable and file_path.exists() and not file_path.is_file():
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message=f"Path is not a file: {path}"
            ))
        
        if file_path.exists():
            size = file_path.stat().st_size
            if size > self.constraints.max_file_size:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"File size ({size} bytes) exceeds limit ({self.constraints.max_file_size})",
                    suggestion="Consider processing in chunks"
                ))
        
        return issues
    
    def validate_code(
        self,
        code: str,
        language: str = "java"
    ) -> List[ValidationIssue]:
        """Validate code input."""
        issues = []
        
        if not code or not code.strip():
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message="Code cannot be empty"
            ))
            return issues
        
        if language == "java":
            issues.extend(self._validate_java_code(code))
        
        return issues
    
    def _validate_java_code(self, code: str) -> List[ValidationIssue]:
        """Validate Java code."""
        issues = []
        
        open_braces = code.count('{')
        close_braces = code.count('}')
        
        if open_braces != close_braces:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message=f"Unbalanced braces: {open_braces} open, {close_braces} close",
                suggestion="Check for missing or extra braces"
            ))
        
        open_parens = code.count('(')
        close_parens = code.count(')')
        
        if open_parens != close_parens:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message=f"Unbalanced parentheses: {open_parens} open, {close_parens} close",
                suggestion="Check for missing or extra parentheses"
            ))
        
        if 'class ' in code and 'class ' not in code.split('{')[0]:
            pass
        
        return issues


class DependencyChecker:
    """Checks tool dependencies."""
    
    def __init__(self):
        self._available_tools: Dict[str, bool] = {}
        self._tool_versions: Dict[str, str] = {}
    
    def check_tool_available(self, tool_name: str) -> Tuple[bool, Optional[str]]:
        """Check if a tool is available.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Tuple of (available, version)
        """
        import shutil
        
        path = shutil.which(tool_name)
        if path:
            return True, path
        
        return False, None
    
    def check_java_available(self) -> Tuple[bool, Optional[str]]:
        """Check if Java is available."""
        return self.check_tool_available('java')
    
    def check_maven_available(self) -> Tuple[bool, Optional[str]]:
        """Check if Maven is available."""
        return self.check_tool_available('mvn')
    
    def check_dependencies(
        self,
        required_tools: List[str]
    ) -> List[ValidationIssue]:
        """Check if all required tools are available.
        
        Args:
            required_tools: List of required tool names
            
        Returns:
            List of validation issues
        """
        issues = []
        
        for tool in required_tools:
            available, version = self.check_tool_available(tool)
            
            if not available:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message=f"Required tool not found: {tool}",
                    suggestion=f"Install {tool} or add it to PATH"
                ))
            else:
                self._available_tools[tool] = True
                self._tool_versions[tool] = version or "unknown"
        
        return issues
    
    def check_file_dependencies(
        self,
        file_path: str,
        required_files: List[str]
    ) -> List[ValidationIssue]:
        """Check if required files exist relative to a file.
        
        Args:
            file_path: Base file path
            required_files: Required relative file paths
            
        Returns:
            List of validation issues
        """
        issues = []
        base_path = Path(file_path).parent
        
        for req_file in required_files:
            full_path = base_path / req_file
            if not full_path.exists():
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"Related file not found: {req_file}",
                    suggestion=f"Create {req_file} or check the path"
                ))
        
        return issues


class SafetyChecker:
    """Checks for safety issues."""
    
    def __init__(self):
        self._dangerous_patterns = [
            (r'Runtime\.getRuntime\(\)\.exec', "Arbitrary command execution"),
            (r'ProcessBuilder', "Process execution"),
            (r'Files\.deleteIfExists', "File deletion"),
            (r'File\.delete\(\)', "File deletion"),
            (r'System\.exit', "System exit"),
            (r'Class\.forName', "Dynamic class loading"),
        ]
    
    def check_code_safety(
        self,
        code: str,
        strict: bool = False
    ) -> List[ValidationIssue]:
        """Check code for safety issues.
        
        Args:
            code: Code to check
            strict: Enable strict mode
            
        Returns:
            List of safety issues
        """
        issues = []
        
        for pattern, description in self._dangerous_patterns:
            if re.search(pattern, code):
                severity = ValidationSeverity.WARNING if not strict else ValidationSeverity.ERROR
                issues.append(ValidationIssue(
                    severity=severity,
                    message=f"Potentially unsafe code detected: {description}",
                    suggestion="Review the code for security implications"
                ))
        
        return issues
    
    def check_path_safety(
        self,
        path: str
    ) -> List[ValidationIssue]:
        """Check path for safety issues."""
        issues = []
        
        if '..' in path:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message="Path contains '..' which could allow directory traversal",
                suggestion="Use absolute paths or validate the path"
            ))
        
        if path.startswith('/'):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.INFO,
                message="Using absolute path",
                suggestion="Consider using relative paths for portability"
            ))
        
        return issues


class ToolValidator:
    """Main tool validator combining all validation strategies.
    
    Features:
    - Input validation
    - Dependency checking
    - Safety verification
    - Pre-execution validation
    """
    
    def __init__(
        self,
        constraints: Optional[ToolConstraints] = None,
        validation_level: ValidationLevel = ValidationLevel.STANDARD
    ):
        self.constraints = constraints or ToolConstraints()
        self.validation_level = validation_level
        
        self.input_validator = InputValidator(self.constraints)
        self.dependency_checker = DependencyChecker()
        self.safety_checker = SafetyChecker()
    
    def validate_tool_call(
        self,
        tool_name: str,
        args: tuple,
        kwargs: dict
    ) -> ValidationResult:
        """Validate a tool call before execution.
        
        Args:
            tool_name: Name of the tool
            args: Positional arguments
            kwargs: Keyword arguments
            
        Returns:
            ValidationResult
        """
        issues = []
        warnings = []
        
        if self.validation_level == ValidationLevel.NONE:
            return ValidationResult(valid=True, issues=issues, warnings=warnings)
        
        issues.extend(self._validate_required_params(tool_name, kwargs))
        
        for key, value in kwargs.items():
            if isinstance(value, str):
                if 'path' in key.lower() or 'file' in key.lower():
                    issues.extend(self.input_validator.validate_file_path(value, must_exist=False))
                    issues.extend(self.safety_checker.check_path_safety(value))
                elif 'code' in key.lower():
                    issues.extend(self.input_validator.validate_code(value))
        
        if self.validation_level in (ValidationLevel.STANDARD, ValidationLevel.STRICT):
            issues.extend(self._validate_tool_dependencies(tool_name))
        
        if self.validation_level == ValidationLevel.STRICT:
            for key, value in kwargs.items():
                if isinstance(value, str) and 'code' in key.lower():
                    issues.extend(self.safety_checker.check_code_safety(value, strict=True))
        
        valid = not any(i.severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL) 
                        for i in issues)
        
        warnings = [i.message for i in issues if i.severity == ValidationSeverity.WARNING]
        
        return ValidationResult(
            valid=valid,
            issues=issues,
            warnings=warnings,
            sanitized_args=self._sanitize_args(kwargs) if valid else None
        )
    
    def _validate_required_params(
        self,
        tool_name: str,
        kwargs: dict
    ) -> List[ValidationIssue]:
        """Validate required parameters are present."""
        issues = []
        
        required_params_map = {
            'generate_tests': ['class_info'],
            'compile_tests': ['test_file'],
            'run_tests': ['test_file'],
            'fix_errors': ['code', 'error'],
        }
        
        required = required_params_map.get(tool_name, [])
        
        for param in required:
            if param not in kwargs or kwargs[param] is None:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message=f"Required parameter missing: {param}",
                    field=param
                ))
        
        return issues
    
    def _validate_tool_dependencies(
        self,
        tool_name: str
    ) -> List[ValidationIssue]:
        """Validate tool dependencies are available."""
        issues = []
        
        tool_deps_map = {
            'compile_tests': ['javac', 'mvn'],
            'run_tests': ['java', 'mvn'],
            'analyze_coverage': ['mvn'],
        }
        
        deps = tool_deps_map.get(tool_name, [])
        
        for dep in deps:
            available, _ = self.dependency_checker.check_tool_available(dep)
            if not available:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"Tool dependency not found: {dep}",
                    suggestion=f"Install {dep} for full functionality"
                ))
        
        return issues
    
    def _sanitize_args(self, kwargs: dict) -> dict:
        """Sanitize arguments for safe execution."""
        sanitized = {}
        
        for key, value in kwargs.items():
            if isinstance(value, str):
                sanitized[key] = value.strip()
            elif isinstance(value, Path):
                sanitized[key] = str(value)
            else:
                sanitized[key] = value
        
        return sanitized
    
    def validate_file_access(
        self,
        file_path: str,
        mode: str = 'r'
    ) -> ValidationResult:
        """Validate file access.
        
        Args:
            file_path: Path to file
            mode: Access mode ('r', 'w', 'rw')
            
        Returns:
            ValidationResult
        """
        issues = []
        
        issues.extend(self.input_validator.validate_file_path(
            file_path,
            must_exist=('r' in mode),
            check_readable=('r' in mode)
        ))
        
        issues.extend(self.safety_checker.check_path_safety(file_path))
        
        valid = not any(i.severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL) 
                        for i in issues)
        
        return ValidationResult(
            valid=valid,
            issues=issues,
            warnings=[i.message for i in issues if i.severity == ValidationSeverity.WARNING]
        )
    
    def check_preconditions(
        self,
        tool_name: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Check preconditions for tool execution.
        
        Args:
            tool_name: Name of the tool
            context: Execution context
            
        Returns:
            List of unmet preconditions
        """
        unmet = []
        
        preconditions_map = {
            'generate_tests': [
                ("Java parser available", lambda: True),
                ("LLM client configured", lambda: context and 'llm_client' in context if context else False),
            ],
            'compile_tests': [
                ("Maven available", lambda: self.dependency_checker.check_maven_available()[0]),
                ("Test file exists", lambda: context and context.get('test_file_exists', False) if context else False),
            ],
            'run_tests': [
                ("Maven available", lambda: self.dependency_checker.check_maven_available()[0]),
                ("Tests compiled", lambda: context and context.get('tests_compiled', False) if context else False),
            ],
        }
        
        preconditions = preconditions_map.get(tool_name, [])
        
        for name, check in preconditions:
            try:
                if not check():
                    unmet.append(name)
            except Exception as e:
                unmet.append(f"{name} (check failed: {e})")
        
        return unmet


def create_tool_validator(
    validation_level: ValidationLevel = ValidationLevel.STANDARD
) -> ToolValidator:
    """Create a ToolValidator instance.
    
    Args:
        validation_level: Level of validation
        
    Returns:
        Configured ToolValidator
    """
    return ToolValidator(validation_level=validation_level)
