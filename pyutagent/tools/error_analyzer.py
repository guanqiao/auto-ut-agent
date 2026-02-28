"""Java compilation error analyzer."""

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Any
from pathlib import Path


class ErrorType(Enum):
    """Types of compilation errors."""
    IMPORT_ERROR = auto()
    SYMBOL_NOT_FOUND = auto()
    TYPE_MISMATCH = auto()
    SYNTAX_ERROR = auto()
    GENERIC_TYPE_ERROR = auto()
    ACCESS_MODIFIER_ERROR = auto()
    METHOD_NOT_FOUND = auto()
    VARIABLE_NOT_FOUND = auto()
    CONSTRUCTOR_NOT_FOUND = auto()
    PACKAGE_NOT_FOUND = auto()
    ANNOTATION_ERROR = auto()
    STATIC_REFERENCE_ERROR = auto()
    UNKNOWN = auto()


@dataclass
class CompilationError:
    """Represents a single compilation error."""
    error_type: ErrorType
    message: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    column_number: Optional[int] = None
    error_token: Optional[str] = None
    suggestions: List[str] = field(default_factory=list)
    fix_hint: str = ""


@dataclass
class ErrorAnalysis:
    """Result of error analysis."""
    errors: List[CompilationError]
    summary: str
    fix_strategy: str
    priority: int  # 1 = highest, 5 = lowest


class CompilationErrorAnalyzer:
    """Analyzes Java compilation errors and suggests fixes."""
    
    # Error patterns for different types of errors
    ERROR_PATTERNS = {
        ErrorType.IMPORT_ERROR: [
            r"package ([\w.]+) does not exist",
            r"cannot find symbol\s+symbol:\s+class ([\w.]+)",
            r"import ([\w.]+) cannot be resolved",
        ],
        ErrorType.SYMBOL_NOT_FOUND: [
            r"cannot find symbol\s+symbol:\s+(\w+)\s+([\w.]+)",
            r"cannot find symbol",
        ],
        ErrorType.TYPE_MISMATCH: [
            r"incompatible types",
            r"cannot convert from ([\w.<>\[\]]+) to ([\w.<>\[\]]+)",
        ],
        ErrorType.SYNTAX_ERROR: [
            r"';' expected",
            r"'\)' expected",
            r"'}' expected",
            r"'{' expected",
            r"illegal start of expression",
        ],
        ErrorType.GENERIC_TYPE_ERROR: [
            r"type argument ([\w.]+) is not within bounds of type-variable ([\w]+)",
            r"cannot infer type arguments",
        ],
        ErrorType.ACCESS_MODIFIER_ERROR: [
            r"([\w.]+) has ([\w]+) access in ([\w.]+)",
            r"attempting to assign weaker access privileges",
        ],
        ErrorType.METHOD_NOT_FOUND: [
            r"cannot find symbol\s+symbol:\s+method ([\w]+)",
            r"method does not override or implement a method",
        ],
        ErrorType.VARIABLE_NOT_FOUND: [
            r"cannot find symbol\s+symbol:\s+variable ([\w]+)",
        ],
        ErrorType.CONSTRUCTOR_NOT_FOUND: [
            r"cannot find symbol\s+symbol:\s+constructor ([\w.]+)",
            r"constructor ([\w.]+) in class ([\w.]+) cannot be applied",
        ],
        ErrorType.PACKAGE_NOT_FOUND: [
            r"package ([\w.]+) does not exist",
        ],
        ErrorType.ANNOTATION_ERROR: [
            r"annotation ([\w.]+) is missing value for the element ([\w]+)",
            r"([\w.]+) is not a repeatable annotation type",
        ],
        ErrorType.STATIC_REFERENCE_ERROR: [
            r"non-static variable ([\w]+) cannot be referenced from a static context",
            r"non-static method ([\w]+) cannot be referenced from a static context",
        ],
    }
    
    def __init__(self):
        """Initialize error analyzer."""
        self.error_patterns = self._compile_patterns()
    
    def _compile_patterns(self) -> Dict[ErrorType, List[re.Pattern]]:
        """Compile error patterns."""
        compiled = {}
        for error_type, patterns in self.ERROR_PATTERNS.items():
            compiled[error_type] = [re.compile(p, re.IGNORECASE) for p in patterns]
        return compiled
    
    def analyze(self, compiler_output: str, test_file: Optional[str] = None) -> ErrorAnalysis:
        """Analyze compilation errors from compiler output.
        
        Args:
            compiler_output: Raw compiler error output
            test_file: Path to the test file being compiled
            
        Returns:
            ErrorAnalysis with parsed errors and fix strategy
        """
        errors = []
        lines = compiler_output.split('\n')
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Look for error line pattern
            error_match = self._match_error_line(line)
            if error_match:
                error = self._parse_error(error_match, lines, i, test_file)
                errors.append(error)
                
                # Skip to next error
                while i < len(lines) and not self._is_new_error(lines[i]):
                    i += 1
                continue
            
            i += 1
        
        # Generate analysis summary
        summary = self._generate_summary(errors)
        fix_strategy = self._determine_fix_strategy(errors)
        priority = self._calculate_priority(errors)
        
        return ErrorAnalysis(
            errors=errors,
            summary=summary,
            fix_strategy=fix_strategy,
            priority=priority
        )
    
    def _match_error_line(self, line: str) -> Optional[re.Match]:
        """Match an error line pattern."""
        # Pattern: file.java:line:column: error: message
        # Also supports: file.java:line:error: message (without column)
        pattern = r'(.+\.java):(\d+)(?::\d+)?\s*:\s*(error|warning):\s*(.+)'
        return re.match(pattern, line)
    
    def _is_new_error(self, line: str) -> bool:
        """Check if line starts a new error."""
        return bool(self._match_error_line(line))
    
    def _parse_error(
        self,
        match: re.Match,
        lines: List[str],
        start_idx: int,
        test_file: Optional[str]
    ) -> CompilationError:
        """Parse a single error from compiler output."""
        file_path = match.group(1)
        line_number = int(match.group(2))
        message = match.group(4)
        
        # Determine error type
        error_type = self._classify_error(message)
        
        # Extract error token
        error_token = self._extract_error_token(message, error_type)
        
        # Generate suggestions
        suggestions = self._generate_suggestions(error_type, message, error_token)
        
        # Generate fix hint
        fix_hint = self._generate_fix_hint(error_type, message, error_token)
        
        return CompilationError(
            error_type=error_type,
            message=message,
            file_path=file_path if file_path != test_file else test_file,
            line_number=line_number,
            error_token=error_token,
            suggestions=suggestions,
            fix_hint=fix_hint
        )
    
    def _classify_error(self, message: str) -> ErrorType:
        """Classify error type from message."""
        message_lower = message.lower()
        
        for error_type, patterns in self.error_patterns.items():
            for pattern in patterns:
                if pattern.search(message):
                    return error_type
        
        # Check for common keywords
        if 'import' in message_lower:
            return ErrorType.IMPORT_ERROR
        elif 'symbol' in message_lower:
            return ErrorType.SYMBOL_NOT_FOUND
        elif 'incompatible' in message_lower or 'convert' in message_lower:
            return ErrorType.TYPE_MISMATCH
        elif 'expected' in message_lower:
            return ErrorType.SYNTAX_ERROR
        
        return ErrorType.UNKNOWN
    
    def _extract_error_token(self, message: str, error_type: ErrorType) -> Optional[str]:
        """Extract the problematic token from error message."""
        # Try to extract symbol name
        patterns = [
            r"symbol:\s+(?:class|method|variable|constructor)\s+(\w+)",
            r"cannot find symbol\s+symbol:\s+(\w+)",
            r"package ([\w.]+) does not exist",
            r"class ([\w.]+)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None
    
    def _generate_suggestions(
        self,
        error_type: ErrorType,
        message: str,
        error_token: Optional[str]
    ) -> List[str]:
        """Generate fix suggestions based on error type."""
        suggestions = []
        
        if error_type == ErrorType.IMPORT_ERROR:
            suggestions.append(f"Add missing import for {error_token}")
            suggestions.append(f"Check if {error_token} is in the classpath")
            suggestions.append("Verify the package name is correct")
            
        elif error_type == ErrorType.SYMBOL_NOT_FOUND:
            suggestions.append(f"Check if {error_token} is defined")
            suggestions.append(f"Verify the correct import for {error_token}")
            suggestions.append("Check for typos in the symbol name")
            
        elif error_type == ErrorType.TYPE_MISMATCH:
            suggestions.append("Check the expected type and actual type")
            suggestions.append("Add appropriate type casting if needed")
            suggestions.append("Verify method return types")
            
        elif error_type == ErrorType.SYNTAX_ERROR:
            suggestions.append("Check for missing semicolons, braces, or parentheses")
            suggestions.append("Verify proper nesting of code blocks")
            
        elif error_type == ErrorType.METHOD_NOT_FOUND:
            suggestions.append(f"Check if method {error_token} exists in the class")
            suggestions.append("Verify method parameters match the signature")
            suggestions.append("Check for correct import of the class")
            
        elif error_type == ErrorType.VARIABLE_NOT_FOUND:
            suggestions.append(f"Declare variable {error_token} before use")
            suggestions.append("Check variable scope and visibility")
            suggestions.append("Verify the variable name is spelled correctly")
            
        elif error_type == ErrorType.STATIC_REFERENCE_ERROR:
            suggestions.append("Create an instance of the class to access non-static members")
            suggestions.append("Or make the member static if appropriate")
            
        elif error_type == ErrorType.ACCESS_MODIFIER_ERROR:
            suggestions.append("Check the access level of the referenced member")
            suggestions.append("Consider using reflection or changing access modifier")
            
        else:
            suggestions.append("Review the error message carefully")
            suggestions.append("Check the line number indicated in the error")
        
        return suggestions
    
    def _generate_fix_hint(
        self,
        error_type: ErrorType,
        message: str,
        error_token: Optional[str]
    ) -> str:
        """Generate a specific fix hint for the error."""
        if error_type == ErrorType.IMPORT_ERROR and error_token:
            return f"Add import statement: import {error_token};"
        
        elif error_type == ErrorType.SYMBOL_NOT_FOUND and error_token:
            return f"Define or import {error_token} before using it"
        
        elif error_type == ErrorType.METHOD_NOT_FOUND and error_token:
            return f"Verify the method signature for {error_token}() matches the class definition"
        
        elif error_type == ErrorType.VARIABLE_NOT_FOUND and error_token:
            return f"Initialize {error_token} with appropriate value before use"
        
        elif error_type == ErrorType.TYPE_MISMATCH:
            return "Ensure types match or add explicit casting"
        
        elif error_type == ErrorType.SYNTAX_ERROR:
            return "Fix syntax issues (missing semicolons, braces, etc.)"
        
        elif error_type == ErrorType.STATIC_REFERENCE_ERROR:
            return "Access instance members through an object instance"
        
        return "Review and fix the indicated error"
    
    def _generate_summary(self, errors: List[CompilationError]) -> str:
        """Generate a summary of errors."""
        if not errors:
            return "No compilation errors found"
        
        error_counts = {}
        for error in errors:
            error_counts[error.error_type] = error_counts.get(error.error_type, 0) + 1
        
        summary_parts = [f"Found {len(errors)} compilation error(s):"]
        for error_type, count in sorted(error_counts.items(), key=lambda x: -x[1]):
            summary_parts.append(f"  - {count} {error_type.name.replace('_', ' ').title()}")
        
        return "\n".join(summary_parts)
    
    def _determine_fix_strategy(self, errors: List[CompilationError]) -> str:
        """Determine the overall fix strategy."""
        if not errors:
            return "No fixes needed"
        
        # Check for import errors first (usually easiest to fix)
        import_errors = [e for e in errors if e.error_type == ErrorType.IMPORT_ERROR]
        if import_errors:
            return "Fix import errors first, then address other issues"
        
        # Check for syntax errors
        syntax_errors = [e for e in errors if e.error_type == ErrorType.SYNTAX_ERROR]
        if syntax_errors:
            return "Fix syntax errors first to ensure code structure is valid"
        
        # Check for symbol errors
        symbol_errors = [e for e in errors if e.error_type in [
            ErrorType.SYMBOL_NOT_FOUND,
            ErrorType.METHOD_NOT_FOUND,
            ErrorType.VARIABLE_NOT_FOUND
        ]]
        if symbol_errors:
            return "Define missing symbols or add correct imports"
        
        return "Address errors in order of priority"
    
    def _calculate_priority(self, errors: List[CompilationError]) -> int:
        """Calculate fix priority (1 = highest)."""
        if not errors:
            return 5
        
        # Priority based on error types
        priority_scores = {
            ErrorType.SYNTAX_ERROR: 1,
            ErrorType.IMPORT_ERROR: 1,
            ErrorType.PACKAGE_NOT_FOUND: 1,
            ErrorType.SYMBOL_NOT_FOUND: 2,
            ErrorType.METHOD_NOT_FOUND: 2,
            ErrorType.VARIABLE_NOT_FOUND: 2,
            ErrorType.CONSTRUCTOR_NOT_FOUND: 2,
            ErrorType.TYPE_MISMATCH: 3,
            ErrorType.GENERIC_TYPE_ERROR: 3,
            ErrorType.ACCESS_MODIFIER_ERROR: 3,
            ErrorType.STATIC_REFERENCE_ERROR: 3,
            ErrorType.ANNOTATION_ERROR: 4,
            ErrorType.UNKNOWN: 4,
        }
        
        min_priority = 5
        for error in errors:
            priority = priority_scores.get(error.error_type, 4)
            min_priority = min(min_priority, priority)
        
        return min_priority
    
    def get_fix_prompt_context(self, analysis: ErrorAnalysis) -> str:
        """Generate context for LLM fix prompt."""
        context_parts = [
            analysis.summary,
            "",
            "Fix Strategy: " + analysis.fix_strategy,
            "",
            "Detailed Errors:"
        ]
        
        for i, error in enumerate(analysis.errors[:10], 1):  # Limit to 10 errors
            context_parts.append(f"\n{i}. {error.error_type.name}")
            context_parts.append(f"   Message: {error.message}")
            if error.line_number:
                context_parts.append(f"   Line: {error.line_number}")
            if error.error_token:
                context_parts.append(f"   Token: {error.error_token}")
            if error.fix_hint:
                context_parts.append(f"   Hint: {error.fix_hint}")
            if error.suggestions:
                context_parts.append(f"   Suggestions: {', '.join(error.suggestions[:2])}")
        
        if len(analysis.errors) > 10:
            context_parts.append(f"\n... and {len(analysis.errors) - 10} more errors")
        
        return "\n".join(context_parts)


class ErrorFixGenerator:
    """Generates specific fixes for compilation errors."""
    
    def __init__(self, analyzer: CompilationErrorAnalyzer):
        """Initialize fix generator.
        
        Args:
            analyzer: Error analyzer instance
        """
        self.analyzer = analyzer
    
    def generate_fixes(
        self,
        test_code: str,
        compiler_output: str,
        class_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate fixes for compilation errors.
        
        Args:
            test_code: Current test code
            compiler_output: Compiler error output
            class_info: Optional class information
            
        Returns:
            Dictionary with fix information
        """
        analysis = self.analyzer.analyze(compiler_output)
        
        fixes = []
        for error in analysis.errors:
            fix = self._generate_specific_fix(error, test_code, class_info)
            fixes.append(fix)
        
        return {
            "analysis": analysis,
            "fixes": fixes,
            "context": self.analyzer.get_fix_prompt_context(analysis)
        }
    
    def _generate_specific_fix(
        self,
        error: CompilationError,
        test_code: str,
        class_info: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate a specific fix for an error."""
        fix = {
            "error_type": error.error_type.name,
            "message": error.message,
            "line": error.line_number,
            "action": "unknown",
            "details": {}
        }
        
        if error.error_type == ErrorType.IMPORT_ERROR:
            fix["action"] = "add_import"
            fix["details"] = {
                "import_statement": f"import {error.error_token};" if error.error_token else "",
                "location": "top_of_file"
            }
        
        elif error.error_type == ErrorType.SYMBOL_NOT_FOUND:
            fix["action"] = "define_or_import"
            fix["details"] = {
                "symbol": error.error_token,
                "suggestion": f"Check import or define {error.error_token}"
            }
        
        elif error.error_type == ErrorType.METHOD_NOT_FOUND:
            fix["action"] = "fix_method_call"
            fix["details"] = {
                "method": error.error_token,
                "suggestion": "Verify method name and parameters"
            }
        
        elif error.error_type == ErrorType.TYPE_MISMATCH:
            fix["action"] = "fix_type"
            fix["details"] = {
                "suggestion": "Add explicit type casting or fix assignment"
            }
        
        elif error.error_type == ErrorType.SYNTAX_ERROR:
            fix["action"] = "fix_syntax"
            fix["details"] = {
                "suggestion": error.fix_hint
            }
        
        return fix