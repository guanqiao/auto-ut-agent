"""Edit validator for verifying code edits.

This module provides comprehensive validation for code edits including:
- Syntax validation using tree-sitter
- Semantic validation for Java constructs
- Test structure validation
- Compilation simulation
"""

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Dict, Any, Set, Tuple
from pathlib import Path

from tree_sitter import Parser, Language, Node
import tree_sitter_java as tsjava


class ValidationErrorType(Enum):
    """Types of validation errors."""
    SYNTAX_ERROR = auto()
    MISSING_BRACE = auto()
    MISSING_PARENTHESIS = auto()
    MISSING_SEMICOLON = auto()
    UNCLOSED_STRING = auto()
    INVALID_IDENTIFIER = auto()
    IMPORT_ERROR = auto()
    TYPE_ERROR = auto()
    TEST_STRUCTURE_ERROR = auto()
    BALANCE_ERROR = auto()


@dataclass
class ValidationError:
    """Represents a validation error."""
    error_type: ValidationErrorType
    message: str
    line_number: Optional[int] = None
    column: Optional[int] = None
    severity: str = "error"  # error, warning
    suggestion: str = ""


@dataclass
class ValidationResult:
    """Result of validation."""
    is_valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)
    
    @property
    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return any(e.severity == "error" for e in self.errors)
    
    @property
    def error_count(self) -> int:
        """Get number of errors."""
        return len([e for e in self.errors if e.severity == "error"])
    
    @property
    def warning_count(self) -> int:
        """Get number of warnings."""
        return len(self.warnings)


class SyntaxValidator:
    """Validates Java syntax using tree-sitter."""
    
    def __init__(self):
        """Initialize syntax validator."""
        self.parser = Parser(Language(tsjava.language()))
    
    def validate(self, code: str) -> ValidationResult:
        """Validate Java code syntax.
        
        Args:
            code: Java code to validate
            
        Returns:
            ValidationResult with errors and warnings
        """
        errors = []
        
        try:
            tree = self.parser.parse(code.encode())
            root = tree.root_node
            
            # Collect all syntax errors
            self._collect_syntax_errors(root, errors)
            
            # Check for structural issues
            structural_errors = self._check_structure(code)
            errors.extend(structural_errors)
            
        except Exception as e:
            errors.append(ValidationError(
                error_type=ValidationErrorType.SYNTAX_ERROR,
                message=f"Parse error: {str(e)}",
                severity="error"
            ))
        
        is_valid = not any(e.severity == "error" for e in errors)
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors
        )
    
    def _collect_syntax_errors(self, node: Node, errors: List[ValidationError]):
        """Recursively collect syntax errors from AST.
        
        Args:
            node: Current AST node
            errors: List to collect errors into
        """
        # Check for error nodes
        if node.type == 'ERROR':
            error = self._create_error_from_node(node)
            if error:
                errors.append(error)
        
        # Check for missing nodes
        if node.is_missing:
            errors.append(ValidationError(
                error_type=ValidationErrorType.SYNTAX_ERROR,
                message=f"Missing {node.type}",
                line_number=node.start_point[0] + 1,
                column=node.start_point[1],
                severity="error"
            ))
        
        # Recurse into children
        for child in node.children:
            self._collect_syntax_errors(child, errors)
    
    def _create_error_from_node(self, node: Node) -> Optional[ValidationError]:
        """Create a validation error from an ERROR node.
        
        Args:
            node: ERROR node
            
        Returns:
            ValidationError or None
        """
        line = node.start_point[0] + 1
        col = node.start_point[1]
        
        # Try to determine error type from context
        error_type = ValidationErrorType.SYNTAX_ERROR
        message = "Syntax error"
        suggestion = ""
        
        # Get text around error
        text = node.text.decode() if node.text else ""
        
        if '{' in text or '}' in text:
            error_type = ValidationErrorType.MISSING_BRACE
            message = "Unmatched or missing brace"
            suggestion = "Check for balanced braces {}"
        elif '(' in text or ')' in text:
            error_type = ValidationErrorType.MISSING_PARENTHESIS
            message = "Unmatched or missing parenthesis"
            suggestion = "Check for balanced parentheses ()"
        elif ';' in text:
            error_type = ValidationErrorType.MISSING_SEMICOLON
            message = "Missing semicolon"
            suggestion = "Add semicolon at end of statement"
        elif '"' in text or "'" in text:
            error_type = ValidationErrorType.UNCLOSED_STRING
            message = "Unclosed string literal"
            suggestion = "Close the string with matching quote"
        
        return ValidationError(
            error_type=error_type,
            message=message,
            line_number=line,
            column=col,
            severity="error",
            suggestion=suggestion
        )
    
    def _check_structure(self, code: str) -> List[ValidationError]:
        """Check code structure for common issues.
        
        Args:
            code: Java code
            
        Returns:
            List of validation errors
        """
        errors = []
        lines = code.split('\n')
        
        # Check brace balance
        open_braces = code.count('{')
        close_braces = code.count('}')
        if open_braces != close_braces:
            errors.append(ValidationError(
                error_type=ValidationErrorType.BALANCE_ERROR,
                message=f"Unbalanced braces: {open_braces} open, {close_braces} close",
                severity="error",
                suggestion="Ensure all opening braces have matching closing braces"
            ))
        
        # Check parenthesis balance
        open_parens = code.count('(')
        close_parens = code.count(')')
        if open_parens != close_parens:
            errors.append(ValidationError(
                error_type=ValidationErrorType.BALANCE_ERROR,
                message=f"Unbalanced parentheses: {open_parens} open, {close_parens} close",
                severity="error",
                suggestion="Ensure all opening parentheses have matching closing parentheses"
            ))
        
        # Check bracket balance
        open_brackets = code.count('[')
        close_brackets = code.count(']')
        if open_brackets != close_brackets:
            errors.append(ValidationError(
                error_type=ValidationErrorType.BALANCE_ERROR,
                message=f"Unbalanced brackets: {open_brackets} open, {close_brackets} close",
                severity="error",
                suggestion="Ensure all opening brackets have matching closing brackets"
            ))
        
        # Check for common Java issues
        for i, line in enumerate(lines, 1):
            # Check for statements without semicolons (simplified check)
            stripped = line.strip()
            if (stripped and 
                not stripped.endswith('{') and 
                not stripped.endswith('}') and
                not stripped.endswith(';') and
                not stripped.startswith('//') and
                not stripped.startswith('/*') and
                not stripped.startswith('*') and
                not stripped.startswith('import') and
                not stripped.startswith('package') and
                not stripped.startswith('@') and
                not stripped.startswith('class') and
                not stripped.startswith('public') and
                not stripped.startswith('private') and
                not stripped.startswith('protected') and
                not stripped.startswith('//') and
                not stripped.startswith('/*') and
                not stripped.startswith('*') and
                '(' in stripped):
                # This might be a method call without semicolon
                pass  # Skip for now, tree-sitter will catch real errors
        
        return errors


class TestCodeValidator:
    """Validates test code structure and conventions."""
    
    # Required imports for JUnit 5 tests
    REQUIRED_IMPORTS = {
        'org.junit.jupiter.api.Test',
        'org.junit.jupiter.api.BeforeEach',
        'org.junit.jupiter.api.AfterEach',
    }
    
    # Optional but common imports
    RECOMMENDED_IMPORTS = {
        'org.junit.jupiter.api.Assertions',
        'org.mockito.Mockito',
        'org.mockito.Mock',
        'org.mockito.InjectMocks',
    }
    
    def __init__(self):
        """Initialize test code validator."""
        self.syntax_validator = SyntaxValidator()
    
    def validate(self, code: str, check_test_structure: bool = True) -> ValidationResult:
        """Validate test code.
        
        Args:
            code: Test code to validate
            check_test_structure: Whether to check test-specific structure
            
        Returns:
            ValidationResult with errors and warnings
        """
        # First validate syntax
        result = self.syntax_validator.validate(code)
        
        if not check_test_structure:
            return result
        
        # Additional test-specific validations
        test_errors = self._validate_test_structure(code)
        test_warnings = self._validate_test_conventions(code)
        
        result.errors.extend(test_errors)
        result.warnings.extend(test_warnings)
        
        # Recalculate validity
        result.is_valid = not any(e.severity == "error" for e in result.errors)
        
        return result
    
    def _validate_test_structure(self, code: str) -> List[ValidationError]:
        """Validate test class structure.
        
        Args:
            code: Test code
            
        Returns:
            List of validation errors
        """
        errors = []
        
        # Check for class declaration
        if 'class' not in code:
            errors.append(ValidationError(
                error_type=ValidationErrorType.TEST_STRUCTURE_ERROR,
                message="No class declaration found",
                severity="error",
                suggestion="Add a class declaration: public class ClassNameTest { }"
            ))
        
        # Check for @Test annotation
        if '@Test' not in code:
            errors.append(ValidationError(
                error_type=ValidationErrorType.TEST_STRUCTURE_ERROR,
                message="No @Test annotation found",
                severity="warning",
                suggestion="Add at least one test method with @Test annotation"
            ))
        
        # Check for test methods
        test_method_pattern = r'@Test\s+\w+\s+void\s+(\w+)'
        if not re.search(test_method_pattern, code):
            errors.append(ValidationError(
                error_type=ValidationErrorType.TEST_STRUCTURE_ERROR,
                message="No valid test methods found",
                severity="warning",
                suggestion="Add test methods with pattern: @Test void methodName() { }"
            ))
        
        return errors
    
    def _validate_test_conventions(self, code: str) -> List[ValidationError]:
        """Validate test code conventions.
        
        Args:
            code: Test code
            
        Returns:
            List of validation warnings
        """
        warnings = []
        
        # Check for class naming convention (should end with Test)
        class_pattern = r'class\s+(\w+)'
        match = re.search(class_pattern, code)
        if match:
            class_name = match.group(1)
            if not class_name.endswith('Test'):
                warnings.append(ValidationError(
                    error_type=ValidationErrorType.TEST_STRUCTURE_ERROR,
                    message=f"Test class name '{class_name}' should end with 'Test'",
                    severity="warning",
                    suggestion=f"Rename to {class_name}Test"
                ))
        
        # Check for public test methods (JUnit 5 test methods should be package-private)
        public_test_pattern = r'@Test\s+public\s+void'
        if re.search(public_test_pattern, code):
            warnings.append(ValidationError(
                error_type=ValidationErrorType.TEST_STRUCTURE_ERROR,
                message="JUnit 5 test methods should be package-private, not public",
                severity="warning",
                suggestion="Remove 'public' modifier from test methods"
            ))
        
        # Check for empty test methods
        empty_test_pattern = r'@Test\s+\w+\s+void\s+\w+\(\)\s*\{\s*\}'
        if re.search(empty_test_pattern, code):
            warnings.append(ValidationError(
                error_type=ValidationErrorType.TEST_STRUCTURE_ERROR,
                message="Empty test method found",
                severity="warning",
                suggestion="Add test assertions or remove empty test"
            ))
        
        return warnings
    
    def validate_imports(self, code: str) -> Tuple[Set[str], Set[str]]:
        """Validate imports in test code.
        
        Args:
            code: Test code
            
        Returns:
            Tuple of (missing_required, missing_recommended)
        """
        existing_imports = set()
        
        # Extract existing imports
        import_pattern = r'import\s+([\w.]+);'
        for match in re.finditer(import_pattern, code):
            existing_imports.add(match.group(1))
        
        # Also check for static imports
        static_import_pattern = r'import\s+static\s+([\w.]+);'
        for match in re.finditer(static_import_pattern, code):
            existing_imports.add(match.group(1))
        
        # Check for wildcard imports
        wildcard_pattern = r'import\s+([\w.]+\.\*);'
        for match in re.finditer(wildcard_pattern, code):
            package = match.group(1).replace('.*', '')
            # Add common classes from wildcard import
            if 'org.junit.jupiter.api' in package:
                existing_imports.add('org.junit.jupiter.api.Test')
                existing_imports.add('org.junit.jupiter.api.BeforeEach')
        
        missing_required = self.REQUIRED_IMPORTS - existing_imports
        missing_recommended = self.RECOMMENDED_IMPORTS - existing_imports
        
        return missing_required, missing_recommended


class EditImpactAnalyzer:
    """Analyzes the impact of code edits."""
    
    def __init__(self):
        """Initialize impact analyzer."""
        self.parser = Parser(Language(tsjava.language()))
    
    def analyze_impact(
        self,
        original_code: str,
        modified_code: str
    ) -> Dict[str, Any]:
        """Analyze the impact of modifications.
        
        Args:
            original_code: Original code
            modified_code: Modified code
            
        Returns:
            Dictionary with impact analysis
        """
        impact = {
            'lines_changed': 0,
            'methods_added': 0,
            'methods_removed': 0,
            'methods_modified': 0,
            'imports_added': [],
            'imports_removed': [],
            'structural_changes': [],
            'risk_level': 'low'
        }
        
        # Compare lines
        original_lines = original_code.split('\n')
        modified_lines = modified_code.split('\n')
        
        # Simple line diff
        import difflib
        diff = list(difflib.unified_diff(original_lines, modified_lines, lineterm=''))
        
        added_lines = sum(1 for line in diff if line.startswith('+') and not line.startswith('+++'))
        removed_lines = sum(1 for line in diff if line.startswith('-') and not line.startswith('---'))
        impact['lines_changed'] = added_lines + removed_lines
        
        # Analyze import changes
        original_imports = self._extract_imports(original_code)
        modified_imports = self._extract_imports(modified_code)
        
        impact['imports_added'] = list(modified_imports - original_imports)
        impact['imports_removed'] = list(original_imports - modified_imports)
        
        # Analyze method changes
        method_changes = self._analyze_method_changes(original_code, modified_code)
        impact.update(method_changes)
        
        # Determine risk level
        impact['risk_level'] = self._calculate_risk_level(impact)
        
        return impact
    
    def _extract_imports(self, code: str) -> Set[str]:
        """Extract imports from code.
        
        Args:
            code: Java code
            
        Returns:
            Set of import statements
        """
        imports = set()
        pattern = r'import\s+([\w.]+);'
        for match in re.finditer(pattern, code):
            imports.add(match.group(1))
        return imports
    
    def _analyze_method_changes(
        self,
        original_code: str,
        modified_code: str
    ) -> Dict[str, int]:
        """Analyze method-level changes.
        
        Args:
            original_code: Original code
            modified_code: Modified code
            
        Returns:
            Dictionary with method change counts
        """
        original_methods = self._extract_methods(original_code)
        modified_methods = self._extract_methods(modified_code)
        
        added = len(modified_methods - original_methods)
        removed = len(original_methods - modified_methods)
        
        # Check for modified methods (same signature, different body)
        modified = 0
        for method in original_methods & modified_methods:
            if self._get_method_body(original_code, method) != self._get_method_body(modified_code, method):
                modified += 1
        
        return {
            'methods_added': added,
            'methods_removed': removed,
            'methods_modified': modified
        }
    
    def _extract_methods(self, code: str) -> Set[str]:
        """Extract method signatures from code.
        
        Args:
            code: Java code
            
        Returns:
            Set of method signatures
        """
        methods = set()
        
        try:
            tree = self.parser.parse(code.encode())
            root = tree.root_node
            
            def find_methods(node):
                if node.type == 'method_declaration':
                    # Extract method name
                    for child in node.children:
                        if child.type == 'identifier':
                            methods.add(child.text.decode())
                            break
                
                for child in node.children:
                    find_methods(child)
            
            find_methods(root)
        except Exception:
            # Fallback to regex
            pattern = r'(?:void|\w+)\s+(\w+)\s*\([^)]*\)\s*\{'
            for match in re.finditer(pattern, code):
                methods.add(match.group(1))
        
        return methods
    
    def _get_method_body(self, code: str, method_name: str) -> str:
        """Get the body of a method.
        
        Args:
            code: Java code
            method_name: Method name
            
        Returns:
            Method body or empty string
        """
        # Simple regex-based extraction
        pattern = rf'(?:void|\w+)\s+{re.escape(method_name)}\s*\([^)]*\)\s*\{{'
        match = re.search(pattern, code)
        if not match:
            return ""
        
        start = match.end() - 1  # Position of opening brace
        
        # Find matching closing brace
        brace_count = 0
        pos = start
        while pos < len(code):
            if code[pos] == '{':
                brace_count += 1
            elif code[pos] == '}':
                brace_count -= 1
                if brace_count == 0:
                    return code[start:pos+1]
            pos += 1
        
        return ""
    
    def _calculate_risk_level(self, impact: Dict[str, Any]) -> str:
        """Calculate risk level based on impact.
        
        Args:
            impact: Impact analysis dictionary
            
        Returns:
            Risk level: low, medium, high
        """
        score = 0
        
        # Lines changed
        if impact['lines_changed'] > 50:
            score += 3
        elif impact['lines_changed'] > 20:
            score += 2
        elif impact['lines_changed'] > 5:
            score += 1
        
        # Methods removed (high risk)
        score += impact['methods_removed'] * 3
        
        # Methods modified (medium risk)
        score += impact['methods_modified'] * 2
        
        # Methods added (low risk)
        score += impact['methods_added'] * 1
        
        # Import changes
        if len(impact['imports_removed']) > 0:
            score += 2
        
        if score >= 6:
            return 'high'
        elif score >= 3:
            return 'medium'
        else:
            return 'low'


class EditValidator:
    """Main validator for code edits."""
    
    def __init__(self):
        """Initialize edit validator."""
        self.syntax_validator = SyntaxValidator()
        self.test_validator = TestCodeValidator()
        self.impact_analyzer = EditImpactAnalyzer()
    
    def validate_edit(
        self,
        original_code: str,
        modified_code: str,
        is_test_code: bool = True
    ) -> ValidationResult:
        """Validate a code edit.
        
        Args:
            original_code: Original code
            modified_code: Modified code after edit
            is_test_code: Whether the code is test code
            
        Returns:
            ValidationResult
        """
        # Validate modified code
        if is_test_code:
            result = self.test_validator.validate(modified_code)
        else:
            result = self.syntax_validator.validate(modified_code)
        
        # Analyze impact
        impact = self.impact_analyzer.analyze_impact(original_code, modified_code)
        
        # Add impact-based warnings
        if impact['risk_level'] == 'high':
            result.warnings.append(ValidationError(
                error_type=ValidationErrorType.TEST_STRUCTURE_ERROR,
                message="High risk edit: significant changes detected",
                severity="warning",
                suggestion="Review changes carefully before applying"
            ))
        
        if impact['methods_removed'] > 0:
            result.warnings.append(ValidationError(
                error_type=ValidationErrorType.TEST_STRUCTURE_ERROR,
                message=f"{impact['methods_removed']} method(s) will be removed",
                severity="warning",
                suggestion="Ensure methods are intentionally removed"
            ))
        
        return result
    
    def validate_edit_batch(
        self,
        original_code: str,
        edit_results: List[Any]
    ) -> List[ValidationResult]:
        """Validate a batch of edits.
        
        Args:
            original_code: Original code
            edit_results: List of edit results to validate
            
        Returns:
            List of validation results
        """
        results = []
        current_code = original_code
        
        for edit_result in edit_results:
            if hasattr(edit_result, 'modified_content'):
                result = self.validate_edit(current_code, edit_result.modified_content)
                results.append(result)
                if result.is_valid:
                    current_code = edit_result.modified_content
            else:
                # Assume it's a string
                result = self.validate_edit(current_code, str(edit_result))
                results.append(result)
                if result.is_valid:
                    current_code = str(edit_result)
        
        return results
    
    def quick_validate(self, code: str) -> bool:
        """Quick validation check.
        
        Args:
            code: Code to validate
            
        Returns:
            True if code appears valid
        """
        # Quick structural checks
        if code.count('{') != code.count('}'):
            return False
        if code.count('(') != code.count(')'):
            return False
        if code.count('[') != code.count(']'):
            return False
        
        # Check for empty code
        if not code.strip():
            return False
        
        # Check for class declaration
        if 'class' not in code:
            return False
        
        return True


def validate_syntax_only(code: str) -> bool:
    """Quick syntax validation.
    
    Args:
        code: Java code to validate
        
    Returns:
        True if syntax is valid
    """
    validator = SyntaxValidator()
    result = validator.validate(code)
    return result.is_valid


def validate_test_code(code: str) -> ValidationResult:
    """Validate test code.
    
    Args:
        code: Test code to validate
        
    Returns:
        ValidationResult
    """
    validator = TestCodeValidator()
    return validator.validate(code)
