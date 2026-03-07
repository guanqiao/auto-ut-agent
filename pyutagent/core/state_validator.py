"""State consistency validator.

This module provides validation for state consistency across the agent,
checking required fields, type consistency, and cross-field dependencies.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from .config import DEFAULT_MAX_ITERATIONS

logger = logging.getLogger(__name__)


@dataclass
class ValidationError:
    """Represents a validation error."""
    field: str
    expected: Any
    actual: Any
    message: str = ""


@dataclass
class ValidationResult:
    """Result of state validation."""
    is_valid: bool = True
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def add_error(self, field: str, expected: Any, actual: Any, message: str = ""):
        """Add a validation error.
        
        Args:
            field: Field name
            expected: Expected value or type
            actual: Actual value
            message: Error message
        """
        self.errors.append(ValidationError(field, expected, actual, message))
        self.is_valid = False
    
    def add_warning(self, message: str):
        """Add a validation warning.
        
        Args:
            message: Warning message
        """
        self.warnings.append(message)
    
    def merge(self, other: 'ValidationResult'):
        """Merge another validation result into this one.
        
        Args:
            other: Another ValidationResult
        """
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        if not other.is_valid:
            self.is_valid = False
    
    def __str__(self) -> str:
        """String representation."""
        if self.is_valid and not self.warnings:
            return "ValidationResult(valid=True)"
        
        parts = [f"ValidationResult(valid={self.is_valid}"]
        if self.errors:
            parts.append(f", errors={len(self.errors)}")
        if self.warnings:
            parts.append(f", warnings={len(self.warnings)}")
        parts.append(")")
        return "".join(parts)


class StateValidator:
    """Validates state consistency across the agent.
    
    Checks:
    - Required fields presence
    - Field type consistency
    - Cross-field dependencies
    - Value ranges
    - State transition validity
    
    Attributes:
        REQUIRED_FIELDS: Mapping of phase names to required field sets
        FIELD_TYPES: Mapping of field names to expected types
    """
    
    REQUIRED_FIELDS: Dict[str, Set[str]] = {
        "parsing": {"target_file"},
        "generating": {"target_file", "class_info"},
        "compiling": {"target_file", "class_info", "test_file"},
        "testing": {"target_file", "class_info", "test_file"},
        "analyzing": {"target_file", "class_info", "test_file"},
        "optimizing": {"target_file", "class_info", "test_file"},
    }
    
    FIELD_TYPES: Dict[str, type] = {
        "target_file": str,
        "test_file": str,
        "class_info": dict,
        "coverage_data": dict,
        "current_iteration": int,
        "current_coverage": (int, float),
        "max_iterations": int,
        "target_coverage": (int, float),
    }
    
    VALID_TRANSITIONS: Dict[str, Set[str]] = {
        "idle": {"parsing"},
        "parsing": {"generating", "failed"},
        "generating": {"compiling", "fixing", "failed"},
        "compiling": {"testing", "fixing", "failed"},
        "testing": {"analyzing", "fixing", "failed"},
        "analyzing": {"optimizing", "completed", "failed"},
        "optimizing": {"compiling", "completed", "failed"},
        "fixing": {"compiling", "testing", "generating", "failed"},
        "completed": {"idle"},
        "failed": {"idle", "parsing"},
        "paused": {"idle", "parsing", "generating", "compiling", "testing", "analyzing", "optimizing", "fixing"},
    }
    
    def validate(self, state: Dict[str, Any], current_phase: str) -> ValidationResult:
        """Validate state for a given phase.
        
        Args:
            state: Current state dictionary
            current_phase: Current phase name
            
        Returns:
            ValidationResult with any errors or warnings
        """
        result = ValidationResult(is_valid=True)
        
        self._validate_required_fields(state, current_phase.lower(), result)
        self._validate_field_types(state, result)
        self._validate_dependencies(state, result)
        self._validate_ranges(state, result)
        
        return result
    
    def _validate_required_fields(self, state: Dict[str, Any], phase: str, result: ValidationResult):
        """Validate required fields are present.
        
        Args:
            state: State dictionary
            phase: Current phase
            result: ValidationResult to update
        """
        required = self.REQUIRED_FIELDS.get(phase, set())
        for field in required:
            if field not in state or state[field] is None:
                result.add_error(field, "present", "missing", f"Required field '{field}' is missing")
    
    def _validate_field_types(self, state: Dict[str, Any], result: ValidationResult):
        """Validate field types.
        
        Args:
            state: State dictionary
            result: ValidationResult to update
        """
        for field, expected_type in self.FIELD_TYPES.items():
            if field in state and state[field] is not None:
                actual_value = state[field]
                if isinstance(expected_type, tuple):
                    if not isinstance(actual_value, expected_type):
                        type_names = " or ".join(t.__name__ for t in expected_type)
                        result.add_error(
                            field, 
                            type_names, 
                            type(actual_value).__name__,
                            f"Field '{field}' has wrong type"
                        )
                else:
                    if not isinstance(actual_value, expected_type):
                        result.add_error(
                            field, 
                            expected_type.__name__, 
                            type(actual_value).__name__,
                            f"Field '{field}' has wrong type"
                        )
    
    def _validate_dependencies(self, state: Dict[str, Any], result: ValidationResult):
        """Validate cross-field dependencies.
        
        Args:
            state: State dictionary
            result: ValidationResult to update
        """
        if "test_file" in state and "class_info" in state:
            test_file = state["test_file"]
            class_info = state["class_info"]
            
            if isinstance(class_info, dict):
                class_name = class_info.get("name", "")
                if class_name and isinstance(test_file, str) and class_name not in test_file:
                    result.add_warning(f"Test file name doesn't match class name: {test_file} vs {class_name}")
        
        if "target_file" in state and "class_info" in state:
            target_file = state["target_file"]
            class_info = state["class_info"]
            
            if isinstance(class_info, dict) and isinstance(target_file, str):
                class_name = class_info.get("name", "")
                if class_name and class_name not in target_file:
                    result.add_warning(f"Target file doesn't match class name: {target_file} vs {class_name}")
    
    def _validate_ranges(self, state: Dict[str, Any], result: ValidationResult):
        """Validate value ranges.
        
        Args:
            state: State dictionary
            result: ValidationResult to update
        """
        if "current_coverage" in state:
            coverage = state["current_coverage"]
            if isinstance(coverage, (int, float)) and not (0 <= coverage <= 1):
                result.add_error("current_coverage", "0-1", coverage, "Coverage should be between 0 and 1")
        
        if "target_coverage" in state:
            coverage = state["target_coverage"]
            if isinstance(coverage, (int, float)) and not (0 <= coverage <= 1):
                result.add_error("target_coverage", "0-1", coverage, "Target coverage should be between 0 and 1")
        
        if "current_iteration" in state:
            iteration = state["current_iteration"]
            if isinstance(iteration, int):
                if iteration < 0:
                    result.add_error("current_iteration", ">=0", iteration, "Iteration cannot be negative")
                
                max_iterations = state.get("max_iterations", DEFAULT_MAX_ITERATIONS)
                if isinstance(max_iterations, int) and iteration > max_iterations:
                    result.add_warning(f"Iteration ({iteration}) exceeds max ({max_iterations})")
    
    def validate_transition(
        self, 
        from_phase: str, 
        to_phase: str, 
        state: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """Validate a phase transition.
        
        Args:
            from_phase: Current phase
            to_phase: Target phase
            state: Optional current state for additional validation
            
        Returns:
            ValidationResult
        """
        result = ValidationResult(is_valid=True)
        
        from_lower = from_phase.lower()
        to_lower = to_phase.lower()
        
        valid_targets = self.VALID_TRANSITIONS.get(from_lower, set())
        if to_lower not in valid_targets:
            result.add_error(
                "transition",
                f"{from_phase} -> {valid_targets}",
                f"{from_phase} -> {to_phase}",
                f"Invalid phase transition from {from_phase} to {to_phase}"
            )
        
        if state and result.is_valid:
            self.validate(state, to_lower)
        
        return result
    
    def validate_class_info(self, class_info: Dict[str, Any]) -> ValidationResult:
        """Validate class_info structure.
        
        Args:
            class_info: Class information dictionary
            
        Returns:
            ValidationResult
        """
        result = ValidationResult(is_valid=True)
        
        required_keys = {"name", "methods", "source"}
        for key in required_keys:
            if key not in class_info:
                result.add_error(key, "present", "missing", f"class_info missing required key '{key}'")
        
        if "methods" in class_info:
            methods = class_info["methods"]
            if not isinstance(methods, list):
                result.add_error("methods", "list", type(methods).__name__, "methods should be a list")
        
        if "name" in class_info:
            name = class_info["name"]
            if not isinstance(name, str) or not name.strip():
                result.add_error("name", "non-empty string", repr(name), "class name should be a non-empty string")
        
        return result
    
    def validate_test_code(self, test_code: str, class_info: Dict[str, Any]) -> ValidationResult:
        """Validate generated test code.
        
        Args:
            test_code: Generated test code
            class_info: Class information
            
        Returns:
            ValidationResult
        """
        result = ValidationResult(is_valid=True)
        
        if not test_code or not test_code.strip():
            result.add_error("test_code", "non-empty", "empty", "Test code is empty")
            return result
        
        class_name = class_info.get("name", "")
        if class_name:
            expected_test_class = f"{class_name}Test"
            if expected_test_class not in test_code:
                result.add_warning(f"Test class name '{expected_test_class}' not found in test code")
        
        required_imports = ["import org.junit", "import static"]
        has_junit_import = any(imp in test_code for imp in required_imports)
        if not has_junit_import:
            result.add_warning("Test code may be missing JUnit imports")
        
        if "@Test" not in test_code:
            result.add_warning("Test code may be missing @Test annotations")
        
        return result


def validate_state(state: Dict[str, Any], phase: str) -> ValidationResult:
    """Convenience function to validate state.
    
    Args:
        state: State dictionary
        phase: Current phase
        
    Returns:
        ValidationResult
    """
    validator = StateValidator()
    return validator.validate(state, phase)


def validate_transition(from_phase: str, to_phase: str) -> ValidationResult:
    """Convenience function to validate a transition.
    
    Args:
        from_phase: Current phase
        to_phase: Target phase
        
    Returns:
        ValidationResult
    """
    validator = StateValidator()
    return validator.validate_transition(from_phase, to_phase)
