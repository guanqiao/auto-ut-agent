"""Result Aggregator for aggregating SubAgent results.

This module provides:
- ResultAggregator: Aggregate and validate results from multiple agents
- Inconsistency detection
- Result validation
- Summary report generation
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from uuid import uuid4

logger = logging.getLogger(__name__)


class AggregationStrategy(Enum):
    """Strategies for aggregating results."""
    FIRST_SUCCESS = "first_success"
    ALL_SUCCESS = "all_success"
    MAJORITY = "majority"
    MERGE = "merge"
    BEST_QUALITY = "best_quality"
    CUSTOM = "custom"


class ValidationStatus(Enum):
    """Status of result validation."""
    VALID = "valid"
    INVALID = "invalid"
    PARTIAL = "partial"
    PENDING = "pending"


class InconsistencyType(Enum):
    """Types of inconsistencies."""
    VALUE_MISMATCH = "value_mismatch"
    TYPE_MISMATCH = "type_mismatch"
    MISSING_KEY = "missing_key"
    EXTRA_KEY = "extra_key"
    CONFLICT = "conflict"
    ORDER_MISMATCH = "order_mismatch"


@dataclass
class Inconsistency:
    """Represents an inconsistency between results."""
    inconsistency_id: str
    inconsistency_type: InconsistencyType
    key: str
    values: List[Tuple[str, Any]]
    description: str
    severity: str = "medium"
    resolution_suggestion: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Result of validation."""
    validation_id: str
    status: ValidationStatus
    total_results: int
    valid_results: int
    invalid_results: int
    inconsistencies: List[Inconsistency] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AggregatedResult:
    """Aggregated result from multiple agents."""
    aggregation_id: str
    success: bool
    strategy: AggregationStrategy
    total_tasks: int
    successful_tasks: int
    failed_tasks: int
    results: List[Dict[str, Any]] = field(default_factory=list)
    merged_result: Any = None
    inconsistencies: List[Inconsistency] = field(default_factory=list)
    validation: Optional[ValidationResult] = None
    summary: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SummaryReport:
    """Summary report of aggregation."""
    report_id: str
    aggregation_id: str
    total_agents: int
    total_tasks: int
    success_rate: float
    execution_time_ms: int
    results_by_agent: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    issues_found: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class ResultAggregator:
    """Aggregator for combining and validating SubAgent results.

    Features:
    - Multiple aggregation strategies
    - Inconsistency detection
    - Result validation
    - Summary report generation
    """

    def __init__(
        self,
        default_strategy: AggregationStrategy = AggregationStrategy.ALL_SUCCESS,
        validation_enabled: bool = True,
        custom_validator: Optional[Callable] = None
    ):
        """Initialize ResultAggregator.

        Args:
            default_strategy: Default aggregation strategy
            validation_enabled: Whether to validate results
            custom_validator: Optional custom validation function
        """
        self.default_strategy = default_strategy
        self.validation_enabled = validation_enabled
        self.custom_validator = custom_validator

        self._aggregation_history: List[AggregatedResult] = []
        self._validators: Dict[str, Callable] = {}

        self._stats = {
            "aggregations": 0,
            "successful_aggregations": 0,
            "inconsistencies_detected": 0
        }

        self._register_default_validators()

        logger.info(f"[ResultAggregator] Initialized with strategy: {default_strategy.value}")

    def _register_default_validators(self) -> None:
        """Register default validators."""
        self._validators["type_check"] = self._validate_types
        self._validators["required_keys"] = self._validate_required_keys
        self._validators["value_range"] = self._validate_value_ranges

    def register_validator(
        self,
        name: str,
        validator: Callable[[List[Dict[str, Any]]], List[str]]
    ) -> None:
        """Register a custom validator.

        Args:
            name: Validator name
            validator: Validation function
        """
        self._validators[name] = validator
        logger.info(f"[ResultAggregator] Registered validator: {name}")

    async def aggregate(
        self,
        results: List[Any],
        strategy: Optional[AggregationStrategy] = None
    ) -> AggregatedResult:
        """Aggregate multiple results.

        Args:
            results: List of results to aggregate
            strategy: Optional strategy override

        Returns:
            AggregatedResult
        """
        strategy = strategy or self.default_strategy
        self._stats["aggregations"] += 1

        normalized_results = self._normalize_results(results)

        inconsistencies = []
        if self.validation_enabled:
            inconsistencies = self.detect_inconsistencies(normalized_results)
            self._stats["inconsistencies_detected"] += len(inconsistencies)

        validation = None
        if self.validation_enabled:
            validation = self.validate_results(normalized_results)

        merged_result = await self._apply_strategy(normalized_results, strategy)

        successful = sum(1 for r in normalized_results if r.get("success", True))
        failed = len(normalized_results) - successful

        aggregated = AggregatedResult(
            aggregation_id=str(uuid4()),
            success=failed == 0,
            strategy=strategy,
            total_tasks=len(normalized_results),
            successful_tasks=successful,
            failed_tasks=failed,
            results=normalized_results,
            merged_result=merged_result,
            inconsistencies=inconsistencies,
            validation=validation,
            summary=self._generate_summary(normalized_results, inconsistencies)
        )

        self._aggregation_history.append(aggregated)

        if aggregated.success:
            self._stats["successful_aggregations"] += 1

        logger.info(f"[ResultAggregator] Aggregated {len(results)} results "
                   f"(success={aggregated.success}, strategy={strategy.value})")

        return aggregated

    def _normalize_results(self, results: List[Any]) -> List[Dict[str, Any]]:
        """Normalize results to a common format.

        Args:
            results: List of results

        Returns:
            List of normalized result dictionaries
        """
        normalized = []

        for result in results:
            if isinstance(result, dict):
                normalized.append(result)
            elif hasattr(result, "to_dict"):
                normalized.append(result.to_dict())
            elif hasattr(result, "__dict__"):
                normalized.append({
                    "value": result,
                    "success": True
                })
            else:
                normalized.append({
                    "value": result,
                    "success": True
                })

        return normalized

    async def _apply_strategy(
        self,
        results: List[Dict[str, Any]],
        strategy: AggregationStrategy
    ) -> Any:
        """Apply aggregation strategy.

        Args:
            results: Normalized results
            strategy: Strategy to apply

        Returns:
            Merged result
        """
        if not results:
            return None

        if strategy == AggregationStrategy.FIRST_SUCCESS:
            return self._first_success(results)
        elif strategy == AggregationStrategy.ALL_SUCCESS:
            return self._all_success(results)
        elif strategy == AggregationStrategy.MAJORITY:
            return self._majority(results)
        elif strategy == AggregationStrategy.MERGE:
            return self._merge(results)
        elif strategy == AggregationStrategy.BEST_QUALITY:
            return self._best_quality(results)
        else:
            return self._all_success(results)

    def _first_success(self, results: List[Dict[str, Any]]) -> Any:
        """Return first successful result."""
        for result in results:
            if result.get("success", True):
                return result.get("result", result)
        return None

    def _all_success(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Return combined results if all successful."""
        combined = {
            "success": all(r.get("success", True) for r in results),
            "results": [r.get("result", r) for r in results]
        }
        return combined

    def _majority(self, results: List[Dict[str, Any]]) -> Any:
        """Return the majority result value."""
        if not results:
            return None

        value_counts: Dict[str, int] = {}
        for result in results:
            value = result.get("result", result)
            key = str(value) if not isinstance(value, (dict, list)) else id(value)
            value_counts[key] = value_counts.get(key, 0) + 1

        majority_key = max(value_counts.keys(), key=lambda k: value_counts[k])

        for result in results:
            value = result.get("result", result)
            key = str(value) if not isinstance(value, (dict, list)) else id(value)
            if key == majority_key:
                return value

        return None

    def _merge(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge all results into a single dictionary."""
        merged: Dict[str, Any] = {}

        for i, result in enumerate(results):
            value = result.get("result", result)
            if isinstance(value, dict):
                for key, val in value.items():
                    if key in merged:
                        if isinstance(merged[key], list):
                            merged[key].append(val)
                        else:
                            merged[key] = [merged[key], val]
                    else:
                        merged[key] = val
            else:
                merged[f"result_{i}"] = value

        return merged

    def _best_quality(self, results: List[Dict[str, Any]]) -> Any:
        """Return the result with best quality score."""
        best_result = None
        best_score = -1

        for result in results:
            if not result.get("success", True):
                continue

            score = result.get("quality_score", result.get("confidence", 0.5))

            if score > best_score:
                best_score = score
                best_result = result.get("result", result)

        return best_result

    def detect_inconsistencies(
        self,
        results: List[Dict[str, Any]]
    ) -> List[Inconsistency]:
        """Detect inconsistencies between results.

        Args:
            results: List of results

        Returns:
            List of inconsistencies
        """
        inconsistencies = []

        if len(results) < 2:
            return inconsistencies

        all_keys: Set[str] = set()
        for result in results:
            value = result.get("result", result)
            if isinstance(value, dict):
                all_keys.update(value.keys())

        for key in all_keys:
            values = []
            for i, result in enumerate(results):
                value = result.get("result", result)
                if isinstance(value, dict):
                    if key in value:
                        values.append((f"result_{i}", value[key]))
                    else:
                        values.append((f"result_{i}", None))

            if len(values) > 1:
                non_none_values = [(src, v) for src, v in values if v is not None]

                if len(non_none_values) != len(values):
                    inconsistencies.append(Inconsistency(
                        inconsistency_id=str(uuid4()),
                        inconsistency_type=InconsistencyType.MISSING_KEY,
                        key=key,
                        values=values,
                        description=f"Key '{key}' is missing in some results",
                        severity="low"
                    ))
                    continue

                unique_values = set()
                for src, v in non_none_values:
                    if isinstance(v, (dict, list)):
                        unique_values.add(str(v)[:100])
                    else:
                        unique_values.add(str(v))

                if len(unique_values) > 1:
                    inconsistencies.append(Inconsistency(
                        inconsistency_id=str(uuid4()),
                        inconsistency_type=InconsistencyType.VALUE_MISMATCH,
                        key=key,
                        values=non_none_values,
                        description=f"Key '{key}' has different values across results",
                        severity="medium",
                        resolution_suggestion="Use majority vote or first value"
                    ))

        return inconsistencies

    def validate_results(
        self,
        results: List[Dict[str, Any]]
    ) -> ValidationResult:
        """Validate results.

        Args:
            results: List of results

        Returns:
            ValidationResult
        """
        errors = []
        warnings = []

        for name, validator in self._validators.items():
            try:
                validator_errors = validator(results)
                errors.extend([f"[{name}] {e}" for e in validator_errors])
            except Exception as e:
                warnings.append(f"Validator {name} failed: {str(e)}")

        if self.custom_validator:
            try:
                custom_errors = self.custom_validator(results)
                errors.extend(custom_errors)
            except Exception as e:
                warnings.append(f"Custom validator failed: {str(e)}")

        valid_count = sum(1 for r in results if r.get("success", True))

        status = ValidationStatus.VALID
        if errors:
            if valid_count == len(results):
                status = ValidationStatus.PARTIAL
            else:
                status = ValidationStatus.INVALID

        return ValidationResult(
            validation_id=str(uuid4()),
            status=status,
            total_results=len(results),
            valid_results=valid_count,
            invalid_results=len(results) - valid_count,
            errors=errors,
            warnings=warnings
        )

    def _validate_types(self, results: List[Dict[str, Any]]) -> List[str]:
        """Validate result types are consistent."""
        errors = []

        if not results:
            return errors

        first_type = type(results[0].get("result", results[0]))

        for i, result in enumerate(results[1:], 1):
            value = result.get("result", result)
            if type(value) != first_type:
                errors.append(
                    f"Type mismatch: result_0 is {first_type.__name__}, "
                    f"result_{i} is {type(value).__name__}"
                )

        return errors

    def _validate_required_keys(
        self,
        results: List[Dict[str, Any]]
    ) -> List[str]:
        """Validate required keys are present."""
        errors = []

        required_keys = {"success"}

        for i, result in enumerate(results):
            for key in required_keys:
                if key not in result:
                    errors.append(f"Missing required key '{key}' in result_{i}")

        return errors

    def _validate_value_ranges(
        self,
        results: List[Dict[str, Any]]
    ) -> List[str]:
        """Validate values are within expected ranges."""
        errors = []

        for i, result in enumerate(results):
            confidence = result.get("confidence")
            if confidence is not None:
                if not 0 <= confidence <= 1:
                    errors.append(
                        f"Invalid confidence value in result_{i}: {confidence}"
                    )

        return errors

    def generate_summary(
        self,
        results: List[Any]
    ) -> SummaryReport:
        """Generate a summary report.

        Args:
            results: List of results

        Returns:
            SummaryReport
        """
        normalized = self._normalize_results(results)

        successful = sum(1 for r in normalized if r.get("success", True))
        total = len(normalized)

        issues = []
        recommendations = []

        inconsistencies = self.detect_inconsistencies(normalized)
        for inc in inconsistencies:
            issues.append(f"{inc.inconsistency_type.value}: {inc.description}")
            if inc.resolution_suggestion:
                recommendations.append(inc.resolution_suggestion)

        if successful < total:
            recommendations.append("Review failed tasks for common error patterns")

        if len(inconsistencies) > len(normalized) * 0.5:
            recommendations.append("Consider improving task decomposition for consistency")

        report = SummaryReport(
            report_id=str(uuid4()),
            aggregation_id="",
            total_agents=len(normalized),
            total_tasks=total,
            success_rate=successful / total if total > 0 else 0,
            execution_time_ms=sum(
                r.get("execution_time_ms", 0) for r in normalized
            ),
            issues_found=issues,
            recommendations=recommendations
        )

        return report

    def _generate_summary(
        self,
        results: List[Dict[str, Any]],
        inconsistencies: List[Inconsistency]
    ) -> str:
        """Generate a text summary."""
        total = len(results)
        successful = sum(1 for r in results if r.get("success", True))

        lines = [
            f"Aggregated {total} results: {successful} successful, {total - successful} failed",
        ]

        if inconsistencies:
            lines.append(f"Detected {len(inconsistencies)} inconsistencies")

        return ". ".join(lines)

    def get_aggregation_history(self, limit: int = 20) -> List[AggregatedResult]:
        """Get aggregation history.

        Args:
            limit: Maximum results

        Returns:
            List of AggregatedResults
        """
        return self._aggregation_history[-limit:]

    def get_stats(self) -> Dict[str, Any]:
        """Get aggregator statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "aggregations": self._stats["aggregations"],
            "successful_aggregations": self._stats["successful_aggregations"],
            "inconsistencies_detected": self._stats["inconsistencies_detected"],
            "success_rate": (
                self._stats["successful_aggregations"] / self._stats["aggregations"]
                if self._stats["aggregations"] > 0 else 0
            )
        }


def create_result_aggregator(
    strategy: AggregationStrategy = AggregationStrategy.ALL_SUCCESS
) -> ResultAggregator:
    """Create a ResultAggregator.

    Args:
        strategy: Default aggregation strategy

    Returns:
        ResultAggregator instance
    """
    return ResultAggregator(default_strategy=strategy)
