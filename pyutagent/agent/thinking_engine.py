"""Agent thinking engine for intelligent reasoning.

This module provides the ability for the agent to think through problems,
analyze situations, and make intelligent decisions with transparent reasoning.
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Callable

from ..core.component_registry import SimpleComponent, component

logger = logging.getLogger(__name__)


class ThinkingType(Enum):
    """Types of thinking processes."""
    ANALYTICAL = "analytical"
    CREATIVE = "creative"
    CRITICAL = "critical"
    STRATEGIC = "strategic"
    REFLECTIVE = "reflective"
    PREDICTIVE = "predictive"
    DIAGNOSTIC = "diagnostic"
    DECISIVE = "decisive"


class ThinkingPhase(Enum):
    """Phases of the thinking process."""
    PERCEPTION = "perception"
    ANALYSIS = "analysis"
    REASONING = "reasoning"
    DECISION = "decision"
    REFLECTION = "reflection"


class ConfidenceLevel(Enum):
    """Confidence levels for thinking results."""
    VERY_HIGH = "very_high"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"


@dataclass
class ReasoningStep:
    """A single step in the reasoning chain."""
    step_id: str
    phase: ThinkingPhase
    thought: str
    evidence: List[str] = field(default_factory=list)
    conclusions: List[str] = field(default_factory=list)
    confidence: float = 0.0
    duration: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "phase": self.phase.name,
            "thought": self.thought,
            "evidence": self.evidence,
            "conclusions": self.conclusions,
            "confidence": self.confidence,
            "duration": self.duration,
            "metadata": self.metadata,
        }


@dataclass
class PredictedIssue:
    """A predicted potential issue."""
    issue_type: str
    description: str
    risk_level: str
    probability: float
    impact: str
    prevention_suggestion: str
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "issue_type": self.issue_type,
            "description": self.description,
            "risk_level": self.risk_level,
            "probability": self.probability,
            "impact": self.impact,
            "prevention_suggestion": self.prevention_suggestion,
            "confidence": self.confidence,
        }


@dataclass
class ThinkingResult:
    """Result of a thinking process."""
    thinking_id: str
    thinking_type: ThinkingType
    situation: str
    context: Dict[str, Any]
    reasoning_chain: List[ReasoningStep]
    conclusions: List[str]
    recommendations: List[str]
    confidence: float
    duration: float
    llm_calls: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    predicted_issues: List[PredictedIssue] = field(default_factory=list)
    alternative_approaches: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_confidence_level(self) -> ConfidenceLevel:
        if self.confidence >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif self.confidence >= 0.7:
            return ConfidenceLevel.HIGH
        elif self.confidence >= 0.5:
            return ConfidenceLevel.MEDIUM
        elif self.confidence >= 0.3:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW

    def to_dict(self) -> Dict[str, Any]:
        return {
            "thinking_id": self.thinking_id,
            "thinking_type": self.thinking_type.name,
            "situation": self.situation,
            "context": self.context,
            "reasoning_chain": [s.to_dict() for s in self.reasoning_chain],
            "conclusions": self.conclusions,
            "recommendations": self.recommendations,
            "confidence": self.confidence,
            "confidence_level": self.get_confidence_level().name,
            "duration": self.duration,
            "llm_calls": self.llm_calls,
            "timestamp": self.timestamp,
            "predicted_issues": [i.to_dict() for i in self.predicted_issues],
            "alternative_approaches": self.alternative_approaches,
            "metadata": self.metadata,
        }

    def format_for_display(self) -> str:
        lines = [
            f"[Thinking] {self.situation}",
            f"  Type: {self.thinking_type.name}",
            f"  Confidence: {self.confidence:.0%} ({self.get_confidence_level().name})",
            "",
        ]

        for step in self.reasoning_chain:
            phase_icon = {
                ThinkingPhase.PERCEPTION: "👁",
                ThinkingPhase.ANALYSIS: "🔍",
                ThinkingPhase.REASONING: "🧠",
                ThinkingPhase.DECISION: "⚡",
                ThinkingPhase.REFLECTION: "🪞",
            }.get(step.phase, "•")

            lines.append(f"  {phase_icon} [{step.phase.name}] {step.thought}")

            if step.evidence:
                for evidence in step.evidence[:3]:
                    lines.append(f"      → {evidence}")

            if step.conclusions:
                for conclusion in step.conclusions[:2]:
                    lines.append(f"      ✓ {conclusion}")

        if self.conclusions:
            lines.append("")
            lines.append("  Conclusions:")
            for conclusion in self.conclusions:
                lines.append(f"    • {conclusion}")

        if self.recommendations:
            lines.append("")
            lines.append("  Recommendations:")
            for rec in self.recommendations:
                lines.append(f"    → {rec}")

        if self.predicted_issues:
            lines.append("")
            lines.append("  Predicted Issues:")
            for issue in self.predicted_issues:
                lines.append(f"    ⚠ [{issue.risk_level}] {issue.description}")

        return "\n".join(lines)


@dataclass
class ErrorThinkingResult(ThinkingResult):
    """Result of thinking about an error."""
    error_category: str = ""
    root_cause: str = ""
    recovery_strategy: str = ""
    fix_suggestions: List[str] = field(default_factory=list)
    similar_past_errors: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result.update({
            "error_category": self.error_category,
            "root_cause": self.root_cause,
            "recovery_strategy": self.recovery_strategy,
            "fix_suggestions": self.fix_suggestions,
            "similar_past_errors": self.similar_past_errors,
        })
        return result


@dataclass
class ThinkingSession:
    """A thinking session that tracks multiple thinking processes."""
    session_id: str
    start_time: datetime
    thinking_history: List[ThinkingResult] = field(default_factory=list)
    accumulated_insights: List[str] = field(default_factory=list)
    decisions_made: List[Dict[str, Any]] = field(default_factory=list)

    def add_thinking(self, result: ThinkingResult):
        self.thinking_history.append(result)
        self.accumulated_insights.extend(result.conclusions)

    def get_summary(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "duration": (datetime.now() - self.start_time).total_seconds(),
            "total_thinking_processes": len(self.thinking_history),
            "total_insights": len(self.accumulated_insights),
            "total_decisions": len(self.decisions_made),
            "average_confidence": (
                sum(t.confidence for t in self.thinking_history) / len(self.thinking_history)
                if self.thinking_history else 0.0
            ),
        }


@component(
    component_id="thinking_engine",
    dependencies=[],
    description="Agent thinking engine for intelligent reasoning"
)
class ThinkingEngine(SimpleComponent):
    """Agent thinking engine for intelligent reasoning.

    This engine enables the agent to:
    - Think through problems transparently
    - Perform multi-step reasoning
    - Predict potential issues
    - Make informed decisions
    - Learn from thinking history

    Features:
    - Transparent thinking process
    - Multi-round reasoning
    - Context-aware decisions
    - Issue prediction
    - Thinking history tracking
    """

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        prompt_builder: Optional[Any] = None,
        enable_deep_thinking: bool = True,
        max_reasoning_steps: int = 10,
        thinking_timeout: float = 30.0,
        enable_prediction: bool = True,
        progress_callback: Optional[Callable[[str, str], None]] = None,
    ):
        super().__init__()
        self.llm_client = llm_client
        self.prompt_builder = prompt_builder
        self.enable_deep_thinking = enable_deep_thinking
        self.max_reasoning_steps = max_reasoning_steps
        self.thinking_timeout = thinking_timeout
        self.enable_prediction = enable_prediction
        self.progress_callback = progress_callback

        self._thinking_history: List[ThinkingResult] = []
        self._sessions: Dict[str, ThinkingSession] = {}
        self._insight_patterns: Dict[str, List[str]] = {}

        logger.info(
            f"[ThinkingEngine] Initialized - "
            f"Deep thinking: {enable_deep_thinking}, "
            f"Max steps: {max_reasoning_steps}, "
            f"Prediction: {enable_prediction}"
        )

    def create_session(self) -> ThinkingSession:
        session = ThinkingSession(
            session_id=str(uuid.uuid4())[:8],
            start_time=datetime.now(),
        )
        self._sessions[session.session_id] = session
        logger.info(f"[ThinkingEngine] Created thinking session: {session.session_id}")
        return session

    async def think(
        self,
        situation: str,
        context: Dict[str, Any],
        thinking_type: ThinkingType = ThinkingType.ANALYTICAL,
        session: Optional[ThinkingSession] = None,
    ) -> ThinkingResult:
        start_time = time.time()
        thinking_id = str(uuid.uuid4())[:8]

        self._notify_progress("THINKING", f"Analyzing: {situation[:50]}...")
        logger.info(f"[ThinkingEngine] Starting {thinking_type.name} thinking: {situation[:100]}")

        reasoning_chain = []
        llm_calls = 0

        perception_step = await self._phase_perception(situation, context)
        reasoning_chain.append(perception_step)

        analysis_step = await self._phase_analysis(situation, context, perception_step)
        reasoning_chain.append(analysis_step)

        if self.enable_deep_thinking and thinking_type in [
            ThinkingType.ANALYTICAL,
            ThinkingType.CRITICAL,
            ThinkingType.STRATEGIC,
        ]:
            reasoning_steps = await self._phase_reasoning(
                situation, context, perception_step, analysis_step
            )
            reasoning_chain.extend(reasoning_steps)
            llm_calls += 1

        decision_step = await self._phase_decision(
            situation, context, reasoning_chain
        )
        reasoning_chain.append(decision_step)

        reflection_step = await self._phase_reflection(
            situation, context, reasoning_chain
        )
        reasoning_chain.append(reflection_step)

        conclusions = self._extract_conclusions(reasoning_chain)
        recommendations = self._extract_recommendations(reasoning_chain)
        confidence = self._calculate_confidence(reasoning_chain)

        predicted_issues = []
        if self.enable_prediction:
            predicted_issues = await self._predict_issues(situation, context, conclusions)
            llm_calls += 1

        alternative_approaches = self._generate_alternatives(reasoning_chain)

        duration = time.time() - start_time

        result = ThinkingResult(
            thinking_id=thinking_id,
            thinking_type=thinking_type,
            situation=situation,
            context=context,
            reasoning_chain=reasoning_chain,
            conclusions=conclusions,
            recommendations=recommendations,
            confidence=confidence,
            duration=duration,
            llm_calls=llm_calls,
            predicted_issues=predicted_issues,
            alternative_approaches=alternative_approaches,
        )

        self._thinking_history.append(result)
        if session:
            session.add_thinking(result)

        self._notify_progress("THINKING_COMPLETE", f"Confidence: {confidence:.0%}")
        logger.info(
            f"[ThinkingEngine] Thinking complete - "
            f"ID: {thinking_id}, "
            f"Confidence: {confidence:.2f}, "
            f"Duration: {duration:.2f}s, "
            f"Steps: {len(reasoning_chain)}"
        )

        return result

    async def think_about_error(
        self,
        error: Exception,
        context: Dict[str, Any],
        session: Optional[ThinkingSession] = None,
    ) -> ErrorThinkingResult:
        start_time = time.time()
        thinking_id = str(uuid.uuid4())[:8]

        error_message = str(error)
        error_type = type(error).__name__

        self._notify_progress("THINKING", f"Analyzing error: {error_type}")
        logger.info(f"[ThinkingEngine] Starting error analysis: {error_type} - {error_message[:100]}")

        reasoning_chain = []

        perception_step = ReasoningStep(
            step_id=f"{thinking_id}_perception",
            phase=ThinkingPhase.PERCEPTION,
            thought=f"Encountered {error_type} error during operation",
            evidence=[
                f"Error type: {error_type}",
                f"Error message: {error_message[:200]}",
            ],
            confidence=0.9,
        )
        reasoning_chain.append(perception_step)

        error_category = self._categorize_error(error, error_message)
        analysis_step = ReasoningStep(
            step_id=f"{thinking_id}_analysis",
            phase=ThinkingPhase.ANALYSIS,
            thought=f"Error belongs to category: {error_category}",
            evidence=self._get_error_evidence(error, error_message, context),
            conclusions=[f"Error category: {error_category}"],
            confidence=0.8,
        )
        reasoning_chain.append(analysis_step)

        root_cause = await self._analyze_root_cause(error, error_message, context)
        reasoning_step = ReasoningStep(
            step_id=f"{thinking_id}_reasoning",
            phase=ThinkingPhase.REASONING,
            thought=f"Root cause analysis: {root_cause}",
            evidence=self._get_root_cause_evidence(error, context),
            conclusions=[f"Root cause: {root_cause}"],
            confidence=0.7,
        )
        reasoning_chain.append(reasoning_step)

        recovery_strategy, fix_suggestions = await self._determine_recovery_strategy(
            error, error_category, root_cause, context
        )
        decision_step = ReasoningStep(
            step_id=f"{thinking_id}_decision",
            phase=ThinkingPhase.DECISION,
            thought=f"Recommended recovery strategy: {recovery_strategy}",
            evidence=[f"Based on error category: {error_category}"],
            conclusions=[f"Strategy: {recovery_strategy}"],
            confidence=0.75,
        )
        reasoning_chain.append(decision_step)

        similar_errors = self._find_similar_errors(error, error_category)

        duration = time.time() - start_time
        confidence = self._calculate_confidence(reasoning_chain)

        result = ErrorThinkingResult(
            thinking_id=thinking_id,
            thinking_type=ThinkingType.DIAGNOSTIC,
            situation=f"Error: {error_type}",
            context=context,
            reasoning_chain=reasoning_chain,
            conclusions=[
                f"Error category: {error_category}",
                f"Root cause: {root_cause}",
                f"Recommended strategy: {recovery_strategy}",
            ],
            recommendations=fix_suggestions,
            confidence=confidence,
            duration=duration,
            llm_calls=1,
            error_category=error_category,
            root_cause=root_cause,
            recovery_strategy=recovery_strategy,
            fix_suggestions=fix_suggestions,
            similar_past_errors=similar_errors,
        )

        self._thinking_history.append(result)
        if session:
            session.add_thinking(result)

        self._notify_progress("ERROR_ANALYZED", f"Strategy: {recovery_strategy}")
        logger.info(
            f"[ThinkingEngine] Error analysis complete - "
            f"Category: {error_category}, "
            f"Strategy: {recovery_strategy}, "
            f"Confidence: {confidence:.2f}"
        )

        return result

    async def predict_issues(
        self,
        current_state: Dict[str, Any],
        context: Dict[str, Any],
    ) -> List[PredictedIssue]:
        if not self.enable_prediction:
            return []

        self._notify_progress("PREDICTING", "Analyzing potential issues...")
        logger.info("[ThinkingEngine] Predicting potential issues")

        predicted = []

        test_code = current_state.get("test_code", "")
        if test_code:
            if "@Mock" in test_code and "when(" not in test_code:
                predicted.append(PredictedIssue(
                    issue_type="UNSTUBBED_MOCK",
                    description="Mocks declared but not stubbed - may cause NullPointerException",
                    risk_level="HIGH",
                    probability=0.8,
                    impact="Test failures due to unmocked dependencies",
                    prevention_suggestion="Add when().thenReturn() for mock behavior",
                    confidence=0.85,
                ))

            if "assertThrows" not in test_code and "try-catch" not in test_code:
                predicted.append(PredictedIssue(
                    issue_type="MISSING_EXCEPTION_TEST",
                    description="No exception handling tests detected",
                    risk_level="MEDIUM",
                    probability=0.6,
                    impact="May miss important error scenarios",
                    prevention_suggestion="Add tests for exception cases using assertThrows",
                    confidence=0.7,
                ))

        coverage = current_state.get("coverage", 0)
        if coverage < 0.5:
            predicted.append(PredictedIssue(
                issue_type="LOW_COVERAGE",
                description=f"Current coverage ({coverage:.0%}) is below target",
                risk_level="MEDIUM",
                probability=0.9,
                impact="May miss important code paths",
                prevention_suggestion="Add more test cases to cover uncovered branches",
                confidence=0.9,
            ))

        predicted.extend(self._predict_from_history())

        logger.info(f"[ThinkingEngine] Predicted {len(predicted)} potential issues")
        return predicted

    async def _phase_perception(
        self,
        situation: str,
        context: Dict[str, Any],
    ) -> ReasoningStep:
        step_id = str(uuid.uuid4())[:8]

        evidence = []
        evidence.append(f"Situation: {situation[:200]}")

        key_context = {k: str(v)[:100] for k, v in list(context.items())[:5]}
        evidence.append(f"Context keys: {', '.join(key_context.keys())}")

        thought = "Gathering information about the current situation"
        if "error" in context:
            thought = f"Detected error context: {type(context['error']).__name__}"
        elif "test_code" in context:
            thought = "Analyzing test code structure"
        elif "coverage" in context:
            thought = f"Current coverage: {context.get('coverage', 0):.0%}"

        return ReasoningStep(
            step_id=f"{step_id}_perception",
            phase=ThinkingPhase.PERCEPTION,
            thought=thought,
            evidence=evidence,
            confidence=0.9,
        )

    async def _phase_analysis(
        self,
        situation: str,
        context: Dict[str, Any],
        perception_step: ReasoningStep,
    ) -> ReasoningStep:
        step_id = str(uuid.uuid4())[:8]

        conclusions = []
        evidence = []

        if "error" in context:
            error = context["error"]
            error_type = type(error).__name__
            conclusions.append(f"Error type identified: {error_type}")
            evidence.append(f"Error message: {str(error)[:100]}")

        if "test_code" in context:
            test_code = context["test_code"]
            test_count = test_code.count("@Test")
            mock_count = test_code.count("@Mock")
            conclusions.append(f"Found {test_count} test methods, {mock_count} mocks")
            evidence.append(f"Code length: {len(test_code)} chars")

        if "coverage" in context:
            coverage = context["coverage"]
            if coverage < 0.5:
                conclusions.append("Coverage below target - needs improvement")
            else:
                conclusions.append("Coverage acceptable")

        thought = "Analyzing the gathered information"
        if conclusions:
            thought = conclusions[0]

        return ReasoningStep(
            step_id=f"{step_id}_analysis",
            phase=ThinkingPhase.ANALYSIS,
            thought=thought,
            evidence=evidence,
            conclusions=conclusions,
            confidence=0.8,
        )

    async def _phase_reasoning(
        self,
        situation: str,
        context: Dict[str, Any],
        perception_step: ReasoningStep,
        analysis_step: ReasoningStep,
    ) -> List[ReasoningStep]:
        steps = []
        step_id = str(uuid.uuid4())[:8]

        if self.llm_client and self.prompt_builder:
            try:
                prompt = self._build_reasoning_prompt(situation, context, analysis_step)
                response = await self.llm_client.generate(prompt)
                reasoning_result = self._parse_reasoning_response(response)

                step = ReasoningStep(
                    step_id=f"{step_id}_llm_reasoning",
                    phase=ThinkingPhase.REASONING,
                    thought=reasoning_result.get("thought", "LLM reasoning"),
                    evidence=reasoning_result.get("evidence", []),
                    conclusions=reasoning_result.get("conclusions", []),
                    confidence=reasoning_result.get("confidence", 0.7),
                )
                steps.append(step)
            except Exception as e:
                logger.warning(f"[ThinkingEngine] LLM reasoning failed: {e}")

        if not steps:
            step = ReasoningStep(
                step_id=f"{step_id}_reasoning",
                phase=ThinkingPhase.REASONING,
                thought="Applying standard reasoning patterns",
                evidence=["Using built-in analysis patterns"],
                conclusions=["Proceeding with standard approach"],
                confidence=0.6,
            )
            steps.append(step)

        return steps

    async def _phase_decision(
        self,
        situation: str,
        context: Dict[str, Any],
        reasoning_chain: List[ReasoningStep],
    ) -> ReasoningStep:
        step_id = str(uuid.uuid4())[:8]

        all_conclusions = []
        for step in reasoning_chain:
            all_conclusions.extend(step.conclusions)

        decision = "Continue with standard approach"
        if "error" in context:
            decision = "Apply error recovery strategy"
        elif "coverage" in context and context.get("coverage", 0) < 0.5:
            decision = "Generate additional tests to improve coverage"

        recommendations = []
        if "error" in context:
            recommendations.append("Analyze error root cause")
            recommendations.append("Apply appropriate recovery strategy")
        elif context.get("coverage", 0) < 0.5:
            recommendations.append("Identify uncovered code paths")
            recommendations.append("Generate targeted tests")

        return ReasoningStep(
            step_id=f"{step_id}_decision",
            phase=ThinkingPhase.DECISION,
            thought=decision,
            evidence=[f"Based on {len(all_conclusions)} conclusions"],
            conclusions=[decision],
            confidence=0.75,
            metadata={"recommendations": recommendations},
        )

    async def _phase_reflection(
        self,
        situation: str,
        context: Dict[str, Any],
        reasoning_chain: List[ReasoningStep],
    ) -> ReasoningStep:
        step_id = str(uuid.uuid4())[:8]

        avg_confidence = sum(s.confidence for s in reasoning_chain) / len(reasoning_chain)

        reflections = []
        if avg_confidence < 0.6:
            reflections.append("Low confidence - consider gathering more information")
        elif avg_confidence > 0.8:
            reflections.append("High confidence in analysis")

        step_count = len(reasoning_chain)
        if step_count < 3:
            reflections.append("Quick analysis - may need deeper investigation")

        return ReasoningStep(
            step_id=f"{step_id}_reflection",
            phase=ThinkingPhase.REFLECTION,
            thought=f"Reflection on {step_count} reasoning steps",
            evidence=[f"Average confidence: {avg_confidence:.2f}"],
            conclusions=reflections if reflections else ["Analysis complete"],
            confidence=avg_confidence,
        )

    async def _predict_issues(
        self,
        situation: str,
        context: Dict[str, Any],
        conclusions: List[str],
    ) -> List[PredictedIssue]:
        return await self.predict_issues(context, context)

    def _categorize_error(self, error: Exception, error_message: str) -> str:
        error_type = type(error).__name__
        message_lower = error_message.lower()

        if "compilation" in message_lower or "cannot find symbol" in message_lower:
            return "COMPILATION_ERROR"
        elif "assertion" in message_lower or "test" in message_lower:
            return "TEST_FAILURE"
        elif "timeout" in message_lower:
            return "TIMEOUT"
        elif "network" in message_lower or "connection" in message_lower:
            return "NETWORK"
        elif "nullpointer" in message_lower or "null" in message_lower:
            return "NULL_POINTER"
        elif "import" in message_lower or "package" in message_lower:
            return "IMPORT_ERROR"
        else:
            return "UNKNOWN"

    def _get_error_evidence(
        self,
        error: Exception,
        error_message: str,
        context: Dict[str, Any],
    ) -> List[str]:
        evidence = []
        evidence.append(f"Error type: {type(error).__name__}")
        evidence.append(f"Message: {error_message[:200]}")

        if "test_code" in context:
            test_code = context["test_code"]
            evidence.append(f"Test code length: {len(test_code)}")

        return evidence

    async def _analyze_root_cause(
        self,
        error: Exception,
        error_message: str,
        context: Dict[str, Any],
    ) -> str:
        category = self._categorize_error(error, error_message)

        root_causes = {
            "COMPILATION_ERROR": "Missing imports or type mismatches in generated code",
            "TEST_FAILURE": "Test assertions failing - logic or mock configuration issue",
            "TIMEOUT": "Operation took too long - may need optimization or async handling",
            "NETWORK": "Network connectivity issue - may be transient",
            "NULL_POINTER": "Null reference accessed - missing null check or mock setup",
            "IMPORT_ERROR": "Missing dependency or incorrect import statement",
        }

        base_cause = root_causes.get(category, "Unknown root cause")

        if self.llm_client:
            try:
                prompt = f"""Analyze the root cause of this error:
Error Type: {type(error).__name__}
Error Message: {error_message[:500]}
Category: {category}

Provide a brief root cause analysis (1-2 sentences)."""
                response = await self.llm_client.generate(prompt)
                if response and len(response) < 500:
                    return response.strip()
            except Exception as e:
                logger.warning(f"[ThinkingEngine] LLM root cause analysis failed: {e}")

        return base_cause

    def _get_root_cause_evidence(
        self,
        error: Exception,
        context: Dict[str, Any],
    ) -> List[str]:
        evidence = []
        if hasattr(error, "__traceback__") and error.__traceback__:
            tb = error.__traceback__
            while tb.tb_next:
                tb = tb.tb_next
            evidence.append(f"Error location: line {tb.tb_lineno}")
        return evidence

    async def _determine_recovery_strategy(
        self,
        error: Exception,
        error_category: str,
        root_cause: str,
        context: Dict[str, Any],
    ) -> tuple:
        strategies = {
            "COMPILATION_ERROR": ("ANALYZE_AND_FIX", [
                "Check and fix import statements",
                "Verify type compatibility",
                "Add missing dependencies",
            ]),
            "TEST_FAILURE": ("ANALYZE_AND_FIX", [
                "Review test assertions",
                "Check mock configurations",
                "Verify test data setup",
            ]),
            "TIMEOUT": ("RETRY_WITH_BACKOFF", [
                "Increase timeout threshold",
                "Optimize slow operations",
                "Consider async handling",
            ]),
            "NETWORK": ("RETRY_WITH_BACKOFF", [
                "Retry with exponential backoff",
                "Check network connectivity",
                "Use fallback if available",
            ]),
            "NULL_POINTER": ("ANALYZE_AND_FIX", [
                "Add null checks",
                "Initialize mock objects",
                "Verify dependency injection",
            ]),
            "IMPORT_ERROR": ("INSTALL_DEPENDENCIES", [
                "Add missing dependency to pom.xml",
                "Verify import statement",
                "Check package availability",
            ]),
        }

        strategy, suggestions = strategies.get(
            error_category,
            ("RETRY", ["Try again", "Check error details"])
        )

        return strategy, suggestions

    def _find_similar_errors(
        self,
        error: Exception,
        error_category: str,
    ) -> List[Dict[str, Any]]:
        similar = []
        error_type = type(error).__name__

        for past_thinking in self._thinking_history[-20:]:
            if isinstance(past_thinking, ErrorThinkingResult):
                if past_thinking.error_category == error_category:
                    similar.append({
                        "thinking_id": past_thinking.thinking_id,
                        "error_category": past_thinking.error_category,
                        "strategy_used": past_thinking.recovery_strategy,
                        "success": past_thinking.confidence > 0.7,
                    })

        return similar[:5]

    def _predict_from_history(self) -> List[PredictedIssue]:
        predicted = []

        if len(self._thinking_history) > 5:
            recent_failures = sum(
                1 for t in self._thinking_history[-5:]
                if t.confidence < 0.5
            )
            if recent_failures >= 3:
                predicted.append(PredictedIssue(
                    issue_type="REPEATED_FAILURES",
                    description="Multiple recent failures detected - may indicate systemic issue",
                    risk_level="HIGH",
                    probability=0.8,
                    impact="Continued failures and wasted resources",
                    prevention_suggestion="Consider resetting or escalating to user",
                    confidence=0.85,
                ))

        return predicted

    def _extract_conclusions(self, reasoning_chain: List[ReasoningStep]) -> List[str]:
        conclusions = []
        for step in reasoning_chain:
            conclusions.extend(step.conclusions)
        return list(dict.fromkeys(conclusions))

    def _extract_recommendations(self, reasoning_chain: List[ReasoningStep]) -> List[str]:
        recommendations = []
        for step in reasoning_chain:
            if "recommendations" in step.metadata:
                recommendations.extend(step.metadata["recommendations"])
        return list(dict.fromkeys(recommendations))

    def _calculate_confidence(self, reasoning_chain: List[ReasoningStep]) -> float:
        if not reasoning_chain:
            return 0.0

        weights = {
            ThinkingPhase.PERCEPTION: 0.1,
            ThinkingPhase.ANALYSIS: 0.2,
            ThinkingPhase.REASONING: 0.3,
            ThinkingPhase.DECISION: 0.25,
            ThinkingPhase.REFLECTION: 0.15,
        }

        total_weight = 0.0
        weighted_confidence = 0.0

        for step in reasoning_chain:
            weight = weights.get(step.phase, 0.1)
            weighted_confidence += step.confidence * weight
            total_weight += weight

        return weighted_confidence / total_weight if total_weight > 0 else 0.0

    def _generate_alternatives(self, reasoning_chain: List[ReasoningStep]) -> List[str]:
        alternatives = []

        for step in reasoning_chain:
            if step.phase == ThinkingPhase.DECISION:
                alternatives.append("Consider alternative approaches if primary fails")
                break

        return alternatives

    def _build_reasoning_prompt(
        self,
        situation: str,
        context: Dict[str, Any],
        analysis_step: ReasoningStep,
    ) -> str:
        return f"""Think through this situation step by step.

Situation: {situation}

Analysis so far:
{chr(10).join(f'- {c}' for c in analysis_step.conclusions)}

Provide your reasoning in this format:
THOUGHT: [Your main thought]
EVIDENCE: [Supporting evidence, one per line]
CONCLUSIONS: [Your conclusions, one per line]
CONFIDENCE: [0.0-1.0]

Focus on identifying root causes and potential solutions."""

    def _parse_reasoning_response(self, response: str) -> Dict[str, Any]:
        result = {
            "thought": "",
            "evidence": [],
            "conclusions": [],
            "confidence": 0.7,
        }

        lines = response.strip().split("\n")
        current_section = None

        for line in lines:
            line = line.strip()
            if line.startswith("THOUGHT:"):
                result["thought"] = line[8:].strip()
                current_section = "thought"
            elif line.startswith("EVIDENCE:"):
                current_section = "evidence"
            elif line.startswith("CONCLUSIONS:"):
                current_section = "conclusions"
            elif line.startswith("CONFIDENCE:"):
                try:
                    result["confidence"] = float(line[11:].strip())
                except ValueError:
                    pass
            elif line.startswith("-") or line.startswith("•"):
                content = line[1:].strip()
                if current_section == "evidence":
                    result["evidence"].append(content)
                elif current_section == "conclusions":
                    result["conclusions"].append(content)

        return result

    def _notify_progress(self, status: str, message: str):
        if self.progress_callback:
            try:
                self.progress_callback(status, message)
            except Exception as e:
                logger.warning(f"[ThinkingEngine] Progress callback failed: {e}")

    def get_thinking_stats(self) -> Dict[str, Any]:
        if not self._thinking_history:
            return {"total": 0}

        return {
            "total": len(self._thinking_history),
            "by_type": {
                t.name: sum(1 for x in self._thinking_history if x.thinking_type == t)
                for t in ThinkingType
            },
            "average_confidence": sum(t.confidence for t in self._thinking_history) / len(self._thinking_history),
            "average_duration": sum(t.duration for t in self._thinking_history) / len(self._thinking_history),
            "total_llm_calls": sum(t.llm_calls for t in self._thinking_history),
            "sessions": len(self._sessions),
        }

    def get_recent_thinking(self, limit: int = 10) -> List[ThinkingResult]:
        return self._thinking_history[-limit:]

    def clear_history(self):
        self._thinking_history.clear()
        self._sessions.clear()
        logger.info("[ThinkingEngine] History cleared")
