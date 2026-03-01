"""Enhanced ReAct Agent with full P0/P1/P2/P3 integration.

This module provides an enhanced agent that deeply integrates all enhancement layers:
- P0: Context management, quality evaluation, partial success handling
- P1: Prompt optimization, error learning, tool orchestration
- P2: Multi-agent collaboration
- P3: Error prediction, strategy optimization, sandbox execution, user interaction, smart analysis
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime

from .react_agent import ReActAgent
from .multi_agent import (
    AgentCoordinator, AgentCapability, AgentRole,
    MessageBus, SharedKnowledgeBase, ExperienceReplay
)
from .prompt_optimizer import PromptOptimizer, ModelType
from .context_manager import ContextManager, CompressionStrategy
from .generation_evaluator import GenerationEvaluator
from .partial_success_handler import PartialSuccessHandler
from .user_interaction import UserInteractionHandler, create_repair_suggestion, UserChoice
from ..core.metrics import MetricsCollector, get_metrics
from ..core.protocols import AgentState, AgentResult
from ..core.error_predictor import ErrorPredictor, ErrorType
from ..core.adaptive_strategy import AdaptiveStrategyManager, ErrorCategory
from ..core.sandbox_executor import SandboxExecutor, SecurityLevel
from ..core.smart_analyzer import SmartCodeAnalyzer
from ..memory.working_memory import WorkingMemory
from ..llm.client import LLMClient
from ..core.container import Container, get_container

logger = logging.getLogger(__name__)


@dataclass
class EnhancedAgentConfig:
    """Configuration for EnhancedAgent."""
    # P0 Configuration
    context_max_tokens: int = 8000
    context_target_tokens: int = 6000
    context_strategy: CompressionStrategy = CompressionStrategy.HYBRID
    
    # P1 Configuration
    enable_prompt_optimization: bool = True
    enable_ab_testing: bool = False
    ab_test_id: Optional[str] = None
    
    # P2 Configuration
    enable_multi_agent: bool = False
    multi_agent_workers: int = 3
    task_allocation_strategy: str = "capability_match"
    
    # P3 Configuration
    enable_error_prediction: bool = True
    enable_strategy_optimization: bool = True
    enable_sandbox_execution: bool = True
    enable_user_interaction: bool = True
    enable_smart_analysis: bool = True
    sandbox_security_level: SecurityLevel = SecurityLevel.MODERATE
    
    # Performance
    enable_metrics: bool = True
    metrics_report_interval: int = 300  # 5 minutes
    
    # Model
    model_name: str = "gpt-4"


class EnhancedAgent(ReActAgent):
    """Enhanced ReAct Agent with full integration of all enhancement layers.
    
    Features:
    - Deep P0/P1/P2/P3 component integration
    - Automatic metrics collection
    - Multi-agent collaboration support
    - Performance monitoring
    - Adaptive optimization
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        working_memory: WorkingMemory,
        project_path: str,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        container: Optional[Container] = None,
        config: Optional[EnhancedAgentConfig] = None
    ):
        """Initialize enhanced agent.
        
        Args:
            llm_client: LLM client
            working_memory: Working memory
            project_path: Project path
            progress_callback: Progress callback
            container: DI container
            config: Enhanced agent configuration
        """
        self.config = config or EnhancedAgentConfig()
        
        # Initialize metrics
        self.metrics = get_metrics() if self.config.enable_metrics else MetricsCollector(enabled=False)
        
        # Initialize multi-agent components if enabled
        self.agent_coordinator: Optional[AgentCoordinator] = None
        self.message_bus: Optional[MessageBus] = None
        self.shared_knowledge: Optional[SharedKnowledgeBase] = None
        self.experience_replay: Optional[ExperienceReplay] = None
        
        if self.config.enable_multi_agent:
            self._init_multi_agent()
        
        # Initialize P3 components
        self.error_predictor: Optional[ErrorPredictor] = None
        self.strategy_manager: Optional[AdaptiveStrategyManager] = None
        self.sandbox_executor: Optional[SandboxExecutor] = None
        self.user_interaction: Optional[UserInteractionHandler] = None
        self.smart_analyzer: Optional[SmartCodeAnalyzer] = None
        
        self._init_p3_components()
        
        # Call parent init with model name for P1 prompt optimization
        super().__init__(
            llm_client=llm_client,
            working_memory=working_memory,
            project_path=project_path,
            progress_callback=progress_callback,
            container=container,
            model_name=self.config.model_name,
            ab_test_id=self.config.ab_test_id
        )
        
        # Start metrics reporting
        if self.config.enable_metrics:
            asyncio.create_task(self._metrics_reporting_loop())
        
        logger.info(f"[EnhancedAgent] Initialized with config: {self.config}")
    
    def _init_multi_agent(self):
        """Initialize multi-agent collaboration components."""
        self.message_bus = MessageBus()
        self.shared_knowledge = SharedKnowledgeBase()
        self.experience_replay = ExperienceReplay()
        
        self.agent_coordinator = AgentCoordinator(
            message_bus=self.message_bus,
            knowledge_base=self.shared_knowledge,
            experience_replay=self.experience_replay
        )
        
        logger.info("[EnhancedAgent] Multi-agent components initialized")
    
    def _init_p3_components(self):
        """Initialize P3 advanced capability components."""
        # Error Predictor
        if self.config.enable_error_prediction:
            self.error_predictor = ErrorPredictor()
            logger.info("[EnhancedAgent] Error predictor initialized")
        
        # Adaptive Strategy Manager
        if self.config.enable_strategy_optimization:
            self.strategy_manager = AdaptiveStrategyManager()
            logger.info("[EnhancedAgent] Strategy manager initialized")
        
        # Sandbox Executor
        if self.config.enable_sandbox_execution:
            self.sandbox_executor = SandboxExecutor(
                security_level=self.config.sandbox_security_level
            )
            logger.info("[EnhancedAgent] Sandbox executor initialized")
        
        # User Interaction Handler
        if self.config.enable_user_interaction:
            self.user_interaction = UserInteractionHandler()
            logger.info("[EnhancedAgent] User interaction handler initialized")
        
        # Smart Code Analyzer
        if self.config.enable_smart_analysis:
            self.smart_analyzer = SmartCodeAnalyzer()
            logger.info("[EnhancedAgent] Smart code analyzer initialized")
    
    async def start_multi_agent_system(self):
        """Start the multi-agent collaboration system."""
        if not self.config.enable_multi_agent or not self.agent_coordinator:
            logger.warning("[EnhancedAgent] Multi-agent not enabled")
            return
        
        # Start coordinator
        await self.agent_coordinator.start()
        
        # Register specialized agents
        await self._register_specialized_agents()
        
        logger.info("[EnhancedAgent] Multi-agent system started")
    
    async def _register_specialized_agents(self):
        """Register specialized agents with the coordinator."""
        from .multi_agent.specialized_agent import SpecializedAgent
        
        # This would create and start actual specialized agents
        # For now, just register placeholder agents
        
        agents_config = [
            ("designer_1", {AgentCapability.TEST_DESIGN}, AgentRole.DESIGNER),
            ("implementer_1", {AgentCapability.TEST_IMPLEMENTATION}, AgentRole.IMPLEMENTER),
            ("reviewer_1", {AgentCapability.TEST_REVIEW}, AgentRole.REVIEWER),
            ("fixer_1", {AgentCapability.ERROR_FIXING}, AgentRole.FIXER),
        ]
        
        for agent_id, capabilities, role in agents_config:
            self.agent_coordinator.register_agent(agent_id, capabilities, role)
        
        logger.info(f"[EnhancedAgent] Registered {len(agents_config)} specialized agents")
    
    async def generate_tests(self, target_file: str) -> AgentResult:
        """Generate tests with full metrics collection.
        
        Args:
            target_file: Target file path
            
        Returns:
            AgentResult with generation results
        """
        with self.metrics.time_operation("generate_tests", {"target_file": target_file}):
            try:
                # Use multi-agent if enabled
                if self.config.enable_multi_agent and self.agent_coordinator:
                    return await self._generate_tests_multi_agent(target_file)
                
                # Use standard single-agent approach
                return await super().generate_tests(target_file)
                
            except Exception as e:
                self.metrics.record_error("generation", "generate_tests", recovered=False)
                logger.exception(f"[EnhancedAgent] Test generation failed: {e}")
                return AgentResult(
                    success=False,
                    message=f"Test generation failed: {str(e)}",
                    state=AgentState.FAILED
                )
    
    async def _generate_tests_multi_agent(self, target_file: str) -> AgentResult:
        """Generate tests using multi-agent collaboration.
        
        Args:
            target_file: Target file path
            
        Returns:
            AgentResult
        """
        logger.info(f"[EnhancedAgent] Using multi-agent for {target_file}")
        
        # Submit tasks to coordinator
        design_task = await self.agent_coordinator.submit_task(
            task_type="design_tests",
            payload={"target_file": target_file, "class_info": self.target_class_info},
            priority=1
        )
        
        # Wait for design completion
        design_success = await self.agent_coordinator.wait_for_task(design_task, timeout=60.0)
        
        if not design_success:
            logger.warning("[EnhancedAgent] Design task failed, falling back to single-agent")
            return await super().generate_tests(target_file)
        
        # Submit implementation task
        impl_task = await self.agent_coordinator.submit_task(
            task_type="implement_tests",
            payload={"target_file": target_file, "design_task_id": design_task},
            priority=2,
            dependencies=[design_task]
        )
        
        # Wait for implementation
        impl_success = await self.agent_coordinator.wait_for_task(impl_task, timeout=120.0)
        
        if impl_success:
            # Get task result
            task_status = self.agent_coordinator.get_task_status(impl_task)
            
            return AgentResult(
                success=True,
                message="Tests generated via multi-agent collaboration",
                test_file=task_status.get("result", {}).get("output", {}).get("test_file"),
                state=AgentState.COMPLETED
            )
        else:
            logger.warning("[EnhancedAgent] Implementation task failed, falling back")
            return await super().generate_tests(target_file)
    
    async def _generate_initial_tests(self, use_streaming: bool = True) -> Any:
        """Generate initial tests with enhanced metrics and optimization.
        
        Args:
            use_streaming: Whether to use streaming
            
        Returns:
            StepResult
        """
        with self.metrics.time_operation("generate_initial_tests"):
            # Record LLM call start
            llm_start = asyncio.get_event_loop().time()
            
            try:
                # Call parent implementation
                result = await super()._generate_initial_tests(use_streaming)
                
                # Record metrics
                llm_time = asyncio.get_event_loop().time() - llm_start
                self.metrics.record_llm_call(
                    tokens=result.data.get("tokens", 0) if result.data else 0,
                    time_taken=llm_time,
                    success=result.success
                )
                
                # Record to experience replay if available
                if self.experience_replay:
                    self.experience_replay.add_experience(
                        task_type="generate_initial_tests",
                        context={"model": self.config.model_name},
                        action="generate",
                        outcome="success" if result.success else "failure",
                        reward=1.0 if result.success else -0.5,
                        agent_id="main_agent"
                    )
                
                return result
                
            except Exception as e:
                llm_time = asyncio.get_event_loop().time() - llm_start
                self.metrics.record_llm_call(tokens=0, time_taken=llm_time, success=False)
                self.metrics.record_error("generation", "initial_tests", recovered=False)
                raise
    
    async def _compile_with_recovery(self) -> bool:
        """Compile with enhanced error tracking.
        
        Returns:
            True if successful
        """
        with self.metrics.time_operation("compile_with_recovery"):
            result = await super()._compile_with_recovery()
            
            if not result:
                self.metrics.record_error("compilation", "compile", recovered=True)
            
            return result
    
    async def _run_tests_with_recovery(self) -> bool:
        """Run tests with enhanced error tracking.
        
        Returns:
            True if successful
        """
        with self.metrics.time_operation("run_tests_with_recovery"):
            result = await super()._run_tests_with_recovery()
            
            if not result:
                self.metrics.record_error("test_execution", "run_tests", recovered=True)
            
            return result
    
    async def _try_recover(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Try to recover with P3 strategy optimization and metrics tracking.
        
        Args:
            error: Error to recover from
            context: Error context
            
        Returns:
            Recovery result
        """
        with self.metrics.time_operation("error_recovery", context):
            # P3: Use adaptive strategy selection if available
            if self.strategy_manager:
                error_category = self._categorize_error(error)
                strategies = self._get_available_strategies(error_category)
                
                if strategies:
                    selection = self.strategy_manager.select_strategy(
                        error_category=error_category,
                        available_strategies=strategies,
                        context=context
                    )
                    
                    logger.info(f"[EnhancedAgent] Selected strategy: {selection.strategy.name} "
                               f"(confidence: {selection.confidence:.2f})")
                    
                    # Use the selected strategy
                    result = await self._execute_strategy(selection.strategy, error, context)
                    
                    # Record outcome
                    success = result.get("should_continue", False)
                    self.strategy_manager.record_outcome(
                        strategy_name=selection.strategy.name,
                        error_category=error_category,
                        success=success,
                        execution_time=result.get("execution_time", 0),
                        context=context
                    )
                    
                    self.metrics.record_error(
                        category="recovery",
                        step=context.get("step", "unknown"),
                        recovered=success
                    )
                    
                    return result
            
            # Fallback to parent implementation
            result = await super()._try_recover(error, context)
            
            # Record recovery metrics
            success = result.get("should_continue", False)
            self.metrics.record_error(
                category="recovery",
                step=context.get("step", "unknown"),
                recovered=success
            )
            
            return result
    
    def _categorize_error(self, error: Exception) -> ErrorCategory:
        """Categorize error for strategy selection."""
        error_msg = str(error).lower()
        
        if "syntax" in error_msg or "parse" in error_msg:
            return ErrorCategory.COMPILATION_ERROR
        elif "import" in error_msg or "module" in error_msg:
            return ErrorCategory.DEPENDENCY_ERROR
        elif "timeout" in error_msg or "deadline" in error_msg:
            return ErrorCategory.TIMEOUT_ERROR
        elif "memory" in error_msg or "resource" in error_msg:
            return ErrorCategory.RESOURCE_ERROR
        elif "assert" in error_msg or "test" in error_msg:
            return ErrorCategory.TEST_FAILURE
        else:
            return ErrorCategory.UNKNOWN_ERROR
    
    def _get_available_strategies(self, error_category: ErrorCategory) -> List[Any]:
        """Get available recovery strategies for error category."""
        from ..core.adaptive_strategy import RecoveryStrategy
        
        strategies = []
        
        if error_category == ErrorCategory.COMPILATION_ERROR:
            strategies.append(RecoveryStrategy(
                name="syntax_fix",
                description="Fix syntax errors",
                applicable_errors={ErrorCategory.COMPILATION_ERROR},
                parameters={"max_attempts": 3}
            ))
        elif error_category == ErrorCategory.TEST_FAILURE:
            strategies.append(RecoveryStrategy(
                name="test_fix",
                description="Fix test failures",
                applicable_errors={ErrorCategory.TEST_FAILURE},
                parameters={"max_attempts": 5}
            ))
        elif error_category == ErrorCategory.DEPENDENCY_ERROR:
            strategies.append(RecoveryStrategy(
                name="dependency_fix",
                description="Fix dependency issues",
                applicable_errors={ErrorCategory.DEPENDENCY_ERROR},
                parameters={"auto_install": True}
            ))
        
        # Always add generic retry
        strategies.append(RecoveryStrategy(
            name="generic_retry",
            description="Generic retry strategy",
            applicable_errors=set(ErrorCategory),
            parameters={"max_attempts": 3}
        ))
        
        return strategies
    
    async def _execute_strategy(
        self,
        strategy: Any,
        error: Exception,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a recovery strategy."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            if strategy.name == "syntax_fix":
                result = await self._fix_syntax_error(error, context)
            elif strategy.name == "test_fix":
                result = await self._fix_test_failure(error, context)
            elif strategy.name == "dependency_fix":
                result = await self._fix_dependency_error(error, context)
            else:
                # Generic retry
                result = await super()._try_recover(error, context)
            
            result["execution_time"] = asyncio.get_event_loop().time() - start_time
            return result
            
        except Exception as e:
            return {
                "should_continue": False,
                "error": str(e),
                "execution_time": asyncio.get_event_loop().time() - start_time
            }
    
    async def _fix_syntax_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Fix syntax errors."""
        # Implementation would use LLM to fix syntax
        return await super()._try_recover(error, context)
    
    async def _fix_test_failure(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Fix test failures."""
        return await super()._try_recover(error, context)
    
    async def _fix_dependency_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Fix dependency errors."""
        return await super()._try_recover(error, context)
    
    async def predict_and_prevent_errors(self, code: str, file_path: str) -> Dict[str, Any]:
        """P3: Predict potential errors before compilation.
        
        Args:
            code: Code to analyze
            file_path: File path
            
        Returns:
            Prediction results
        """
        if not self.error_predictor:
            return {"enabled": False}
        
        with self.metrics.time_operation("error_prediction"):
            prediction = self.error_predictor.predict_compilation_errors(code, file_path)
            
            if prediction.has_errors:
                logger.warning(f"[EnhancedAgent] Predicted {len(prediction.predicted_errors)} errors")
                
                # Try to auto-fix high-confidence predictions
                for error in prediction.predicted_errors:
                    if error.confidence > 0.8:
                        suggestion = self.error_predictor.suggest_fix(error, code)
                        if suggestion:
                            logger.info(f"[EnhancedAgent] Suggested fix for {error.error_type}: {suggestion['description']}")
            
            return {
                "enabled": True,
                "has_errors": prediction.has_errors,
                "predicted_errors": [
                    {
                        "type": e.error_type.value,
                        "severity": e.severity.value,
                        "confidence": e.confidence,
                        "message": e.message
                    }
                    for e in prediction.predicted_errors
                ],
                "overall_confidence": prediction.overall_confidence
            }
    
    async def execute_in_sandbox(
        self,
        code: str,
        class_name: str,
        method_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """P3: Execute code in sandboxed environment.
        
        Args:
            code: Code to execute
            class_name: Class name
            method_name: Optional method name
            
        Returns:
            Execution results
        """
        if not self.sandbox_executor:
            return {"enabled": False, "message": "Sandbox not enabled"}
        
        with self.metrics.time_operation("sandbox_execution"):
            result = await self.sandbox_executor.execute_sandboxed(
                code=code,
                class_name=class_name,
                method_name=method_name
            )
            
            return {
                "enabled": True,
                "status": result.status.value,
                "success": result.status.value == "success",
                "output": result.output,
                "execution_time": result.execution_time,
                "security_violations": [
                    {"type": v.violation_type, "severity": v.severity}
                    for v in result.security_violations
                ] if result.security_violations else []
            }
    
    async def request_user_confirmation(
        self,
        title: str,
        description: str,
        code_before: Optional[str] = None,
        code_after: Optional[str] = None,
        confidence: float = 0.5
    ) -> Dict[str, Any]:
        """P3: Request user confirmation for changes.
        
        Args:
            title: Suggestion title
            description: Description
            code_before: Code before change
            code_after: Code after change
            confidence: Confidence score
            
        Returns:
            User choice result
        """
        if not self.user_interaction:
            # Auto-accept if user interaction not enabled
            return {"choice": "accept", "auto": True}
        
        suggestion = create_repair_suggestion(
            title=title,
            description=description,
            code_before=code_before,
            code_after=code_after,
            confidence=confidence,
            estimated_impact="medium" if confidence > 0.7 else "high"
        )
        
        choice, feedback = await self.user_interaction.request_confirmation(
            suggestion=suggestion,
            context={"agent": "enhanced", "timestamp": datetime.now().isoformat()},
            auto_decide=confidence > 0.95  # Auto-decide for very high confidence
        )
        
        return {
            "choice": choice.value,
            "feedback": feedback,
            "auto": confidence > 0.95
        }
    
    async def analyze_code_semantics(self, file_path: str) -> Dict[str, Any]:
        """P3: Analyze code semantics and dependencies.
        
        Args:
            file_path: File to analyze
            
        Returns:
            Analysis results
        """
        if not self.smart_analyzer:
            return {"enabled": False}
        
        with self.metrics.time_operation("semantic_analysis"):
            # Quick file analysis
            from ..core.smart_analyzer import quick_analyze_file
            result = quick_analyze_file(file_path)
            
            return {
                "enabled": True,
                "file": result["file"],
                "entities_count": len(result["entities"]),
                "entities": result["entities"]
            }
    
    async def analyze_change_impact(self, entity_id: str) -> Dict[str, Any]:
        """P3: Analyze impact of code changes.
        
        Args:
            entity_id: Entity to analyze
            
        Returns:
            Impact analysis results
        """
        if not self.smart_analyzer or not self.smart_analyzer.impact_analyzer:
            return {"enabled": False}
        
        with self.metrics.time_operation("impact_analysis"):
            impact = self.smart_analyzer.analyze_change_impact(entity_id)
            
            return {
                "enabled": True,
                "changed_entity": impact.changed_entity_id,
                "directly_affected_count": len(impact.directly_affected),
                "indirectly_affected_count": len(impact.indirectly_affected),
                "tests_to_check": len(impact.test_entities_to_check),
                "risk_score": impact.risk_score,
                "estimated_effort_minutes": impact.estimated_effort
            }
    
    async def _metrics_reporting_loop(self):
        """Background loop for periodic metrics reporting."""
        while not self._stop_requested:
            await asyncio.sleep(self.config.metrics_report_interval)
            
            if self.metrics.enabled:
                summary = self.metrics.get_summary()
                logger.info(f"[EnhancedAgent] Metrics Summary:\n{self.metrics.generate_report()}")
                
                # Save to file
                report_path = Path(self.project_path) / ".pyutagent" / "metrics" / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                report_path.parent.mkdir(parents=True, exist_ok=True)
                self.metrics.save_report(report_path)
    
    def get_enhanced_stats(self) -> Dict[str, Any]:
        """Get enhanced statistics including all layers.
        
        Returns:
            Statistics dictionary
        """
        stats = {
            "metrics": self.metrics.get_summary() if self.metrics.enabled else None,
            "config": {
                "model_name": self.config.model_name,
                "enable_multi_agent": self.config.enable_multi_agent,
                "enable_prompt_optimization": self.config.enable_prompt_optimization,
                "enable_error_prediction": self.config.enable_error_prediction,
                "enable_strategy_optimization": self.config.enable_strategy_optimization,
                "enable_sandbox_execution": self.config.enable_sandbox_execution,
                "enable_user_interaction": self.config.enable_user_interaction,
                "enable_smart_analysis": self.config.enable_smart_analysis,
            }
        }
        
        # Add multi-agent stats if enabled
        if self.agent_coordinator:
            stats["multi_agent"] = self.agent_coordinator.get_stats()
        
        # Add shared knowledge stats if available
        if self.shared_knowledge:
            stats["shared_knowledge"] = self.shared_knowledge.get_stats()
        
        # Add experience replay stats if available
        if self.experience_replay:
            stats["experience_replay"] = self.experience_replay.get_stats()
        
        # Add P3 stats
        stats["p3_capabilities"] = {
            "error_prediction": {
                "enabled": self.error_predictor is not None,
                "predictions_made": self.error_predictor.total_predictions if self.error_predictor else 0,
                "accuracy": self.error_predictor.get_accuracy() if self.error_predictor else None
            } if self.error_predictor else {"enabled": False},
            "strategy_optimization": {
                "enabled": self.strategy_manager is not None,
                "strategies_tracked": len(self.strategy_manager.strategy_performances) if self.strategy_manager else 0
            } if self.strategy_manager else {"enabled": False},
            "sandbox_execution": {
                "enabled": self.sandbox_executor is not None,
                "security_level": self.config.sandbox_security_level.value
            } if self.sandbox_executor else {"enabled": False},
            "user_interaction": {
                "enabled": self.user_interaction is not None,
                "stats": self.user_interaction.get_interaction_stats() if self.user_interaction else None
            } if self.user_interaction else {"enabled": False},
            "smart_analysis": {
                "enabled": self.smart_analyzer is not None
            } if self.smart_analyzer else {"enabled": False}
        }
        
        return stats
    
    async def stop(self):
        """Stop the enhanced agent gracefully."""
        # Stop multi-agent system if running
        if self.agent_coordinator:
            await self.agent_coordinator.stop()
        
        # Save final metrics report
        if self.metrics.enabled:
            final_report_path = Path(self.project_path) / ".pyutagent" / "metrics" / "final_report.txt"
            final_report_path.parent.mkdir(parents=True, exist_ok=True)
            self.metrics.save_report(final_report_path)
        
        # Call parent stop
        super().stop()
        
        logger.info("[EnhancedAgent] Stopped")
