"""Enhanced ReAct Agent with full P0/P1/P2/P3 integration.

This module provides an enhanced agent that deeply integrates all enhancement layers:
- P0: Context management, quality evaluation, partial success handling
- P1: Prompt optimization, error learning, tool orchestration
- P2: Multi-agent collaboration
- P3: Error prediction, strategy optimization
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
from ..core.metrics import MetricsCollector, get_metrics
from ..core.protocols import AgentState, AgentResult
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
        """Try to recover with metrics tracking.
        
        Args:
            error: Error to recover from
            context: Error context
            
        Returns:
            Recovery result
        """
        with self.metrics.time_operation("error_recovery", context):
            result = await super()._try_recover(error, context)
            
            # Record recovery metrics
            success = result.get("should_continue", False)
            self.metrics.record_error(
                category="recovery",
                step=context.get("step", "unknown"),
                recovered=success
            )
            
            return result
    
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
