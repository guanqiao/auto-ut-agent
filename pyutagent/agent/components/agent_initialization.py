"""Agent Initialization - Component initialization and dependency injection."""

import logging
from pathlib import Path
from typing import Optional, Any, Dict, List, Tuple

from pyutagent.core.container import Container, get_container
from pyutagent.core.config import get_settings
from pyutagent.agent.prompts import PromptBuilder
from pyutagent.agent.actions import ActionRegistry
from pyutagent.core.error_recovery import ErrorRecoveryManager
from pyutagent.core.retry_manager import InfiniteRetryManager, RetryManagerConfig, RetryStrategy
from pyutagent.tools.java_parser import JavaCodeParser
from pyutagent.tools.maven_tools import MavenRunner, CoverageAnalyzer, ProjectScanner
from pyutagent.tools.aider_integration import AiderCodeFixer, AiderConfig
from pyutagent.tools.build_tool_manager import BuildToolManager, BuildToolRunner
from pyutagent.memory.working_memory import WorkingMemory
from pyutagent.llm.client import LLMClient

logger = logging.getLogger(__name__)


class AgentInitializer:
    """Handles initialization of all ReActAgent components.
    
    Responsible for:
    - Dependency injection from container
    - Component initialization (P0-P3 enhancements)
    - Lazy initialization of optional components
    - Build tool detection and setup
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        working_memory: WorkingMemory,
        project_path: str,
        container: Optional[Container] = None,
        model_name: Optional[str] = None,
        ab_test_id: Optional[str] = None
    ):
        """Initialize the agent initializer.
        
        Args:
            llm_client: LLM client
            working_memory: Working memory
            project_path: Project path
            container: Optional DI container
            model_name: Model name for optimization
            ab_test_id: Optional A/B test ID
        """
        self.llm_client = llm_client
        self.working_memory = working_memory
        self.project_path = project_path
        self.container = container or get_container()
        self.model_name = model_name or "gpt-4"
        self.ab_test_id = ab_test_id
        
        logger.info(f"[AgentInitializer] Initializing - Project: {project_path}, Model: {self.model_name}")
    
    def initialize_all_components(self, agent_instance: Any) -> Dict[str, Any]:
        """Initialize all components and return them as a dictionary.
        
        Args:
            agent_instance: The agent instance to attach components to
            
        Returns:
            Dictionary of all initialized components
        """
        components = {}
        
        components.update(self._init_basic_dependencies())
        components.update(self._init_build_tools())
        components.update(self._init_error_handling(agent_instance))
        components.update(self._init_p0_components())
        components.update(self._init_p1_components())
        components.update(self._init_p2_components())
        components.update(self._init_p3_components())
        components.update(self._init_phase4_components())
        components.update(self._init_lazy_components())
        components.update(self._init_tool_service())
        
        logger.info("[AgentInitializer] All components initialized")
        return components
    
    def _try_resolve(self, component_type):
        """Try to resolve a component from the container.
        
        Args:
            component_type: The type to resolve
            
        Returns:
            The resolved instance or None
        """
        try:
            return self.container.resolve(component_type)
        except KeyError:
            return None
        except Exception as e:
            logger.debug(f"[AgentInitializer] Failed to resolve {component_type}: {e}")
            return None
    
    def _init_basic_dependencies(self) -> Dict[str, Any]:
        """Initialize basic dependencies (PromptBuilder, ActionRegistry, etc.)."""
        components = {}
        
        prompt_builder = self._try_resolve(PromptBuilder)
        if not prompt_builder:
            prompt_builder = PromptBuilder()
            logger.debug("[AgentInitializer] Created default PromptBuilder")
        
        action_registry = self._try_resolve(ActionRegistry)
        if not action_registry:
            action_registry = ActionRegistry()
            logger.debug("[AgentInitializer] Created default ActionRegistry")
        
        java_parser = self._try_resolve(JavaCodeParser)
        if not java_parser:
            java_parser = JavaCodeParser()
            logger.debug("[AgentInitializer] Created default JavaCodeParser")
        
        components["prompt_builder"] = prompt_builder
        components["action_registry"] = action_registry
        components["java_parser"] = java_parser
        
        return components
    
    def _init_build_tools(self) -> Dict[str, Any]:
        """Initialize build tools (BuildToolManager, MavenRunner, etc.)."""
        components = {}
        
        build_tool_manager = BuildToolManager(self.project_path)
        build_tool_info = build_tool_manager.detect_build_tool()
        build_runner = build_tool_manager.get_runner()
        
        components["build_tool_manager"] = build_tool_manager
        components["build_tool_info"] = build_tool_info
        components["build_runner"] = build_runner
        
        maven_runner = self._try_resolve(MavenRunner)
        if not maven_runner:
            maven_runner = MavenRunner(self.project_path)
            logger.debug("[AgentInitializer] Created default MavenRunner")
        components["maven_runner"] = maven_runner
        
        coverage_analyzer = self._try_resolve(CoverageAnalyzer)
        if not coverage_analyzer:
            coverage_analyzer = CoverageAnalyzer(self.project_path)
            logger.debug("[AgentInitializer] Created default CoverageAnalyzer")
        components["coverage_analyzer"] = coverage_analyzer
        
        project_scanner = self._try_resolve(ProjectScanner)
        if not project_scanner:
            project_scanner = ProjectScanner(self.project_path)
            logger.debug("[AgentInitializer] Created default ProjectScanner")
        components["project_scanner"] = project_scanner
        
        if build_tool_info.tool_type.name != "UNKNOWN":
            logger.info(f"[AgentInitializer] Detected build tool: {build_tool_info.tool_type.name}")
        else:
            logger.warning("[AgentInitializer] No build tool detected, using Maven as fallback")
        
        return components
    
    def _init_error_handling(self, agent_instance: Any) -> Dict[str, Any]:
        """Initialize error handling components."""
        components = {}
        
        def on_recovery_progress(state: str, message: str):
            """Handle recovery progress updates."""
            logger.info(f"[AgentInitializer] Recovery progress - State: {state}, Message: {message}")
            if hasattr(agent_instance, '_update_state'):
                agent_instance._update_state("FIXING", f"[{state}] {message}")
        
        error_recovery = ErrorRecoveryManager(
            llm_client=self.llm_client,
            project_path=self.project_path,
            prompt_builder=self._try_resolve(PromptBuilder) or PromptBuilder(),
            progress_callback=on_recovery_progress
        )
        components["error_recovery"] = error_recovery
        
        retry_config = RetryManagerConfig(
            strategy=RetryStrategy.ADAPTIVE,
            base_delay=2.0,
            max_delay=30.0,
            exponential_base=1.5
        )
        retry_manager = InfiniteRetryManager(retry_config)
        components["retry_manager"] = retry_manager
        
        logger.debug(f"[AgentInitializer] Retry manager initialized - Strategy: {retry_config.strategy}")
        
        return components
    
    def _init_p0_components(self) -> Dict[str, Any]:
        """Initialize P0 enhancement components."""
        from pyutagent.agent.context_manager import ContextManager, CompressionStrategy
        from pyutagent.agent.generation_evaluator import GenerationEvaluator
        from pyutagent.agent.partial_success_handler import PartialSuccessHandler
        from pyutagent.agent.streaming import StreamingTestGenerator, StreamingConfig
        from pyutagent.tools.smart_editor import SmartCodeEditor
        
        components = {}
        
        context_manager = ContextManager(
            max_tokens=8000,
            target_tokens=6000,
            strategy=CompressionStrategy.HYBRID
        )
        components["context_manager"] = context_manager
        
        generation_evaluator = GenerationEvaluator()
        components["generation_evaluator"] = generation_evaluator
        
        partial_success_handler = PartialSuccessHandler()
        components["partial_success_handler"] = partial_success_handler
        
        streaming_generator = StreamingTestGenerator(
            self.llm_client,
            StreamingConfig(enable_preview=True)
        )
        components["streaming_generator"] = streaming_generator
        
        smart_editor = SmartCodeEditor(fuzzy_threshold=0.8)
        components["smart_editor"] = smart_editor
        
        logger.debug("[AgentInitializer] P0 components initialized")
        return components
    
    def _init_p1_components(self) -> Dict[str, Any]:
        """Initialize P1 enhancement components."""
        from pyutagent.agent.prompt_optimizer import PromptOptimizer, ModelType, ModelCharacteristics
        from pyutagent.core.error_learner import create_error_learner
        from pyutagent.agent.tool_orchestrator import create_tool_orchestrator
        from pyutagent.memory.context_compressor import create_context_compressor
        from pyutagent.tools.project_analyzer import ProjectAnalyzer
        
        components = {}
        
        prompt_optimizer = PromptOptimizer()
        components["prompt_optimizer"] = prompt_optimizer
        
        model_type = ModelType.UNKNOWN
        try:
            model_characteristics = ModelCharacteristics()
            model_type = model_characteristics.detect_model_type(self.model_name)
            logger.info(f"[AgentInitializer] Detected model type: {model_type.value}")
        except Exception as e:
            logger.warning(f"[AgentInitializer] Failed to detect model type: {e}")
        components["model_type"] = model_type
        
        error_learner = create_error_learner()
        components["error_learner"] = error_learner
        
        tool_orchestrator = create_tool_orchestrator()
        components["tool_orchestrator"] = tool_orchestrator
        
        context_compressor = create_context_compressor(max_tokens=8000)
        components["context_compressor"] = context_compressor
        
        project_analyzer = ProjectAnalyzer(self.project_path)
        components["project_analyzer"] = project_analyzer
        
        logger.debug("[AgentInitializer] P1 components initialized")
        return components
    
    def _init_p2_components(self) -> Dict[str, Any]:
        """Initialize P2 enhancement components."""
        from pyutagent.core.parallel_recovery import create_parallel_recovery_manager
        from pyutagent.core.sandbox import create_sandbox
        from pyutagent.core.tool_cache import create_tool_cache
        from pyutagent.core.checkpoint import create_checkpoint_manager
        
        components = {}
        
        parallel_recovery = create_parallel_recovery_manager(max_parallel=3)
        components["parallel_recovery"] = parallel_recovery
        
        sandbox = create_sandbox(allow_network=True, timeout=60.0)
        components["sandbox"] = sandbox
        
        tool_cache = create_tool_cache(maxsize=100, ttl=300)
        components["tool_cache"] = tool_cache
        
        checkpoint_manager = create_checkpoint_manager()
        components["checkpoint_manager"] = checkpoint_manager
        
        logger.debug("[AgentInitializer] P2 components initialized")
        return components
    
    def _init_p3_components(self) -> Dict[str, Any]:
        """Initialize P3 enhancement components."""
        from pyutagent.core.error_predictor import create_error_predictor
        from pyutagent.core.strategy_optimizer import create_strategy_optimizer
        from pyutagent.agent.user_interaction import create_user_interaction_handler, InteractiveFixer
        from pyutagent.agent.tool_validator import create_tool_validator
        
        components = {}
        
        error_predictor = create_error_predictor()
        components["error_predictor"] = error_predictor
        
        strategy_optimizer = create_strategy_optimizer()
        components["strategy_optimizer"] = strategy_optimizer
        
        user_interaction = create_user_interaction_handler(timeout=300)
        components["user_interaction"] = user_interaction
        
        interactive_fixer = InteractiveFixer(user_interaction)
        components["interactive_fixer"] = interactive_fixer
        
        tool_validator = create_tool_validator()
        components["tool_validator"] = tool_validator
        
        logger.debug("[AgentInitializer] P3 components initialized")
        return components
    
    def _init_phase4_components(self) -> Dict[str, Any]:
        """Initialize Phase 4 competitive features."""
        from pyutagent.core.code_interpreter import TestCodeInterpreter, InterpreterConfig
        from pyutagent.core.refactoring_engine import RefactoringEngine
        from pyutagent.core.test_quality_analyzer import TestQualityAnalyzer
        
        components = {}
        
        code_interpreter = TestCodeInterpreter(
            config=InterpreterConfig(timeout_seconds=60.0, max_output_size=10000)
        )
        components["code_interpreter"] = code_interpreter
        
        refactoring_engine = RefactoringEngine()
        components["refactoring_engine"] = refactoring_engine
        
        test_quality_analyzer = TestQualityAnalyzer()
        components["test_quality_analyzer"] = test_quality_analyzer
        
        logger.debug("[AgentInitializer] Phase 4 components initialized")
        return components
    
    def _init_lazy_components(self) -> Dict[str, Any]:
        """Initialize lazy-loaded and cached components."""
        from pyutagent.core.metrics import MetricsCollector, get_metrics
        
        components = {}
        
        components["_static_analysis_manager"] = None
        components["_error_knowledge_base"] = None
        components["_adaptive_strategy_manager"] = None
        components["_vector_store"] = None
        
        metrics_collector = get_metrics()
        components["metrics_collector"] = metrics_collector
        
        components["_build_tool_info_cache"] = None
        components["_embedding_cache"] = {}
        
        logger.debug("[AgentInitializer] Lazy components initialized")
        return components
    
    def _init_tool_service(self) -> Dict[str, Any]:
        """Initialize AgentToolService for tool management."""
        from pyutagent.agent.tool_service import AgentToolService
        
        components = {}
        
        tool_service = AgentToolService(
            project_path=self.project_path,
            base_path=self.project_path
        )
        
        components["tool_service"] = tool_service
        components["_tool_schemas"] = tool_service.get_schemas_for_llm()
        
        logger.info("[AgentInitializer] AgentToolService initialized")
        logger.debug(f"[AgentInitializer] Registered {len(tool_service.list_available_tools())} tools")
        
        return components
    
    def _init_aider_fixer(self) -> Optional[Any]:
        """Initialize AiderCodeFixer for enhanced error fixing.
        
        Returns:
            AiderCodeFixer instance or None
        """
        try:
            aider_fixer = self._try_resolve(AiderCodeFixer)
            if not aider_fixer:
                aider_config = self._try_resolve(AiderConfig)
                if not aider_config:
                    aider_config = AiderConfig()
                aider_fixer = AiderCodeFixer(
                    llm_client=self.llm_client,
                    config=aider_config
                )
                logger.debug("[AgentInitializer] Created default AiderCodeFixer")
            else:
                logger.debug("[AgentInitializer] Resolved AiderCodeFixer from container")
            return aider_fixer
        except (ImportError, ModuleNotFoundError, ValueError, Exception) as e:
            logger.warning(f"[AgentInitializer] Failed to create AiderCodeFixer: {e}")
            return None
