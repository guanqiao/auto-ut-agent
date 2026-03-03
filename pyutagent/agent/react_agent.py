"""ReAct Agent for UT generation with self-feedback loop and infinite retry."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
import asyncio

from .base_agent import BaseAgent, StepResult
from ..core.protocols import AgentState, AgentResult
from ..utils.code_extractor import CodeExtractor

logger = logging.getLogger(__name__)
from .prompts import PromptBuilder
from .actions import ActionRegistry
from ..core.error_recovery import ErrorRecoveryManager, ErrorCategory, RecoveryStrategy
from ..core.retry_manager import InfiniteRetryManager, RetryConfig, RetryStrategy
from ..core.container import Container, get_container
from ..tools.java_parser import JavaCodeParser
from ..tools.maven_tools import MavenRunner, CoverageAnalyzer, ProjectScanner
from ..tools.aider_integration import AiderCodeFixer, AiderConfig
from ..memory.working_memory import WorkingMemory
from ..llm.client import LLMClient
from ..core.config import get_settings

# Import existing P0 components
from .context_manager import ContextManager, CompressionStrategy
from .generation_evaluator import GenerationEvaluator, EvaluationResult
from .partial_success_handler import PartialSuccessHandler, PartialTestResult

# Import new enhancement modules - P0
from .streaming import StreamingCodeGenerator, StreamingTestGenerator, StreamingConfig
from ..tools.smart_editor import SmartCodeEditor, EditType, EditResult

# Import P1 PromptOptimizer
from .prompt_optimizer import PromptOptimizer, ModelType, ModelCharacteristics, optimize_prompt

# Import new enhancement modules - P1
from ..core.error_learner import ErrorPatternLearner, create_error_learner
from .tool_orchestrator import ToolOrchestrator, create_tool_orchestrator
from ..memory.context_compressor import ContextCompressor, create_context_compressor
from ..tools.project_analyzer import ProjectAnalyzer, MultiFileCoordinator

# Import new enhancement modules - P2
from ..core.parallel_recovery import ParallelRecoveryManager, create_parallel_recovery_manager
from ..core.sandbox import SandboxedToolExecutor, create_sandbox
from ..core.tool_cache import ToolResultCache, create_tool_cache
from ..core.checkpoint import CheckpointManager, create_checkpoint_manager

# Import new enhancement modules - P3
from ..core.error_predictor import ErrorPredictor, create_error_predictor
from ..core.strategy_optimizer import StrategyOptimizer, create_strategy_optimizer
from .user_interaction import UserInteractionHandler, InteractiveFixer, create_user_interaction_handler
from .tool_validator import ToolValidator, create_tool_validator

# Import Phase 4 competitive features
from ..core.code_interpreter import TestCodeInterpreter, InterpreterConfig
from ..core.refactoring_engine import RefactoringEngine, RefactoringType
from ..core.test_quality_analyzer import TestQualityAnalyzer, QualityDimension

# Import additional enhancement modules (previously unintegrated)
from ..tools.build_tool_manager import (
    BuildToolManager, BuildToolRunner, BuildToolType, BuildToolInfo,
    TestResult as BuildTestResult, CoverageResult as BuildCoverageResult
)
from ..tools.static_analysis_manager import StaticAnalysisManager, AnalysisToolType
from ..core.error_knowledge_base import ErrorKnowledgeBase, ErrorContext
from ..core.adaptive_strategy import AdaptiveStrategyManager, StrategyPerformance
from ..core.metrics import MetricsCollector, get_metrics
from ..memory.vector_store import SQLiteVecStore


class ReActAgent(BaseAgent):
    """ReAct agent for iterative UT generation with feedback loop.
    
    Key features:
    - Infinite retry until success or user stops
    - AI-powered error recovery for all error types
    - Local + LLM double-layer error analysis
    - Automatic strategy adjustment based on failure history
    - Dependency injection for better testability
    - Context window management for large files
    - Pre-compilation code quality evaluation
    - Partial success handling for incremental repair
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        working_memory: WorkingMemory,
        project_path: str,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        container: Optional[Container] = None,
        model_name: Optional[str] = None,
        ab_test_id: Optional[str] = None
    ):
        """Initialize ReAct agent.
        
        Args:
            llm_client: LLM client for generation
            working_memory: Working memory for context
            project_path: Path to the project
            progress_callback: Optional callback for progress updates
            container: Optional dependency injection container
            model_name: LLM model name for prompt optimization
            ab_test_id: Optional A/B test ID for prompt variant testing
        """
        super().__init__(llm_client, working_memory, project_path, progress_callback)
        
        self._container = container or get_container()
        self.model_name = model_name or "gpt-4"
        self.ab_test_id = ab_test_id
        
        logger.info(f"[ReActAgent] Initializing agent - Project: {project_path}, Model: {self.model_name}")
        
        self._init_dependencies(project_path)
        
        logger.info("[ReActAgent] Initialization complete")
    
    def _init_dependencies(self, project_path: str):
        """Initialize dependencies from container or create defaults.
        
        Args:
            project_path: Path to the project
        """
        self.prompt_builder = self._try_resolve(PromptBuilder)
        if not self.prompt_builder:
            self.prompt_builder = PromptBuilder()
            logger.debug("[ReActAgent] Created default PromptBuilder")
        
        self.action_registry = self._try_resolve(ActionRegistry)
        if not self.action_registry:
            self.action_registry = ActionRegistry()
            logger.debug("[ReActAgent] Created default ActionRegistry")
        
        self.java_parser = self._try_resolve(JavaCodeParser)
        if not self.java_parser:
            self.java_parser = JavaCodeParser()
            logger.debug("[ReActAgent] Created default JavaCodeParser")
        
        # Initialize BuildToolManager for multi-build-system support
        self.build_tool_manager = BuildToolManager(project_path)
        self.build_tool_info = self.build_tool_manager.detect_build_tool()
        self.build_runner = self.build_tool_manager.get_runner()
        
        # Legacy MavenRunner for backward compatibility
        self.maven_runner = self._try_resolve(MavenRunner)
        if not self.maven_runner:
            self.maven_runner = MavenRunner(project_path)
            logger.debug("[ReActAgent] Created default MavenRunner")
        
        # Log detected build tool
        if self.build_tool_info.tool_type != BuildToolType.UNKNOWN:
            logger.info(f"[ReActAgent] Detected build tool: {self.build_tool_info.tool_type.name}")
        else:
            logger.warning("[ReActAgent] No build tool detected, using Maven as fallback")
        
        self.coverage_analyzer = self._try_resolve(CoverageAnalyzer)
        if not self.coverage_analyzer:
            self.coverage_analyzer = CoverageAnalyzer(project_path)
            logger.debug("[ReActAgent] Created default CoverageAnalyzer")
        
        self.project_scanner = self._try_resolve(ProjectScanner)
        if not self.project_scanner:
            self.project_scanner = ProjectScanner(project_path)
            logger.debug("[ReActAgent] Created default ProjectScanner")
        
        logger.debug("[ReActAgent] Dependencies initialized from container")
        
        self.error_recovery = ErrorRecoveryManager(
            llm_client=self.llm_client,
            project_path=self.project_path,
            prompt_builder=self.prompt_builder,
            progress_callback=self._on_recovery_progress
        )
        
        self._init_aider_fixer()
        
        # Initialize new P0 components
        self._init_p0_components()
        
        retry_config = RetryConfig(
            strategy=RetryStrategy.ADAPTIVE,
            base_delay=2.0,
            max_delay=30.0,
            exponential_base=1.5
        )
        self.retry_manager = InfiniteRetryManager(retry_config)
        
        logger.debug(f"[ReActAgent] Retry manager initialized - Strategy: {retry_config.strategy}, BaseDelay: {retry_config.base_delay}s")
        
        self.current_test_file: Optional[str] = None
        self.target_class_info: Optional[Dict[str, Any]] = None
        self._stop_requested = False
    
    def _init_p0_components(self):
        """Initialize P0 enhancement components."""
        # Context Manager for handling large files
        self.context_manager = ContextManager(
            max_tokens=8000,
            target_tokens=6000,
            strategy=CompressionStrategy.HYBRID
        )
        logger.debug("[ReActAgent] ContextManager initialized")
        
        # Generation Evaluator for pre-compilation validation
        self.generation_evaluator = GenerationEvaluator()
        logger.debug("[ReActAgent] GenerationEvaluator initialized")
        
        # Partial Success Handler for incremental repair
        self.partial_success_handler = PartialSuccessHandler()
        logger.debug("[ReActAgent] PartialSuccessHandler initialized")
        
        # Initialize new P0-P3 enhancement components
        self._init_enhancement_components()
    
    def _init_p1_components(self):
        """Initialize P1 enhancement components."""
        # Prompt Optimizer for model-specific prompt optimization
        self.prompt_optimizer = PromptOptimizer()
        logger.debug("[ReActAgent] PromptOptimizer initialized")
        
        # Record model type for optimization
        self.model_type = ModelType.UNKNOWN
        try:
            model_characteristics = ModelCharacteristics()
            self.model_type = model_characteristics.detect_model_type(self.model_name)
            logger.info(f"[ReActAgent] Detected model type: {self.model_type.value}")
        except Exception as e:
            logger.warning(f"[ReActAgent] Failed to detect model type: {e}")
        
        # Load A/B test if configured
        self.current_ab_variant_id = None
        if self.ab_test_id:
            self.current_ab_variant_id, _ = self.prompt_optimizer.get_prompt_for_test(
                self.ab_test_id,
                class_name="placeholder"
            )
            if self.current_ab_variant_id:
                logger.info(f"[ReActAgent] Loaded A/B test variant: {self.current_ab_variant_id}")
            else:
                logger.warning(f"[ReActAgent] Failed to load A/B test: {self.ab_test_id}")
    
    def _init_enhancement_components(self):
        """Initialize all new enhancement components."""
        # P0: Streaming and Smart Editor
        self.streaming_generator = StreamingTestGenerator(
            self.llm_client,
            StreamingConfig(enable_preview=True)
        )
        logger.debug("[ReActAgent] StreamingTestGenerator initialized")
        
        self.smart_editor = SmartCodeEditor(fuzzy_threshold=0.8)
        logger.debug("[ReActAgent] SmartCodeEditor initialized")
        
        # P1: Prompt Optimizer, Error learning, tool orchestration, context compression, project analysis
        self._init_p1_components()
        
        self.error_learner = create_error_learner()
        logger.debug("[ReActAgent] ErrorPatternLearner initialized")
        
        self.tool_orchestrator = create_tool_orchestrator()
        logger.debug("[ReActAgent] ToolOrchestrator initialized")
        
        self.context_compressor = create_context_compressor(max_tokens=8000)
        logger.debug("[ReActAgent] ContextCompressor initialized")
        
        self.project_analyzer = ProjectAnalyzer(self.project_path)
        logger.debug("[ReActAgent] ProjectAnalyzer initialized")
        
        # P2: Parallel recovery, sandbox, cache, checkpoint
        self.parallel_recovery = create_parallel_recovery_manager(max_parallel=3)
        logger.debug("[ReActAgent] ParallelRecoveryManager initialized")
        
        self.sandbox = create_sandbox(allow_network=True, timeout=60.0)
        logger.debug("[ReActAgent] SandboxedToolExecutor initialized")
        
        self.tool_cache = create_tool_cache(maxsize=100, ttl=300)
        logger.debug("[ReActAgent] ToolResultCache initialized")
        
        self.checkpoint_manager = create_checkpoint_manager()
        logger.debug("[ReActAgent] CheckpointManager initialized")
        
        # P3: Error prediction, strategy optimization, user interaction, tool validation
        self.error_predictor = create_error_predictor()
        logger.debug("[ReActAgent] ErrorPredictor initialized")
        
        self.strategy_optimizer = create_strategy_optimizer()
        logger.debug("[ReActAgent] StrategyOptimizer initialized")
        
        self.user_interaction = create_user_interaction_handler(timeout=300)
        logger.debug("[ReActAgent] UserInteractionHandler initialized")
        
        self.interactive_fixer = InteractiveFixer(self.user_interaction)
        logger.debug("[ReActAgent] InteractiveFixer initialized")
        
        self.tool_validator = create_tool_validator()
        logger.debug("[ReActAgent] ToolValidator initialized")
        
        # Phase 4: Competitive features - Code interpreter, refactoring, quality analysis
        self.code_interpreter = TestCodeInterpreter(
            config=InterpreterConfig(timeout_seconds=60.0, max_output_size=10000)
        )
        logger.debug("[ReActAgent] TestCodeInterpreter initialized")
        
        self.refactoring_engine = RefactoringEngine()
        logger.debug("[ReActAgent] RefactoringEngine initialized")
        
        self.test_quality_analyzer = TestQualityAnalyzer()
        logger.debug("[ReActAgent] TestQualityAnalyzer initialized")
        
        # Lazy-initialized modules (initialized on first use)
        self._static_analysis_manager: Optional[StaticAnalysisManager] = None
        self._error_knowledge_base: Optional[ErrorKnowledgeBase] = None
        self._adaptive_strategy_manager: Optional[AdaptiveStrategyManager] = None
        self._vector_store: Optional[SQLiteVecStore] = None
        
        # Metrics collection (lightweight, initialize immediately)
        self.metrics_collector = get_metrics()
        logger.debug("[ReActAgent] MetricsCollector initialized")
        
        # Cache for build tool info
        self._build_tool_info_cache: Optional[Dict[str, Any]] = None
        
        # Cache for embeddings
        self._embedding_cache: Dict[str, List[float]] = {}
    
    @property
    def static_analysis_manager(self) -> StaticAnalysisManager:
        """Lazy-initialized static analysis manager with graceful degradation."""
        if self._static_analysis_manager is None:
            try:
                self._static_analysis_manager = StaticAnalysisManager(self.project_path)
                logger.debug("[ReActAgent] StaticAnalysisManager lazy-initialized")
            except Exception as e:
                logger.warning(f"[ReActAgent] StaticAnalysisManager initialization failed: {e}, using no-op fallback")
                self._static_analysis_manager = self._create_noop_static_analysis_manager()
        return self._static_analysis_manager
    
    def _create_noop_static_analysis_manager(self) -> Any:
        """Create a no-op static analysis manager as fallback."""
        class NoOpStaticAnalysisManager:
            async def run_all_analysis(self, target_files=None, include_tests=False):
                from dataclasses import dataclass, field
                from typing import List
                from enum import Enum
                
                class AnalysisToolType(Enum):
                    SPOTBUGS = "spotbugs"
                    PMD = "pmd"
                    CHECKSTYLE = "checkstyle"
                
                @dataclass
                class BugInfo:
                    bug_type: str = ""
                    severity: str = "LOW"
                    message: str = ""
                    class_name: str = ""
                    method_name: str = ""
                    line_number: int = 0
                    suggestion: str = ""
                
                @dataclass
                class AnalysisResult:
                    tool_type: AnalysisToolType = AnalysisToolType.SPOTBUGS
                    success: bool = False
                    bug_count: int = 0
                    bugs: List[BugInfo] = field(default_factory=list)
                
                return {AnalysisToolType.SPOTBUGS: AnalysisResult()}
        return NoOpStaticAnalysisManager()
    
    @property
    def error_knowledge_base(self) -> ErrorKnowledgeBase:
        """Lazy-initialized error knowledge base with graceful degradation."""
        if self._error_knowledge_base is None:
            try:
                db_path = str(Path(self.project_path) / ".utagent" / "error_knowledge.db")
                self._error_knowledge_base = ErrorKnowledgeBase(db_path=db_path)
                logger.debug("[ReActAgent] ErrorKnowledgeBase lazy-initialized")
            except Exception as e:
                logger.warning(f"[ReActAgent] ErrorKnowledgeBase initialization failed: {e}, using in-memory fallback")
                self._error_knowledge_base = self._create_in_memory_error_knowledge_base()
        return self._error_knowledge_base
    
    def _create_in_memory_error_knowledge_base(self) -> Any:
        """Create an in-memory error knowledge base as fallback."""
        from dataclasses import dataclass, field
        from typing import List, Optional
        from enum import Enum
        
        class ErrorCategory(Enum):
            COMPILATION = "compilation"
            RUNTIME = "runtime"
            ASSERTION = "assertion"
            TIMEOUT = "timeout"
            UNKNOWN = "unknown"
        
        @dataclass
        class ErrorContext:
            error_message: str
            category: ErrorCategory = ErrorCategory.UNKNOWN
            stack_trace: Optional[str] = None
            test_class: Optional[str] = None
            test_method: Optional[str] = None
        
        @dataclass
        class SearchResult:
            solution: Any
            similarity_score: float = 0.0
        
        class InMemoryErrorKnowledgeBase:
            def __init__(self):
                self._solutions: List[Any] = []
            
            def find_similar_errors(self, error_context: ErrorContext, min_similarity: float = 0.6, max_results: int = 5) -> List[SearchResult]:
                return []
            
            def record_solution(self, error_context: ErrorContext, fix_description: str, fix_code: Optional[str] = None, metadata: Optional[dict] = None) -> str:
                return "in-memory-solution-id"
            
            def record_outcome(self, error_context: ErrorContext, solution_id: str, success: bool, metadata: Optional[dict] = None) -> None:
                pass
        
        return InMemoryErrorKnowledgeBase()
    
    @property
    def adaptive_strategy_manager(self) -> AdaptiveStrategyManager:
        """Lazy-initialized adaptive strategy manager with graceful degradation."""
        if self._adaptive_strategy_manager is None:
            try:
                db_path = str(Path(self.project_path) / ".utagent" / "adaptive_strategy.db")
                self._adaptive_strategy_manager = AdaptiveStrategyManager(db_path=db_path)
                logger.debug("[ReActAgent] AdaptiveStrategyManager lazy-initialized")
            except Exception as e:
                logger.warning(f"[ReActAgent] AdaptiveStrategyManager initialization failed: {e}, using default fallback")
                self._adaptive_strategy_manager = self._create_default_adaptive_strategy_manager()
        return self._adaptive_strategy_manager
    
    def _create_default_adaptive_strategy_manager(self) -> Any:
        """Create a default adaptive strategy manager as fallback."""
        class DefaultAdaptiveStrategyManager:
            def select_strategy(self, error_category: str, available_strategies: List, context: dict, allow_exploration: bool = True):
                if available_strategies:
                    return available_strategies[0]
                from ..core.parallel_recovery import RecoveryStrategy
                return RecoveryStrategy.DEFAULT
            
            def record_attempt(self, strategy, error_category: str, success: bool, execution_time_ms: float, context: dict):
                pass
        
        return DefaultAdaptiveStrategyManager()
    
    @property
    def vector_store(self) -> Optional[SQLiteVecStore]:
        """Lazy-initialized vector store (optional, may return None if unavailable)."""
        if self._vector_store is None:
            try:
                db_path = str(Path(self.project_path) / ".utagent" / "vectors.db")
                self._vector_store = SQLiteVecStore(db_path=db_path, dimension=384)
                logger.debug("[ReActAgent] VectorStore lazy-initialized")
            except Exception as e:
                logger.warning(f"[ReActAgent] VectorStore initialization failed: {e}")
                self._vector_store = None
        return self._vector_store
    
    def _init_aider_fixer(self):
        """Initialize AiderCodeFixer for enhanced error fixing."""
        self.aider_fixer = self._try_resolve(AiderCodeFixer)
        if not self.aider_fixer:
            try:
                aider_config = self._try_resolve(AiderConfig)
                if not aider_config:
                    aider_config = AiderConfig()
                self.aider_fixer = AiderCodeFixer(
                    llm_client=self.llm_client,
                    config=aider_config
                )
                logger.debug("[ReActAgent] Created default AiderCodeFixer")
            except (ImportError, ModuleNotFoundError) as e:
                logger.warning(f"[ReActAgent] Aider dependencies not available: {e}")
                self.aider_fixer = None
            except ValueError as e:
                logger.warning(f"[ReActAgent] Invalid Aider configuration: {e}")
                self.aider_fixer = None
            except Exception as e:
                logger.warning(f"[ReActAgent] Failed to create AiderCodeFixer: {e}")
                self.aider_fixer = None
        else:
            logger.debug("[ReActAgent] Resolved AiderCodeFixer from container")
    
    def _try_resolve(self, component_type):
        """Try to resolve a component from the container.

        Args:
            component_type: The type to resolve

        Returns:
            The resolved instance or None
        """
        try:
            return self._container.resolve(component_type)
        except KeyError:
            return None
        except Exception as e:
            logger.debug(f"[ReActAgent] Failed to resolve {component_type}: {e}")
            return None
    
    def _on_recovery_progress(self, state: str, message: str):
        """Handle recovery progress updates."""
        logger.info(f"[ReActAgent] Recovery progress - State: {state}, Message: {message}")
        self._update_state(AgentState.FIXING, f"[{state}] {message}")
    
    def stop(self):
        """Stop agent execution gracefully (legacy, use pause instead)."""
        logger.info("[ReActAgent] Stopping agent execution")
        super().request_stop()
        self.retry_manager.stop()
    
    def pause(self):
        """Pause agent execution.
        
        The agent will pause at the next checkpoint.
        """
        logger.info("[ReActAgent] Pausing agent execution")
        super().pause()
        self.retry_manager.stop()
    
    def resume(self):
        """Resume agent execution."""
        logger.info("[ReActAgent] Resuming agent execution")
        super().resume()
        self.retry_manager.reset()
    
    def terminate(self):
        """Terminate agent execution immediately."""
        logger.info("[ReActAgent] Terminating agent execution")
        super().terminate()
        self.retry_manager.stop()
        self.error_recovery.clear_history()
    
    def reset(self):
        """Reset agent state."""
        logger.info("[ReActAgent] Resetting agent state")
        super().reset()
        self.retry_manager.reset()
        self.error_recovery.clear_history()
        self.checkpoint_manager.clear()
    
    async def generate_tests(self, target_file: str) -> AgentResult:
        """Generate tests for a target file with feedback loop."""
        logger.info(f"[ReActAgent] Starting test generation - Target: {target_file}")
        return await self.run_feedback_loop(target_file)
    
    async def resume_from_checkpoint(
        self,
        checkpoint_id: Optional[str] = None
    ) -> AgentResult:
        """Resume execution from a checkpoint.
        
        Args:
            checkpoint_id: ID of checkpoint to resume from, or None for latest
            
        Returns:
            AgentResult after resuming
        """
        logger.info(f"[ReActAgent] Resuming from checkpoint: {checkpoint_id or 'latest'}")
        
        checkpoint = self.checkpoint_manager.load_checkpoint(checkpoint_id)
        if not checkpoint:
            logger.warning("[ReActAgent] No checkpoint found to resume from")
            return AgentResult(
                success=False,
                message="No checkpoint found to resume from",
                state=AgentState.FAILED
            )
        
        state = checkpoint.state
        target_file = state.get("target_file")
        self.current_test_file = state.get("test_file")
        self.target_class_info = state.get("class_info")
        
        logger.info(f"[ReActAgent] Restored state from checkpoint - "
                    f"Target: {target_file}, TestFile: {self.current_test_file}")
        
        return await self.run_feedback_loop(target_file)
    
    async def run_feedback_loop(self, target_file: str) -> AgentResult:
        """Run the complete feedback loop for UT generation with infinite retry.
        
        The loop follows this pattern:
        1. Parse target Java file (with retry)
        2. Generate initial tests (with retry)
        3. Compile tests -> if fails, AI analyzes & fixes -> retry
        4. Run tests -> if fails, AI analyzes & fixes -> retry
        5. Check coverage -> if < target, generate additional tests -> back to 3
        6. Repeat until success or user stops
        """
        self._stop_requested = False
        self._terminated = False
        self._pause_event.set()  # Ensure not paused at start
        
        logger.info(f"[ReActAgent] 🎯 Starting test generation for: {Path(target_file).name}")
        logger.info(f"[ReActAgent] 📊 Configuration - MaxIterations: {self.max_iterations}, TargetCoverage: {self.target_coverage:.1%}")
        
        # Check for pause/terminate before starting
        await self._check_pause()
        if self._terminated:
            return self._create_terminated_result("before starting")
        
        # Phase 1: Parse target file
        parse_result = await self._phase_parse_target(target_file)
        if not parse_result.success:
            return parse_result
        
        # Phase 2: Generate initial tests
        generate_result = await self._phase_generate_initial_tests()
        if not generate_result.success:
            return generate_result
        
        # Save checkpoint after initial generation
        self._save_initial_checkpoint(target_file)
        
        # Phase 3-6: Compile-Test-Analyze-Optimize loop
        return await self._phase_feedback_loop()
    
    async def _phase_parse_target(self, target_file: str) -> AgentResult:
        """Phase 1: Parse target Java file.
        
        Args:
            target_file: Path to the target Java file
            
        Returns:
            AgentResult with parsing results
        """
        self._update_state(AgentState.PARSING, "📖 Step 1/6: Parsing target Java file...")
        logger.info("[ReActAgent] 📖 Step 1: Parsing target file")
        
        parse_result = await self._execute_with_recovery(
            self._parse_target_file,
            target_file,
            step_name="parsing"
        )
        
        if not parse_result.success or self._stop_requested:
            logger.error(f"[ReActAgent] Failed to parse target file - {parse_result.message}")
            return AgentResult(
                success=False,
                message=f"Failed to parse target file after all recovery attempts: {parse_result.message}",
                errors=[parse_result.message]
            )
        
        self.target_class_info = parse_result.data.get("class_info")
        self.working_memory.current_file = target_file
        class_name = self.target_class_info.get('name', 'unknown')
        method_count = len(self.target_class_info.get('methods', []))
        logger.info(f"[ReActAgent] ✅ Parsing complete - Class: {class_name}, Methods: {method_count}")
        
        return AgentResult(success=True, state=AgentState.PARSING)
    
    async def _phase_generate_initial_tests(self) -> AgentResult:
        """Phase 2: Generate initial tests.
        
        Returns:
            AgentResult with generation results
        """
        class_name = self.target_class_info.get('name', 'unknown')
        method_count = len(self.target_class_info.get('methods', []))
        
        self._update_state(AgentState.GENERATING, f"✨ Step 2/6: Generating initial tests for {class_name}...")
        logger.info("[ReActAgent] ✨ Step 2: Generating initial tests")
        logger.info(f"[ReActAgent] 🤖 Calling LLM to generate tests for {class_name} with {method_count} methods...")
        
        generate_result = await self._execute_with_recovery(
            self._generate_initial_tests,
            step_name="generating initial tests"
        )
        
        if not generate_result.success or self._stop_requested:
            logger.error(f"[ReActAgent] Failed to generate initial tests - {generate_result.message}")
            return AgentResult(
                success=False,
                message=f"Failed to generate tests after all recovery attempts: {generate_result.message}",
                errors=[generate_result.message]
            )
        
        self.current_test_file = generate_result.data.get("test_file")
        logger.info(f"[ReActAgent] ✅ Initial test generation complete - TestFile: {self.current_test_file}")
        
        return AgentResult(success=True, state=AgentState.GENERATING)
    
    def _save_initial_checkpoint(self, target_file: str):
        """Save checkpoint after initial test generation.
        
        Args:
            target_file: Path to the target file
        """
        self.checkpoint_manager.save_checkpoint(
            step="initial_generation",
            iteration=0,
            state={
                "target_file": target_file,
                "test_file": self.current_test_file,
                "class_info": self.target_class_info
            }
        )
    
    async def _phase_feedback_loop(self) -> AgentResult:
        """Phase 3-6: Compile-Test-Analyze-Optimize loop.
        
        Returns:
            AgentResult with final results
        """
        loop_start_time = asyncio.get_event_loop().time()
        
        self._update_state(AgentState.COMPILING, "🔨 Step 3/6: Compiling generated tests...")
        logger.info("[ReActAgent] 🔨 Step 3: Starting compile-test loop")
        
        while not self._stop_requested and not self._terminated:
            # Check for pause at the beginning of each iteration
            if await self._check_should_stop("during pause"):
                break
            
            self.current_iteration += 1
            self.working_memory.increment_iteration()
            
            logger.info(f"[ReActAgent] 🔄 ===== Iteration {self.current_iteration}/{self.max_iterations} started =====")
            
            # Check termination conditions
            if self._should_stop_iteration():
                break
            
            # Execute one complete iteration
            iteration_result = await self._execute_iteration()
            if iteration_result:
                return iteration_result
        
        return self._create_final_result(loop_start_time)
    
    def _should_stop_iteration(self) -> bool:
        """Check if iteration should stop based on termination conditions.
        
        Returns:
            True if should stop, False otherwise
        """
        if self.current_iteration > self.max_iterations:
            logger.warning(f"[ReActAgent] Max iterations reached - Max: {self.max_iterations}, FinalCoverage: {self.working_memory.current_coverage:.1%}")
            self._update_state(
                AgentState.COMPLETED,
                f"Max iterations ({self.max_iterations}) reached. Final coverage: {self.working_memory.current_coverage:.1%}"
            )
            return True
        
        if self.working_memory.current_coverage >= self.target_coverage:
            logger.info(f"[ReActAgent] Target coverage reached - Current: {self.working_memory.current_coverage:.1%}, Target: {self.target_coverage:.1%}")
            self._update_state(
                AgentState.COMPLETED,
                f"Target coverage reached: {self.working_memory.current_coverage:.1%}"
            )
            return True
        
        return False
    
    async def _check_should_stop(self, context: str) -> bool:
        """Check if execution should stop (pause/terminate).
        
        Args:
            context: Context for logging
            
        Returns:
            True if should stop, False otherwise
        """
        await self._check_pause()
        if self._terminated:
            logger.info(f"[ReActAgent] ⏹️ Terminated {context}")
            return True
        return False
    
    async def _execute_iteration(self) -> Optional[AgentResult]:
        """Execute one complete iteration of the feedback loop.
        
        Returns:
            AgentResult if iteration completes or fails, None to continue
        """
        # Step 3: Compile
        if await self._check_should_stop("before compilation"):
            return None
        
        if not await self._iteration_compile():
            return None
        
        # Step 4: Test
        if await self._check_should_stop("before testing"):
            return None
        
        if not await self._iteration_test():
            return None
        
        # Step 5: Coverage analysis
        if await self._check_should_stop("before coverage analysis"):
            return None
        
        coverage_result = await self._iteration_coverage()
        if not coverage_result:
            return None
        
        current_coverage = coverage_result.data.get("line_coverage", 0.0)
        
        # Check if target coverage reached
        if current_coverage >= self.target_coverage:
            return self._create_success_result(current_coverage)
        
        # Step 6: Generate additional tests
        if await self._check_should_stop("before generating additional tests"):
            return None
        
        if not await self._iteration_generate_additional(current_coverage):
            return None
        
        logger.info(f"[ReActAgent] ✅ ===== Iteration {self.current_iteration} complete =====")
        return None
    
    async def _iteration_compile(self) -> bool:
        """Execute compilation step.
        
        Returns:
            True if successful, False to retry
        """
        logger.info(f"[ReActAgent] 🔨 Step 3: Compiling tests (Iteration {self.current_iteration})")
        compile_success = await self._compile_with_recovery()
        
        if not compile_success or self._stop_requested or self._terminated:
            if self._stop_requested or self._terminated:
                logger.info("[ReActAgent] ⏹️ User stopped - Compilation phase")
                return False
            logger.warning("[ReActAgent] ⚠️ Compilation failed, preparing to retry...")
            return False
        
        return True
    
    async def _iteration_test(self) -> bool:
        """Execute test running step.
        
        Returns:
            True if successful, False to retry
        """
        self._update_state(AgentState.TESTING, f"🧪 Step 4/6: Running tests (Iteration {self.current_iteration})...")
        logger.info(f"[ReActAgent] 🧪 Step 4: Running tests (Iteration {self.current_iteration})")
        
        test_success = await self._run_tests_with_recovery()
        
        if not test_success or self._stop_requested or self._terminated:
            if self._stop_requested or self._terminated:
                logger.info("[ReActAgent] ⏹️ User stopped - Test phase")
                return False
            logger.warning("[ReActAgent] ⚠️ Tests failed, preparing to retry...")
            return False
        
        return True
    
    async def _iteration_coverage(self) -> Optional[StepResult]:
        """Execute coverage analysis step.
        
        Returns:
            StepResult if successful, None to retry
        """
        self._update_state(AgentState.ANALYZING, f"📊 Step 5/6: Analyzing coverage (Iteration {self.current_iteration})...")
        logger.info(f"[ReActAgent] 📊 Step 5: Analyzing coverage (Iteration {self.current_iteration})")
        
        coverage_result = await self._execute_with_recovery(
            self._analyze_coverage,
            step_name="analyzing coverage"
        )
        
        if not coverage_result.success:
            logger.warning("[ReActAgent] Coverage analysis failed, preparing to retry")
            self._update_state(AgentState.FIXING, "Coverage analysis failed, retrying...")
            return None
        
        current_coverage = coverage_result.data.get("line_coverage", 0.0)
        self.working_memory.update_coverage(current_coverage)
        logger.info(f"[ReActAgent] 📈 Current coverage: {current_coverage:.1%} (Target: {self.target_coverage:.1%})")
        
        return coverage_result
    
    async def _iteration_generate_additional(self, current_coverage: float) -> bool:
        """Execute additional test generation step.
        
        Args:
            current_coverage: Current coverage percentage
            
        Returns:
            True if successful, False to retry
        """
        logger.info(f"[ReActAgent] 🚀 Step 6: Generating additional tests - Coverage: {current_coverage:.1%} < Target: {self.target_coverage:.1%}")
        self._update_state(
            AgentState.OPTIMIZING,
            f"🚀 Coverage {current_coverage:.1%} < target {self.target_coverage:.1%}, calling LLM to generate additional tests..."
        )
        
        additional_result = await self._execute_with_recovery(
            self._generate_additional_tests,
            {"line_coverage": current_coverage},
            step_name="generating additional tests"
        )
        
        if not additional_result.success:
            logger.warning("[ReActAgent] ⚠️ Additional test generation failed, preparing to retry...")
            self._update_state(AgentState.FIXING, "⚠️ Additional test generation failed, retrying...")
            return False
        
        return True
    
    def _create_success_result(self, coverage: float) -> AgentResult:
        """Create success result when target coverage is reached.
        
        Args:
            coverage: Final coverage percentage
            
        Returns:
            AgentResult with success status
        """
        logger.info(f"[ReActAgent] 🎉 Target coverage reached! {coverage:.1%}")
        self._update_state(
            AgentState.COMPLETED,
            f"🎉 Target coverage reached: {coverage:.1%}"
        )
        return AgentResult(
            success=True,
            message=f"Successfully generated tests with {coverage:.1%} coverage",
            test_file=self.current_test_file,
            coverage=coverage,
            iterations=self.current_iteration
        )
    
    def _create_terminated_result(self, context: str) -> AgentResult:
        """Create result when generation is terminated.
        
        Args:
            context: Context for the message
            
        Returns:
            AgentResult with terminated status
        """
        return AgentResult(
            success=False,
            message=f"Generation terminated by user {context}",
            state=AgentState.FAILED
        )
    
    def _create_final_result(self, loop_start_time: float) -> AgentResult:
        """Create final result after loop completion.
        
        Args:
            loop_start_time: Start time of the loop
            
        Returns:
            AgentResult with final status
        """
        final_coverage = self.working_memory.current_coverage
        elapsed = asyncio.get_event_loop().time() - loop_start_time
        
        if self._terminated:
            logger.info(f"[ReActAgent] ⏹️ User terminated - Iterations: {self.current_iteration}, Coverage: {final_coverage:.1%}, Time: {elapsed:.1f}s")
            return AgentResult(
                success=False,
                message="Generation terminated by user",
                test_file=self.current_test_file,
                coverage=final_coverage,
                iterations=self.current_iteration,
                state=AgentState.FAILED
            )
        
        if self._stop_requested:
            logger.info(f"[ReActAgent] ⏹️ User stopped - Iterations: {self.current_iteration}, Coverage: {final_coverage:.1%}, Time: {elapsed:.1f}s")
            return AgentResult(
                success=False,
                message="Generation stopped by user",
                test_file=self.current_test_file,
                coverage=final_coverage,
                iterations=self.current_iteration,
                state=AgentState.PAUSED
            )
        
        logger.info(f"[ReActAgent] ✨ Feedback loop completed - Iterations: {self.current_iteration}, Coverage: {final_coverage:.1%}, Time: {elapsed:.1f}s")
        return AgentResult(
            success=final_coverage > 0,
            message=f"Completed after {self.current_iteration} iterations with {final_coverage:.1%} coverage",
            test_file=self.current_test_file,
            coverage=final_coverage,
            iterations=self.current_iteration
        )
    
    async def _execute_with_recovery(
        self,
        operation,
        *args,
        step_name: str = "operation",
        **kwargs
    ) -> StepResult:
        """Execute an operation with automatic error recovery.
        
        Args:
            operation: The operation to execute
            *args: Positional arguments
            step_name: Name of the step for logging
            **kwargs: Keyword arguments
            
        Returns:
            StepResult
        """
        attempt = 0
        
        logger.info(f"[ReActAgent] Starting step execution - Step: {step_name}")
        
        while not self._stop_requested and not self._terminated:
            attempt += 1
            
            logger.debug(f"[ReActAgent] Step attempt - Step: {step_name}, Attempt: {attempt}")
            
            try:
                if asyncio.iscoroutinefunction(operation):
                    result = await operation(*args, **kwargs)
                else:
                    result = operation(*args, **kwargs)
                
                if result.success:
                    logger.info(f"[ReActAgent] Step executed successfully - Step: {step_name}, Attempt: {attempt}")
                    return result
                else:
                    logger.warning(f"[ReActAgent] Step returned failure - Step: {step_name}, Attempt: {attempt}, Message: {result.message}")
                    error = Exception(result.message)
                    recovery_result = await self._try_recover(
                        error,
                        {"step": step_name, "attempt": attempt, "result": result}
                    )
                    
                    if not recovery_result.get("should_continue", True):
                        logger.error(f"[ReActAgent] Recovery failed, step terminated - Step: {step_name}")
                        return StepResult(
                            success=False,
                            state=AgentState.FAILED,
                            message=f"Recovery failed for {step_name}"
                        )
                    
                    action = recovery_result.get("action", "retry")
                    logger.info(f"[ReActAgent] Applying recovery action - Action: {action}")
                    
                    if action == "fix":
                        fixed_code = recovery_result.get("fixed_code")
                        if fixed_code:
                            await self._write_test_file(fixed_code)
                    elif action == "reset":
                        logger.info("[ReActAgent] Resetting and regenerating")
                        return await self._execute_with_recovery(
                            self._generate_initial_tests,
                            step_name="regenerating tests"
                        )
                    
                    continue
                    
            except Exception as e:
                logger.exception(f"[ReActAgent] Step execution exception - Step: {step_name}, Attempt: {attempt}, Error: {e}")
                
                recovery_result = await self._try_recover(
                    e,
                    {"step": step_name, "attempt": attempt}
                )
                
                if not recovery_result.get("should_continue", True):
                    logger.error(f"[ReActAgent] Recovery failed, step terminated - Step: {step_name}")
                    return StepResult(
                        success=False,
                        state=AgentState.FAILED,
                        message=f"Recovery failed for {step_name}: {str(e)}"
                    )
                
                action = recovery_result.get("action", "retry")
                if action == "fix":
                    fixed_code = recovery_result.get("fixed_code")
                    if fixed_code:
                        await self._write_test_file(fixed_code)
                elif action == "skip":
                    logger.info(f"[ReActAgent] Skipping step - Step: {step_name}")
                    return StepResult(
                        success=True,
                        state=AgentState.COMPLETED,
                        message=f"Skipped {step_name}",
                        data={}
                    )
                
                continue
        
        if self._terminated:
            logger.info(f"[ReActAgent] Step terminated - Step: {step_name}")
            return StepResult(
                success=False,
                state=AgentState.FAILED,
                message="Operation terminated by user"
            )
        
        logger.info(f"[ReActAgent] User stopped step - Step: {step_name}")
        return StepResult(
            success=False,
            state=AgentState.PAUSED,
            message="Operation stopped by user"
        )
    
    async def _try_recover(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Try to recover from an error with learning and optimization.
        
        Args:
            error: The error that occurred
            context: Error context
            
        Returns:
            Recovery result
        """
        import time
        start_time = time.time()
        
        logger.info(f"[ReActAgent] Attempting recovery - Error: {error}, Context: {context}")
        
        error_category = self._categorize_error(error, context)
        
        suggested_strategy = self.error_learner.suggest_strategy(error, error_category, context)
        if suggested_strategy:
            strategy, confidence = suggested_strategy
            logger.info(f"[ReActAgent] Error learner suggests {strategy.name} with confidence {confidence:.2f}")
            
            optimization = self.strategy_optimizer.optimize_strategy_selection(error_category, context)
            logger.info(f"[ReActAgent] Strategy optimizer recommends {optimization.recommended_strategy.name}")
        
        current_test_code = None
        if self.current_test_file:
            try:
                test_file_path = Path(self.project_path) / self.current_test_file
                if test_file_path.exists():
                    current_test_code = test_file_path.read_text(encoding='utf-8')
                    logger.debug(f"[ReActAgent] Read current test code - Length: {len(current_test_code)}")
            except Exception as e:
                logger.warning(f"[ReActAgent] Failed to read test code: {e}")
        
        recovery_result = await self.error_recovery.recover(
            error,
            error_context=context,
            current_test_code=current_test_code,
            target_class_info=self.target_class_info
        )
        
        elapsed_time = time.time() - start_time
        success = recovery_result.get("action") not in ("abort", "fail")
        strategy_used = RecoveryStrategy.ANALYZE_AND_FIX
        if recovery_result.get("action") == "fix":
            strategy_used = RecoveryStrategy.ANALYZE_AND_FIX
        elif recovery_result.get("action") == "reset":
            strategy_used = RecoveryStrategy.RESET_AND_REGENERATE
        elif recovery_result.get("action") == "retry":
            strategy_used = RecoveryStrategy.RETRY_IMMEDIATE
        
        self.error_learner.learn_from_recovery(
            error=error,
            error_category=error_category,
            strategy=strategy_used,
            success=success,
            context=context,
            time_to_recover=elapsed_time,
            attempts_needed=context.get("attempt", 1)
        )
        
        self.strategy_optimizer.record_result(
            error_category=error_category,
            strategy=strategy_used,
            success=success,
            time_taken=elapsed_time,
            attempts=context.get("attempt", 1)
        )
        
        logger.info(f"[ReActAgent] Recovery result - Action: {recovery_result.get('action')}, ShouldContinue: {recovery_result.get('should_continue')}")
        
        return recovery_result
    
    def _categorize_error(self, error: Exception, context: Dict[str, Any]) -> ErrorCategory:
        """Categorize an error for learning purposes."""
        error_message = str(error).lower()
        step = context.get("step", "")
        
        if "compile" in step or "compilation" in error_message:
            return ErrorCategory.COMPILATION_ERROR
        elif "test" in step and "fail" in error_message:
            return ErrorCategory.TEST_FAILURE
        elif "timeout" in error_message:
            return ErrorCategory.TIMEOUT
        elif "network" in error_message or "connection" in error_message:
            return ErrorCategory.NETWORK
        elif "api" in error_message or "llm" in error_message:
            return ErrorCategory.LLM_API_ERROR
        elif "parse" in error_message:
            return ErrorCategory.PARSING_ERROR
        else:
            return ErrorCategory.UNKNOWN
    
    async def _compile_with_recovery(self) -> bool:
        """Compile tests with automatic error recovery.
        
        Returns:
            True if compilation successful
        """
        attempt = 0
        
        logger.info("[ReActAgent] 🔨 Starting test compilation (with recovery)")
        
        while not self._stop_requested and not self._terminated:
            attempt += 1
            self._update_state(AgentState.COMPILING, f"🔨 Attempt {attempt}: Compiling tests...")
            
            logger.info(f"[ReActAgent] 🔨 Compilation attempt {attempt} - Running Maven compile...")
            
            try:
                result = await self._compile_tests()
                
                if result.success:
                    logger.info(f"[ReActAgent] ✅ Compilation successful - Attempt: {attempt}")
                    self._update_state(AgentState.COMPILING, "✅ Compilation successful")
                    return True
                else:
                    errors = result.data.get("errors", [])
                    self._update_state(
                        AgentState.FIXING,
                        f"❌ Compilation failed with {len(errors)} error(s). Analyzing..."
                    )
                    
                    logger.warning(f"[ReActAgent] ❌ Compilation failed - Errors: {len(errors)}, calling LLM to fix...")
                    
                    error = Exception("Compilation failed: " + "\n".join(errors[:3]))
                    recovery_result = await self._try_recover(
                        error,
                        {"step": "compilation", "attempt": attempt, "compiler_output": "\n".join(errors)}
                    )
                    
                    if not recovery_result.get("should_continue", True):
                        logger.error("[ReActAgent] Compilation error recovery failed")
                        self._update_state(AgentState.FAILED, "Recovery failed, cannot fix compilation errors")
                        return False
                    
                    action = recovery_result.get("action", "retry")
                    logger.info(f"[ReActAgent] 🔧 Compilation recovery action - Action: {action}")
                    
                    if action == "fix":
                        fixed_code = recovery_result.get("fixed_code")
                        if fixed_code:
                            await self._write_test_file(fixed_code)
                            self._update_state(AgentState.FIXING, "🔧 Applied fix, retrying compilation...")
                            logger.info("[ReActAgent] 🔧 Applied LLM fix, retrying compilation...")
                    elif action == "reset":
                        self._update_state(AgentState.FIXING, "🔄 Resetting and regenerating...")
                        logger.info("[ReActAgent] 🔄 Resetting and regenerating tests...")
                        reset_result = await self._execute_with_recovery(
                            self._generate_initial_tests,
                            step_name="regenerating after compilation failure"
                        )
                        if not reset_result.success:
                            return False
                    elif action == "fallback":
                        self._update_state(AgentState.FIXING, "🔄 Trying alternative approach...")
                        logger.info("[ReActAgent] 🔄 Trying alternative approach...")
                    
                    continue
                    
            except Exception as e:
                logger.exception(f"[ReActAgent] ❌ Compilation exception: {e}")
                self._update_state(AgentState.FIXING, f"❌ Compilation error: {str(e)}")
                
                recovery_result = await self._try_recover(
                    e,
                    {"step": "compilation", "attempt": attempt}
                )
                
                if not recovery_result.get("should_continue", True):
                    logger.error("[ReActAgent] ❌ Compilation recovery failed, cannot continue")
                    return False
                
                continue
        
        if self._terminated:
            logger.info("[ReActAgent] ⏹️ Compilation terminated")
        else:
            logger.info("[ReActAgent] ⏹️ Compilation stopped (user request)")
        return False
    
    async def _run_tests_with_recovery(self) -> bool:
        """Run tests with automatic error recovery and partial success handling.
        
        Returns:
            True if tests pass
        """
        attempt = 0
        
        logger.info("[ReActAgent] 🧪 Starting test execution (with recovery and partial success handling)")
        
        while not self._stop_requested and not self._terminated:
            attempt += 1
            self._update_state(AgentState.TESTING, f"🧪 Attempt {attempt}: Running tests...")
            
            logger.info(f"[ReActAgent] 🧪 Test run attempt {attempt} - Running Maven test...")
            
            try:
                result = await self._run_tests()
                
                if result.success:
                    logger.info(f"[ReActAgent] ✅ All tests passed - Attempt: {attempt}")
                    self._update_state(AgentState.TESTING, "✅ All tests passed")
                    return True
                else:
                    failures = result.data.get("failures", [])
                    
                    # P0: Check for partial success scenario
                    test_output = result.data.get("stdout", "") if result.data else ""
                    settings = get_settings()
                    surefire_dir = Path(self.project_path) / settings.project_paths.target_surefire_reports
                    
                    partial_result = self.partial_success_handler.analyze_test_results(
                        test_output=test_output,
                        surefire_reports_dir=surefire_dir if surefire_dir.exists() else None
                    )
                    
                    # Check if we should use incremental fix
                    if partial_result.has_partial_success:
                        logger.info(f"[ReActAgent] 🔄 Partial success detected - "
                                   f"Passed: {partial_result.passed_tests}, Failed: {partial_result.failed_tests}")
                        
                        if self.partial_success_handler.should_attempt_incremental_fix(partial_result):
                            logger.info("[ReActAgent] 🔄 Attempting incremental fix for failed tests only")
                            
                            incremental_success = await self._handle_incremental_fix(partial_result)
                            if incremental_success:
                                logger.info("[ReActAgent] ✅ Incremental fix successful")
                                return True
                            else:
                                logger.warning("[ReActAgent] ⚠️ Incremental fix failed, falling back to full fix")
                    
                    self._update_state(
                        AgentState.FIXING,
                        f"❌ {len(failures)} test(s) failed. Analyzing..."
                    )
                    
                    logger.warning(f"[ReActAgent] ❌ Tests failed - Failures: {len(failures)}, calling LLM to fix...")
                    
                    error = Exception(f"Test failures: {len(failures)} tests failed")
                    recovery_result = await self._try_recover(
                        error,
                        {"step": "test_execution", "attempt": attempt, "failures": failures}
                    )
                    
                    if not recovery_result.get("should_continue", True):
                        logger.error("[ReActAgent] Test failure recovery failed")
                        self._update_state(AgentState.FAILED, "Recovery failed, cannot fix test failures")
                        return False
                    
                    action = recovery_result.get("action", "retry")
                    logger.info(f"[ReActAgent] 🔧 Test recovery action - Action: {action}")
                    
                    if action == "fix":
                        fixed_code = recovery_result.get("fixed_code")
                        if fixed_code:
                            await self._write_test_file(fixed_code)
                            self._update_state(AgentState.FIXING, "🔧 Applied fix, retrying tests...")
                            logger.info("[ReActAgent] 🔧 Applied LLM fix, retrying tests...")
                    elif action == "reset":
                        self._update_state(AgentState.FIXING, "🔄 Resetting and regenerating...")
                        logger.info("[ReActAgent] 🔄 Resetting and regenerating tests...")
                        reset_result = await self._execute_with_recovery(
                            self._generate_initial_tests,
                            step_name="regenerating after test failure"
                        )
                        if not reset_result.success:
                            return False
                    
                    continue
                    
            except Exception as e:
                logger.exception(f"[ReActAgent] ❌ Test execution exception: {e}")
                self._update_state(AgentState.FIXING, f"❌ Test execution error: {str(e)}")
                
                recovery_result = await self._try_recover(
                    e,
                    {"step": "test_execution", "attempt": attempt}
                )
                
                if not recovery_result.get("should_continue", True):
                    logger.error("[ReActAgent] ❌ Test recovery failed, cannot continue")
                    return False
                
                continue
        
        if self._terminated:
            logger.info("[ReActAgent] ⏹️ Test execution terminated")
        else:
            logger.info("[ReActAgent] ⏹️ Test execution stopped (user request)")
        return False
    
    async def _write_test_file(self, code: str):
        """Write test code to file.
        
        Args:
            code: Test code to write
        """
        if not self.current_test_file:
            logger.warning("[ReActAgent] Cannot write test file - current_test_file is empty")
            return
        
        try:
            test_file_path = Path(self.project_path) / self.current_test_file
            test_file_path.write_text(code, encoding='utf-8')
            logger.info(f"[ReActAgent] Wrote test file - Path: {test_file_path}, Length: {len(code)}")
        except PermissionError as e:
            logger.error(f"[ReActAgent] Permission denied writing test file: {e}")
            self._update_state(AgentState.FAILED, f"Permission denied: {e}")
        except OSError as e:
            logger.error(f"[ReActAgent] OS error writing test file: {e}")
            self._update_state(AgentState.FAILED, f"File system error: {e}")
        except Exception as e:
            logger.exception(f"[ReActAgent] Failed to write test file: {e}")
            self._update_state(AgentState.FAILED, f"Failed to write test file: {e}")
    
    async def _parse_target_file(self, target_file: str) -> StepResult:
        """Parse the target Java file."""
        logger.info(f"[ReActAgent] Parsing target file - File: {target_file}")
        
        try:
            file_path = Path(self.project_path) / target_file
            if not file_path.exists():
                logger.error(f"[ReActAgent] Target file not found - Path: {file_path}")
                return StepResult(
                    success=False,
                    state=AgentState.FAILED,
                    message=f"File not found: {target_file}"
                )
            
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            logger.debug(f"[ReActAgent] Read file content - Length: {len(source_code)}")
            
            parsed_class = self.java_parser.parse(source_code.encode('utf-8'))
            
            class_info = {
                'name': parsed_class.name,
                'package': parsed_class.package,
                'methods': [
                    {
                        'name': m.name,
                        'return_type': m.return_type,
                        'parameters': m.parameters,
                        'modifiers': m.modifiers,
                        'annotations': m.annotations,
                    }
                    for m in parsed_class.methods
                ],
                'fields': parsed_class.fields,
                'imports': parsed_class.imports,
                'annotations': parsed_class.annotations,
                'source': source_code,
            }
            
            logger.info(f"[ReActAgent] Parsing complete - Class: {class_info.get('name', 'unknown')}, Methods: {len(class_info.get('methods', []))}")
            
            return StepResult(
                success=True,
                state=AgentState.PARSING,
                message=f"Successfully parsed {class_info.get('name', 'unknown')}",
                data={"class_info": class_info, "source_code": source_code}
            )
        except Exception as e:
            logger.exception(f"[ReActAgent] Failed to parse file: {e}")
            return StepResult(
                success=False,
                state=AgentState.FAILED,
                message=f"Error parsing file: {str(e)}"
            )
    
    async def _generate_initial_tests(self, use_streaming: bool = True) -> StepResult:
        """Generate initial test cases with context management, streaming and quality evaluation."""
        logger.info("[ReActAgent] Generating initial tests with P0-P3 enhancements")
        
        try:
            source_code = self.target_class_info.get("source", "")
            
            compressed_context = self.context_compressor.build_context(
                query=f"Generate tests for {self.target_class_info.get('name', 'class')}",
                target_file=None,
                additional_context={"class_info": self.target_class_info}
            )
            
            if compressed_context.snippets_included > 0:
                logger.info(f"[ReActAgent] Context compression - "
                           f"Snippets: {compressed_context.snippets_included}, "
                           f"Tokens: {compressed_context.total_tokens}")
            
            effective_source = compressed_context.content if compressed_context.content else source_code
            
            # Build base prompt
            base_prompt = self.prompt_builder.build_initial_test_prompt(
                class_info=self.target_class_info,
                source_code=effective_source
            )
            
            # P1: Optimize prompt for specific model
            prompt = self._optimize_prompt(base_prompt, "test_generation")
            
            logger.debug(f"[ReActAgent] Initial test prompt - Length: {len(prompt)}, Model: {self.model_name}")
            
            test_code = None
            
            if use_streaming:
                logger.info("[ReActAgent] Using streaming generation")
                
                def on_chunk(chunk: str):
                    if self.progress_callback:
                        self.progress_callback({
                            "type": "streaming_chunk",
                            "chunk": chunk[:100] + "..." if len(chunk) > 100 else chunk
                        })
                
                streaming_result = await self.streaming_generator.generate_with_streaming(
                    prompt=prompt,
                    on_chunk=on_chunk,
                    on_progress=lambda p: logger.debug(f"[ReActAgent] Streaming progress: {p:.1%}")
                )
                
                if streaming_result.success:
                    test_code = self._extract_java_code(streaming_result.complete_code)
                    logger.info(f"[ReActAgent] Streaming generation complete - "
                               f"Tokens: {streaming_result.total_tokens}, "
                               f"Time: {streaming_result.total_time:.2f}s")
                else:
                    logger.warning(f"[ReActAgent] Streaming generation failed: {streaming_result.state}")
                    response = await self.llm_client.agenerate(prompt)
                    test_code = self._extract_java_code(response)
            else:
                response = await self.llm_client.agenerate(prompt)
                test_code = self._extract_java_code(response)
            
            logger.debug(f"[ReActAgent] Extracted test code - Length: {len(test_code)}")
            
            eval_result = self.generation_evaluator.evaluate(
                test_code=test_code,
                target_class_info=self.target_class_info
            )
            
            logger.info(f"[ReActAgent] Generation evaluation - Score: {eval_result.overall_score:.2f}, "
                       f"Acceptable: {eval_result.is_acceptable}")
            
            if not eval_result.is_acceptable:
                critical_issues = eval_result.get_critical_issues()
                if critical_issues:
                    logger.warning(f"[ReActAgent] Critical issues detected: {len(critical_issues)}")
            
            if eval_result.coverage_estimate:
                logger.info(f"[ReActAgent] Estimated coverage potential - "
                           f"Line: {eval_result.coverage_estimate.line_coverage_potential:.1%}, "
                           f"Method: {eval_result.coverage_estimate.method_coverage_potential:.1%}")
            
            prediction_result = self.error_predictor.analyze(test_code, code_type="test")
            if prediction_result.predictions:
                logger.info(f"[ReActAgent] Error prediction - "
                           f"Predictions: {len(prediction_result.predictions)}, "
                           f"Risk: {prediction_result.overall_risk_score:.2f}")
                
                if prediction_result.overall_risk_score > 0.7:
                    logger.warning(f"[ReActAgent] High risk detected, may need additional review")
            
            class_name = self.target_class_info.get("name", "Unknown")
            test_file_name = f"{class_name}Test.java"
            
            settings = get_settings()
            test_dir = Path(self.project_path) / settings.project_paths.src_test_java
            package_path = self.target_class_info.get("package", "").replace(".", "/")
            if package_path:
                test_dir = test_dir / package_path
            
            test_dir.mkdir(parents=True, exist_ok=True)
            test_file_path = test_dir / test_file_name
            
            with open(test_file_path, 'w', encoding='utf-8') as f:
                f.write(test_code)
            
            self.current_test_file = str(test_file_path.relative_to(self.project_path))
            self.working_memory.add_generated_test(
                file=self.current_test_file,
                method="initial",
                code=test_code
            )
            
            logger.info(f"[ReActAgent] Initial test generation complete - TestFile: {self.current_test_file}")
            
            # P1: Record A/B test result if applicable
            if self.ab_test_id and self.current_ab_variant_id:
                self._record_generation_result(success=True, response_time_ms=0)
            
            return StepResult(
                success=True,
                state=AgentState.GENERATING,
                message=f"Generated initial tests: {self.current_test_file}",
                data={"test_file": self.current_test_file, "test_code": test_code}
            )
        except Exception as e:
            logger.exception(f"[ReActAgent] Failed to generate initial tests: {e}")
            return StepResult(
                success=False,
                state=AgentState.FAILED,
                message=f"Error generating tests: {str(e)}"
            )
    
    async def _compile_tests(self) -> StepResult:
        """Compile the generated tests asynchronously with validation."""
        logger.info("[ReActAgent] Compiling tests with validation")

        validation_result = self.tool_validator.validate_tool_call(
            "compile_tests",
            (),
            {"test_file": self.current_test_file}
        )
        
        if not validation_result.valid:
            logger.warning(f"[ReActAgent] Tool validation failed: {validation_result.warnings}")
            if validation_result.has_errors:
                return StepResult(
                    success=False,
                    state=AgentState.FAILED,
                    message=f"Validation failed: {'; '.join(str(i.message) for i in validation_result.issues if hasattr(i, 'message'))}"
                )

        try:
            logger.debug("[ReActAgent] Getting Maven dependency classpath")

            maven_process = await asyncio.create_subprocess_exec(
                "mvn", "dependency:build-classpath", "-Dmdep.outputFile=cp.txt", "-q",
                cwd=self.project_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await maven_process.communicate()

            classpath = ""
            cp_file = Path(self.project_path) / "cp.txt"
            if cp_file.exists():
                classpath = cp_file.read_text(encoding='utf-8').strip()
                logger.debug(f"[ReActAgent] Classpath length: {len(classpath)}")

            settings = get_settings()
            classpath = f"{self.project_path}/{settings.project_paths.target_classes};{self.project_path}/{settings.project_paths.target_test_classes};{classpath}"

            test_file_path = Path(self.project_path) / self.current_test_file

            compile_process = await asyncio.create_subprocess_exec(
                "javac", "-cp", classpath,
                "-d", str(Path(self.project_path) / "target" / "test-classes"),
                str(test_file_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await compile_process.communicate()

            if compile_process.returncode == 0:
                logger.info("[ReActAgent] Compilation successful")
                return StepResult(
                    success=True,
                    state=AgentState.COMPILING,
                    message="Tests compiled successfully"
                )
            else:
                errors = [stderr.decode('utf-8', errors='replace')] if stderr else ["Unknown compilation error"]
                logger.warning(f"[ReActAgent] Compilation failed - Errors: {len(errors)}")
                return StepResult(
                    success=False,
                    state=AgentState.FIXING,
                    message="Compilation failed",
                    data={"errors": errors, "stdout": stdout.decode('utf-8', errors='replace') if stdout else ""}
                )
        except Exception as e:
            logger.exception(f"[ReActAgent] Compilation exception: {e}")
            return StepResult(
                success=False,
                state=AgentState.FAILED,
                message=f"Error compiling tests: {str(e)}"
            )
    
    async def _run_tests(self) -> StepResult:
        """Run the generated tests."""
        logger.info("[ReActAgent] Running tests")
        
        try:
            success = self.maven_runner.run_tests()
            
            if success:
                logger.info("[ReActAgent] All tests passed")
                return StepResult(
                    success=True,
                    state=AgentState.TESTING,
                    message="All tests passed"
                )
            else:
                failures = self._parse_test_failures()
                logger.warning(f"[ReActAgent] Tests failed - Failures: {len(failures)}")
                return StepResult(
                    success=False,
                    state=AgentState.FIXING,
                    message="Some tests failed",
                    data={"failures": failures}
                )
        except Exception as e:
            logger.exception(f"[ReActAgent] Test execution exception: {e}")
            return StepResult(
                success=False,
                state=AgentState.FAILED,
                message=f"Error running tests: {str(e)}"
            )
    
    async def _analyze_coverage(self) -> StepResult:
        """Analyze test coverage."""
        logger.info("[ReActAgent] Analyzing coverage")
        
        try:
            logger.debug("[ReActAgent] Generating coverage report")
            self.maven_runner.generate_coverage()
            
            report = self.coverage_analyzer.parse_report()
            
            if report:
                logger.info(f"[ReActAgent] Coverage analysis complete - Line: {report.line_coverage:.1%}, Branch: {report.branch_coverage:.1%}, Method: {report.method_coverage:.1%}")
                return StepResult(
                    success=True,
                    state=AgentState.ANALYZING,
                    message=f"Coverage: {report.line_coverage:.1%}",
                    data={
                        "line_coverage": report.line_coverage,
                        "branch_coverage": report.branch_coverage,
                        "method_coverage": report.method_coverage,
                        "report": report
                    }
                )
            else:
                logger.warning("[ReActAgent] Failed to parse coverage report")
                return StepResult(
                    success=False,
                    state=AgentState.FAILED,
                    message="Failed to parse coverage report"
                )
        except Exception as e:
            logger.exception(f"[ReActAgent] Coverage analysis exception: {e}")
            return StepResult(
                success=False,
                state=AgentState.FAILED,
                message=f"Error analyzing coverage: {str(e)}"
            )
    
    async def _handle_incremental_fix(self, partial_result: PartialTestResult) -> bool:
        """Handle incremental fix for partial test success.
        
        Args:
            partial_result: Partial test results with passed/failed tests
            
        Returns:
            True if incremental fix was successful
        """
        try:
            # Read current test code
            test_file_path = Path(self.project_path) / self.current_test_file
            with open(test_file_path, 'r', encoding='utf-8') as f:
                current_test_code = f.read()
            
            # Create incremental fix prompt
            fix_prompt = self.partial_success_handler.create_incremental_fix_prompt(
                test_code=current_test_code,
                partial_result=partial_result,
                target_class_info=self.target_class_info
            )
            
            logger.debug(f"[ReActAgent] Incremental fix prompt - Length: {len(fix_prompt)}")
            
            # Call LLM for incremental fix
            response = await self.llm_client.agenerate(fix_prompt)
            fixed_code = self._extract_java_code(response)
            
            # Merge the fix
            merge_result = self.partial_success_handler.merge_incremental_fix(
                original_code=current_test_code,
                fixed_code=fixed_code,
                partial_result=partial_result
            )
            
            if merge_result.success and merge_result.new_test_code:
                # Write the merged code
                await self._write_test_file(merge_result.new_test_code)
                
                logger.info(f"[ReActAgent] Incremental fix applied - "
                           f"Preserved: {len(merge_result.preserved_tests)}, "
                           f"Fixed: {len(merge_result.fixed_tests)}")
                
                # Run tests again to verify
                verify_result = await self._run_tests()
                return verify_result.success
            else:
                logger.error(f"[ReActAgent] Incremental fix merge failed: {merge_result.error_message}")
                return False
                
        except Exception as e:
            logger.exception(f"[ReActAgent] Incremental fix failed: {e}")
            return False
    
    async def _generate_additional_tests(self, coverage_data: Dict[str, Any]) -> StepResult:
        """Generate additional tests for uncovered code with context management."""
        logger.info("[ReActAgent] Generating additional tests with P0 enhancements")
        
        try:
            report = coverage_data.get("report")
            uncovered_info = self._get_uncovered_info(report)
            
            logger.debug(f"[ReActAgent] Uncovered info - Lines: {len(uncovered_info.get('lines', []))}")
            
            test_file_path = Path(self.project_path) / self.current_test_file
            with open(test_file_path, 'r', encoding='utf-8') as f:
                current_test_code = f.read()
            
            prompt = self.prompt_builder.build_additional_tests_prompt(
                class_info=self.target_class_info,
                existing_tests=current_test_code,
                uncovered_info=uncovered_info,
                current_coverage=coverage_data.get("line_coverage", 0.0)
            )
            
            logger.debug(f"[ReActAgent] Additional tests prompt - Length: {len(prompt)}")
            
            response = await self.llm_client.agenerate(prompt)
            additional_tests = self._extract_java_code(response)
            
            logger.debug(f"[ReActAgent] Extracted additional test code - Length: {len(additional_tests)}")
            
            self._append_tests_to_file(test_file_path, additional_tests)
            
            logger.info("[ReActAgent] Additional test generation complete")
            
            return StepResult(
                success=True,
                state=AgentState.OPTIMIZING,
                message="Generated additional tests for uncovered code",
                data={"additional_tests": additional_tests}
            )
        except Exception as e:
            logger.exception(f"[ReActAgent] Failed to generate additional tests: {e}")
            return StepResult(
                success=False,
                state=AgentState.FAILED,
                message=f"Error generating additional tests: {str(e)}"
            )
    
    def _extract_java_code(self, response: str) -> str:
        """Extract Java code from LLM response."""
        return CodeExtractor.extract_java_code(response)
    
    def _parse_test_failures(self) -> List[Dict[str, Any]]:
        """Parse test failures from Maven output."""
        failures = []
        settings = get_settings()
        surefire_dir = Path(self.project_path) / settings.project_paths.target_surefire_reports
        
        if surefire_dir.exists():
            for report_file in surefire_dir.glob("*.txt"):
                content = report_file.read_text()
                if "FAILURE" in content or "ERROR" in content:
                    failures.append({
                        "test_name": report_file.stem,
                        "error": content[:500]
                    })
        
        logger.debug(f"[ReActAgent] Parsed test failures - Failures: {len(failures)}")
        return failures
    
    def _get_uncovered_info(self, report) -> Dict[str, Any]:
        """Get information about uncovered code."""
        uncovered_info = {
            "methods": [],
            "lines": [],
            "branches": []
        }
        
        if report and report.files:
            for file_coverage in report.files:
                for line_num, is_covered in file_coverage.lines:
                    if not is_covered:
                        uncovered_info["lines"].append(line_num)
        
        logger.debug(f"[ReActAgent] Uncovered info - Lines: {len(uncovered_info['lines'])}")
        return uncovered_info
    
    def _append_tests_to_file(self, test_file_path: Path, additional_tests: str) -> bool:
        """Append additional tests to existing test file using smart editor.
        
        Returns:
            True if successful
        """
        try:
            with open(test_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            last_brace = content.rfind('}')
            if last_brace > 0:
                search_pattern = content[last_brace-50:last_brace+1] if last_brace > 50 else content[:last_brace+1]
                
                edit_result = self.smart_editor.apply_search_replace(
                    code=content,
                    search=search_pattern,
                    replace=search_pattern.rstrip('}') + "\n" + additional_tests + "\n}",
                    fuzzy=True
                )
                
                if edit_result.success:
                    with open(test_file_path, 'w', encoding='utf-8') as f:
                        f.write(edit_result.modified_code)
                    logger.debug(f"[ReActAgent] Smart appended tests - Path: {test_file_path}")
                    return True
            
            new_content = content[:last_brace] + "\n" + additional_tests + "\n" + content[last_brace:] if last_brace > 0 else content + "\n" + additional_tests
            
            with open(test_file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            logger.debug(f"[ReActAgent] Appended tests to file - Path: {test_file_path}, AddedLength: {len(additional_tests)}")
            return True
            
        except Exception as e:
            logger.error(f"[ReActAgent] Failed to append tests: {e}")
            return False
    
    async def _apply_incremental_fix(
        self,
        code: str,
        error: str,
        error_location: Optional[tuple] = None
    ) -> Optional[str]:
        """Apply incremental fix using smart editor.
        
        Args:
            code: Original code
            error: Error message
            error_location: (line, column) of error
            
        Returns:
            Fixed code or None
        """
        try:
            edit_result = await self.smart_editor.incremental_fix(
                code=code,
                error=error,
                error_location=error_location
            )
            
            if edit_result.success:
                logger.info(f"[ReActAgent] Incremental fix applied - Type: {edit_result.edit_type}")
                return edit_result.modified_code
            
            logger.warning(f"[ReActAgent] Incremental fix failed: {edit_result.message}")
            return None
            
        except Exception as e:
            logger.error(f"[ReActAgent] Incremental fix exception: {e}")
            return None
    
    def _optimize_prompt(self, base_prompt: str, task_type: str) -> str:
        """Optimize prompt for the configured model.
        
        Args:
            base_prompt: Original prompt
            task_type: Type of task (test_generation, error_fix, etc.)
            
        Returns:
            Optimized prompt
        """
        try:
            # Check if A/B test is configured
            if self.ab_test_id and hasattr(self, 'prompt_optimizer'):
                variant_id, prompt = self.prompt_optimizer.get_prompt_for_test(
                    self.ab_test_id,
                    class_name=self.target_class_info.get('name', 'Unknown') if self.target_class_info else 'Unknown',
                    package=self.target_class_info.get('package', '') if self.target_class_info else '',
                    methods=', '.join([m.get('name', '') for m in self.target_class_info.get('methods', [])]) if self.target_class_info else '',
                    source_code=self.target_class_info.get('source', '') if self.target_class_info else ''
                )
                
                if variant_id and prompt:
                    self.current_ab_variant_id = variant_id
                    logger.debug(f"[ReActAgent] Using A/B test variant: {variant_id}")
                    return prompt
            
            # Fall back to model-specific optimization
            if hasattr(self, 'prompt_optimizer'):
                optimized = self.prompt_optimizer.optimize_for_model(
                    base_prompt=base_prompt,
                    model_name=self.model_name,
                    task_type=task_type
                )
                logger.debug(f"[ReActAgent] Prompt optimized for {self.model_name}")
                return optimized
            
            # Last resort: use convenience function
            return optimize_prompt(base_prompt, self.model_name, task_type)
            
        except Exception as e:
            logger.warning(f"[ReActAgent] Prompt optimization failed: {e}, using original prompt")
            return base_prompt
    
    def _record_generation_result(self, success: bool, response_time_ms: int = 0):
        """Record generation result for A/B testing.
        
        Args:
            success: Whether generation was successful
            response_time_ms: Response time in milliseconds
        """
        if not self.ab_test_id or not self.current_ab_variant_id:
            return
        
        try:
            if hasattr(self, 'prompt_optimizer'):
                self.prompt_optimizer.record_ab_test_result(
                    test_id=self.ab_test_id,
                    variant_id=self.current_ab_variant_id,
                    success=success,
                    response_time_ms=response_time_ms
                )
                logger.debug(f"[ReActAgent] Recorded A/B test result: {success}")
        except Exception as e:
            logger.warning(f"[ReActAgent] Failed to record A/B test result: {e}")
    
    def get_ab_test_analysis(self, test_id: Optional[str] = None) -> Dict[str, Any]:
        """Get A/B test analysis results.
        
        Args:
            test_id: Test ID to analyze, or None for current test
            
        Returns:
            Analysis results
        """
        test_id = test_id or self.ab_test_id
        
        if not test_id or not hasattr(self, 'prompt_optimizer'):
            return {"error": "No A/B test configured"}
        
        try:
            return self.prompt_optimizer.analyze_ab_test(test_id)
        except Exception as e:
            logger.error(f"[ReActAgent] Failed to analyze A/B test: {e}")
            return {"error": str(e)}
    
    async def analyze_test_quality(self, test_code: Optional[str] = None) -> Dict[str, Any]:
        """Analyze test code quality using the quality analyzer.
        
        Args:
            test_code: Test code to analyze, or None to use current test file
            
        Returns:
            Quality analysis report
        """
        if test_code is None:
            if not self.current_test_file:
                return {"error": "No test file available"}
            
            try:
                test_file_path = Path(self.project_path) / self.current_test_file
                test_code = test_file_path.read_text(encoding='utf-8')
            except Exception as e:
                logger.error(f"[ReActAgent] Failed to read test file: {e}")
                return {"error": str(e)}
        
        try:
            report = self.test_quality_analyzer.analyze(test_code)
            
            logger.info(f"[ReActAgent] Quality analysis - Score: {report.overall_score:.1f}, "
                       f"Issues: {report.total_issues}, Critical: {report.critical_issues}")
            
            return {
                "overall_score": report.overall_score,
                "total_issues": report.total_issues,
                "critical_issues": report.critical_issues,
                "test_methods_analyzed": report.test_methods_analyzed,
                "dimension_scores": {
                    dim: {"score": score.score, "grade": score.grade}
                    for dim, score in report.dimension_scores.items()
                },
                "improvement_suggestions": report.improvement_suggestions,
                "report_markdown": self.test_quality_analyzer.generate_report_markdown(report)
            }
        except Exception as e:
            logger.error(f"[ReActAgent] Quality analysis failed: {e}")
            return {"error": str(e)}
    
    async def suggest_refactorings(self, test_code: Optional[str] = None) -> List[Dict[str, Any]]:
        """Suggest refactorings for test code.
        
        Args:
            test_code: Test code to analyze, or None to use current test file
            
        Returns:
            List of refactoring suggestions
        """
        if test_code is None:
            if not self.current_test_file:
                return []
            
            try:
                test_file_path = Path(self.project_path) / self.current_test_file
                test_code = test_file_path.read_text(encoding='utf-8')
            except Exception as e:
                logger.error(f"[ReActAgent] Failed to read test file: {e}")
                return []
        
        try:
            suggestions = self.refactoring_engine.analyze(test_code)
            
            logger.info(f"[ReActAgent] Refactoring analysis - Suggestions: {len(suggestions)}")
            
            return [
                {
                    "type": s.refactoring_type.value,
                    "description": s.description,
                    "location": s.location,
                    "priority": s.priority,
                    "confidence": s.confidence,
                    "impact": s.impact,
                    "rationale": s.rationale,
                    "suggested_code": s.suggested_code,
                    "original_code": s.original_code
                }
                for s in suggestions
            ]
        except Exception as e:
            logger.error(f"[ReActAgent] Refactoring analysis failed: {e}")
            return []
    
    async def apply_refactoring(
        self,
        refactoring_type: str,
        test_code: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Apply a specific refactoring to test code.
        
        Args:
            refactoring_type: Type of refactoring to apply
            test_code: Test code to refactor, or None to use current test file
            **kwargs: Additional arguments for the refactoring
            
        Returns:
            Refactoring result
        """
        if test_code is None:
            if not self.current_test_file:
                return {"success": False, "error": "No test file available"}
            
            try:
                test_file_path = Path(self.project_path) / self.current_test_file
                test_code = test_file_path.read_text(encoding='utf-8')
            except Exception as e:
                logger.error(f"[ReActAgent] Failed to read test file: {e}")
                return {"success": False, "error": str(e)}
        
        try:
            suggestions = self.refactoring_engine.analyze(test_code)
            
            matching_suggestions = [
                s for s in suggestions
                if s.refactoring_type.value == refactoring_type
            ]
            
            if not matching_suggestions:
                logger.warning(f"[ReActAgent] No matching refactoring found: {refactoring_type}")
                return {"success": False, "error": f"No matching refactoring found: {refactoring_type}"}
            
            best_suggestion = max(matching_suggestions, key=lambda s: s.confidence)
            
            result = self.refactoring_engine.apply_refactoring(test_code, best_suggestion)
            
            if result.success:
                logger.info(f"[ReActAgent] Refactoring applied - Type: {refactoring_type}, "
                           f"Changes: {len(result.changes_made)}")
                
                if self.current_test_file:
                    await self._write_test_file(result.refactored_code)
            else:
                logger.warning(f"[ReActAgent] Refactoring failed: {result.errors}")
            
            return {
                "success": result.success,
                "refactoring_type": result.refactoring_type.value,
                "changes_made": result.changes_made,
                "errors": result.errors
            }
        except Exception as e:
            logger.error(f"[ReActAgent] Refactoring failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def auto_refactor_tests(
        self,
        test_code: Optional[str] = None,
        max_refactorings: int = 5,
        min_confidence: float = 0.8
    ) -> Dict[str, Any]:
        """Automatically apply high-confidence refactorings.
        
        Args:
            test_code: Test code to refactor, or None to use current test file
            max_refactorings: Maximum number of refactorings to apply
            min_confidence: Minimum confidence threshold
            
        Returns:
            Auto-refactoring results
        """
        if test_code is None:
            if not self.current_test_file:
                return {"success": False, "error": "No test file available"}
            
            try:
                test_file_path = Path(self.project_path) / self.current_test_file
                test_code = test_file_path.read_text(encoding='utf-8')
            except Exception as e:
                logger.error(f"[ReActAgent] Failed to read test file: {e}")
                return {"success": False, "error": str(e)}
        
        try:
            refactored_code, results = self.refactoring_engine.auto_refactor(
                test_code,
                max_refactorings=max_refactorings,
                min_confidence=min_confidence
            )
            
            logger.info(f"[ReActAgent] Auto-refactoring complete - "
                       f"Applied: {len(results)}, Original length: {len(test_code)}, "
                       f"New length: {len(refactored_code)}")
            
            if results and self.current_test_file:
                await self._write_test_file(refactored_code)
            
            return {
                "success": len(results) > 0,
                "refactorings_applied": len(results),
                "results": [
                    {
                        "type": r.refactoring_type.value,
                        "success": r.success,
                        "changes": r.changes_made
                    }
                    for r in results
                ],
                "summary": self.refactoring_engine.get_refactoring_summary()
            }
        except Exception as e:
            logger.error(f"[ReActAgent] Auto-refactoring failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def execute_test_in_interpreter(
        self,
        test_code: str,
        test_method_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute test code in the code interpreter.
        
        Args:
            test_code: Test code to execute
            test_method_name: Specific test method to run, or None for all
            
        Returns:
            Execution result
        """
        try:
            result = self.code_interpreter.execute_test(
                test_code=test_code,
                test_method_name=test_method_name
            )
            
            logger.info(f"[ReActAgent] Code interpreter execution - "
                       f"Success: {result.success}, "
                       f"TestsRun: {result.tests_run}, "
                       f"Failures: {result.failures}, "
                       f"Errors: {result.errors}")
            
            return {
                "success": result.success,
                "output": result.output,
                "error_output": result.error_output,
                "tests_run": result.tests_run,
                "tests_passed": result.tests_passed,
                "failures": result.failures,
                "errors": result.errors,
                "execution_time": result.execution_time,
                "assertion_failures": result.assertion_failures,
                "runtime_errors": result.runtime_errors
            }
        except Exception as e:
            logger.error(f"[ReActAgent] Code interpreter execution failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def validate_test_with_interpreter(
        self,
        test_code: Optional[str] = None
    ) -> Dict[str, Any]:
        """Validate test code using the code interpreter before compilation.
        
        Args:
            test_code: Test code to validate, or None to use current test file
            
        Returns:
            Validation result with potential issues
        """
        if test_code is None:
            if not self.current_test_file:
                return {"valid": False, "error": "No test file available"}
            
            try:
                test_file_path = Path(self.project_path) / self.current_test_file
                test_code = test_file_path.read_text(encoding='utf-8')
            except Exception as e:
                logger.error(f"[ReActAgent] Failed to read test file: {e}")
                return {"valid": False, "error": str(e)}
        
        try:
            validation_result = self.code_interpreter.validate_test_code(test_code)
            
            logger.info(f"[ReActAgent] Test validation - Valid: {validation_result['valid']}, "
                       f"Issues: {len(validation_result.get('issues', []))}")
            
            return validation_result
        except Exception as e:
            logger.error(f"[ReActAgent] Test validation failed: {e}")
            return {"valid": False, "error": str(e)}
    
    def get_quality_trend(self, last_n: int = 10) -> Dict[str, Any]:
        """Get quality trend from recent analyses.
        
        Args:
            last_n: Number of recent analyses to include
            
        Returns:
            Quality trend data
        """
        try:
            trend = self.test_quality_analyzer.get_quality_trend(last_n)
            logger.debug(f"[ReActAgent] Quality trend: {trend.get('trend', 'unknown')}")
            return trend
        except Exception as e:
            logger.error(f"[ReActAgent] Failed to get quality trend: {e}")
            return {"error": str(e)}
    
    async def run_static_analysis(
        self,
        source_path: Optional[str] = None,
        tools: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Run static analysis on source code.
        
        Args:
            source_path: Path to source file/directory, or None for project
            tools: List of tools to use (spotbugs, pmd, checkstyle)
            
        Returns:
            Static analysis results
        """
        try:
            target_files = None
            if source_path:
                path = Path(source_path)
                if path.is_file():
                    target_files = [str(path)]
            
            results = await self.static_analysis_manager.run_all_analysis(
                target_files=target_files,
                include_tests=False
            )
            
            formatted_results = {}
            total_issues = 0
            
            for tool_type, analysis_result in results.items():
                tool_name = tool_type.name.lower()
                formatted_results[tool_name] = {
                    "success": analysis_result.success,
                    "bug_count": analysis_result.bug_count,
                    "bugs": [
                        {
                            "type": bug.bug_type,
                            "severity": bug.severity.value,
                            "message": bug.message,
                            "class_name": bug.class_name,
                            "method_name": bug.method_name,
                            "line_number": bug.line_number,
                            "suggestion": bug.suggestion
                        }
                        for bug in analysis_result.bugs
                    ]
                }
                total_issues += analysis_result.bug_count
            
            logger.info(f"[ReActAgent] Static analysis complete - "
                       f"Tools: {list(formatted_results.keys())}, "
                       f"Total issues: {total_issues}")
            
            return {
                "success": True,
                "results": formatted_results,
                "total_issues": total_issues
            }
        except Exception as e:
            logger.error(f"[ReActAgent] Static analysis failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def query_error_knowledge(
        self,
        error_message: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Query error knowledge base for similar errors and solutions.
        
        Args:
            error_message: Error message to search for
            limit: Maximum number of results
            
        Returns:
            List of similar errors and their solutions
        """
        try:
            from ..core.error_knowledge_base import ErrorContext, ErrorCategory
            
            error_context = ErrorContext(
                error_message=error_message,
                category=ErrorCategory.UNKNOWN
            )
            
            similar_errors = self.error_knowledge_base.find_similar_errors(
                error_context=error_context,
                min_similarity=0.6,
                max_results=limit
            )
            
            logger.info(f"[ReActAgent] Error knowledge query - "
                       f"Found: {len(similar_errors)} similar errors")
            
            return [
                {
                    "solution_id": result.solution.solution_id,
                    "error_pattern": result.solution.error_pattern,
                    "category": result.solution.category.value,
                    "fix_description": result.solution.fix_description,
                    "fix_code": result.solution.fix_code,
                    "similarity_score": result.similarity_score,
                    "success_rate": result.solution.success_rate,
                    "status": result.solution.status.value
                }
                for result in similar_errors
            ]
        except Exception as e:
            logger.error(f"[ReActAgent] Error knowledge query failed: {e}")
            return []
    
    async def record_error_solution(
        self,
        error_message: str,
        error_category: str,
        solution_description: str,
        success: bool
    ) -> bool:
        """Record an error and its solution to the knowledge base.
        
        Args:
            error_message: The error message
            error_category: Category of the error
            solution_description: Description of the solution
            success: Whether the solution worked
            
        Returns:
            True if recorded successfully
        """
        try:
            from ..core.error_knowledge_base import ErrorContext, ErrorCategory
            
            category = ErrorCategory.UNKNOWN
            try:
                category = ErrorCategory(error_category.lower())
            except ValueError:
                pass
            
            error_context = ErrorContext(
                error_message=error_message,
                category=category
            )
            
            solution_id = self.error_knowledge_base.record_solution(
                error_context=error_context,
                fix_description=solution_description
            )
            
            self.error_knowledge_base.record_outcome(
                error_context=error_context,
                solution_id=solution_id,
                success=success
            )
            
            logger.info(f"[ReActAgent] Recorded error solution - "
                       f"Category: {error_category}, Success: {success}")
            
            return True
        except Exception as e:
            logger.error(f"[ReActAgent] Failed to record error solution: {e}")
            return False
    
    def get_adaptive_strategy_recommendation(
        self,
        error_category: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Get strategy recommendation from adaptive strategy manager.
        
        Args:
            error_category: Category of the error
            context: Additional context for strategy selection
            
        Returns:
            Strategy recommendation with confidence
        """
        try:
            from ..core.parallel_recovery import RecoveryStrategy
            
            available_strategies = list(RecoveryStrategy)
            
            selected_strategy = self.adaptive_strategy_manager.select_strategy(
                error_category=error_category,
                available_strategies=available_strategies,
                context=context or {},
                allow_exploration=True
            )
            
            logger.info(f"[ReActAgent] Adaptive strategy recommendation - "
                       f"Strategy: {selected_strategy.name}, "
                       f"Category: {error_category}")
            
            return {
                "strategy": selected_strategy.name,
                "strategy_value": selected_strategy.value,
                "confidence": 0.8
            }
        except Exception as e:
            logger.error(f"[ReActAgent] Failed to get strategy recommendation: {e}")
            return {"strategy": "DEFAULT", "confidence": 0.0, "error": str(e)}
    
    def record_strategy_outcome(
        self,
        strategy_name: str,
        success: bool,
        execution_time_ms: float,
        error_category: Optional[str] = None
    ) -> None:
        """Record the outcome of a strategy execution.
        
        Args:
            strategy_name: Name of the strategy
            success: Whether it succeeded
            execution_time_ms: Execution time in milliseconds
            error_category: Category of the error being handled
        """
        try:
            from ..core.parallel_recovery import RecoveryStrategy
            
            try:
                strategy = RecoveryStrategy[strategy_name.upper()]
            except KeyError:
                strategy = RecoveryStrategy.DEFAULT
            
            self.adaptive_strategy_manager.record_attempt(
                strategy=strategy,
                error_category=error_category or "unknown",
                success=success,
                execution_time_ms=execution_time_ms,
                context={}
            )
            
            logger.debug(f"[ReActAgent] Recorded strategy outcome - "
                        f"Strategy: {strategy_name}, Success: {success}")
        except Exception as e:
            logger.error(f"[ReActAgent] Failed to record strategy outcome: {e}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get collected performance metrics.
        
        Returns:
            Performance metrics summary
        """
        try:
            report = self.metrics_collector.generate_report()
            
            logger.debug(f"[ReActAgent] Performance metrics - "
                        f"Operations: {report.get('total_operations', 0)}")
            
            return report
        except Exception as e:
            logger.error(f"[ReActAgent] Failed to get performance metrics: {e}")
            return {"error": str(e)}
    
    async def semantic_search_code(
        self,
        query: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Perform semantic search over code using vector store.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of matching code snippets
        """
        if not self.vector_store:
            logger.warning("[ReActAgent] VectorStore not available for semantic search")
            return []
        
        try:
            query_embedding = self._generate_embedding(query)
            if query_embedding is None:
                logger.warning("[ReActAgent] Could not generate embedding for query")
                return []
            
            results = self.vector_store.search(query_embedding=query_embedding, k=limit)
            
            logger.info(f"[ReActAgent] Semantic search - "
                       f"Query: '{query[:50]}...', Results: {len(results)}")
            
            return [
                {
                    "id": r.id,
                    "content": r.text,
                    "score": r.score,
                    "metadata": r.metadata
                }
                for r in results
            ]
        except Exception as e:
            logger.error(f"[ReActAgent] Semantic search failed: {e}")
            return []
    
    async def index_code_for_search(
        self,
        code: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Index code snippet for semantic search.
        
        Args:
            code: Code snippet to index
            metadata: Additional metadata
            
        Returns:
            True if indexed successfully
        """
        if not self.vector_store:
            return False
        
        try:
            embedding = self._generate_embedding(code)
            if embedding is None:
                logger.warning("[ReActAgent] Could not generate embedding for code")
                return False
            
            self.vector_store.add(
                texts=[code],
                embeddings=[embedding],
                metadatas=[metadata or {}]
            )
            
            logger.debug(f"[ReActAgent] Indexed code snippet - Length: {len(code)}")
            return True
        except Exception as e:
            logger.error(f"[ReActAgent] Failed to index code: {e}")
            return False
    
    def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for text using available embedding model.
        
        Uses caching to avoid recomputing embeddings for the same text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector or None if not available
        """
        import hashlib
        
        cache_key = hashlib.md5(text.encode()).hexdigest()
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]
        
        try:
            if hasattr(self, 'embedding_model') and self.embedding_model:
                embedding = self.embedding_model.embed(text)
                self._embedding_cache[cache_key] = embedding
                return embedding
            
            import struct
            
            hash_bytes = hashlib.sha256(text.encode()).digest()
            embedding = []
            for i in range(0, min(len(hash_bytes) * 8, 384), 4):
                if i + 4 <= len(hash_bytes):
                    val = struct.unpack('f', hash_bytes[i:i+4])[0]
                else:
                    val = 0.0
                embedding.append(val)
            
            while len(embedding) < 384:
                embedding.append(0.0)
            
            embedding = embedding[:384]
            self._embedding_cache[cache_key] = embedding
            return embedding
        except Exception as e:
            logger.debug(f"[ReActAgent] Embedding generation failed: {e}")
            return None
    
    def get_build_tool_info(self) -> Dict[str, Any]:
        """Get information about the detected build tool.
        
        Results are cached for performance.
        
        Returns:
            Build tool information
        """
        if self._build_tool_info_cache is not None:
            return self._build_tool_info_cache
        
        self._build_tool_info_cache = {
            "tool_type": self.build_tool_info.tool_type.name,
            "version": self.build_tool_info.version,
            "config_file": str(self.build_tool_info.config_file) if self.build_tool_info.config_file else None,
            "wrapper_available": self.build_tool_info.wrapper_available,
            "executable_path": self.build_tool_info.executable_path
        }
        return self._build_tool_info_cache
    
    async def run_tests_with_build_tool(
        self,
        test_class: Optional[str] = None,
        test_method: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run tests using the detected build tool.
        
        Args:
            test_class: Specific test class to run
            test_method: Specific test method to run
            
        Returns:
            Test results
        """
        if not self.build_runner:
            logger.error("[ReActAgent] No build runner available")
            return {"success": False, "error": "No build runner available"}
        
        try:
            result = await self.build_runner.run_tests(
                test_class=test_class,
                test_method=test_method
            )
            
            logger.info(f"[ReActAgent] Build tool test run - "
                       f"Success: {result.success}, "
                       f"Tests: {result.test_count}, "
                       f"Failures: {result.failure_count}")
            
            return {
                "success": result.success,
                "test_count": result.test_count,
                "failure_count": result.failure_count,
                "error_count": result.error_count,
                "skipped_count": result.skipped_count,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
        except Exception as e:
            logger.error(f"[ReActAgent] Build tool test run failed: {e}")
            return {"success": False, "error": str(e)}
