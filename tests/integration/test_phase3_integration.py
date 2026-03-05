"""Phase 3 Integration Tests."""

import pytest
from datetime import datetime


class TestTaskPlanningCapabilityIntegration:
    """Test Task Planning + Capability Registry integration."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures."""
        from pyutagent.agent.capability_registry import CapabilityRegistry
        CapabilityRegistry.reset_instance()
        yield
        CapabilityRegistry.reset_instance()
    
    def test_planning_with_capability_registry(self):
        """Test task planning with capability registry."""
        from pyutagent.agent.planning import SimpleTaskDecomposer, DecompositionContext
        from pyutagent.agent.capability_registry import (
            CapabilityType,
            AgentCapability,
            get_capability_registry,
        )
        
        registry = get_capability_registry()
        
        cap = AgentCapability(
            name="test_gen",
            capability_type=CapabilityType.TEST_GENERATION,
        )
        cap.score.base_score = 0.9
        registry.register("agent_1", cap, "TestAgent", priority=2)
        
        decomposer = SimpleTaskDecomposer()
        context = DecompositionContext(
            task_description="Generate unit tests for UserService",
            task_type="test_generation",
        )
        result = decomposer.decompose(context)
        
        assert len(result) > 0
        
        provider = registry.find_best_provider(CapabilityType.TEST_GENERATION)
        assert provider is not None
        assert provider.agent_id == "agent_1"


class TestCollaborationPerformanceIntegration:
    """Test Collaboration + Performance Dashboard integration."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures."""
        from pyutagent.agent.capability_registry import CapabilityRegistry
        from pyutagent.agent.performance_dashboard import reset_dashboard
        
        CapabilityRegistry.reset_instance()
        reset_dashboard()
        yield
        CapabilityRegistry.reset_instance()
        reset_dashboard()
    
    def test_collaboration_with_performance_tracking(self):
        """Test collaboration with performance tracking."""
        from pyutagent.agent.capability_registry import (
            CapabilityType,
            AgentCapability,
            get_capability_registry,
        )
        from pyutagent.agent.collaboration import (
            CollaborationOrchestrator,
            Task,
            CollaborationPattern,
        )
        from pyutagent.agent.performance_dashboard import (
            get_dashboard,
            time_operation,
        )
        
        registry = get_capability_registry()
        dashboard = get_dashboard()
        
        cap = AgentCapability(
            name="code_gen",
            capability_type=CapabilityType.CODE_GENERATION,
        )
        cap.score.base_score = 0.8
        registry.register("agent_1", cap, "CodeAgent")
        
        with time_operation("collaboration_test"):
            orchestrator = CollaborationOrchestrator(registry)
            task = Task(
                task_id="task_1",
                task_type="code_generation",
                description="Generate code",
            )
        
        summary = dashboard.get_summary()
        assert summary["total_metrics"] > 0


class TestContextCompressionPerformanceIntegration:
    """Test Context Compression + Performance integration."""
    
    def test_compression_with_performance_tracking(self):
        """Test compression with performance tracking."""
        from pyutagent.agent.context_compression import create_compressor
        from pyutagent.agent.performance_dashboard import (
            get_dashboard,
            reset_dashboard,
            time_operation,
        )
        
        reset_dashboard()
        dashboard = get_dashboard()
        compressor = create_compressor()
        
        code = """
public class Test {
    public void method1() {}
    public void method2() {}
}
"""
        
        with time_operation("compression"):
            result = compressor.compress(code, target_tokens=50)
        
        assert result.original_tokens > 0
        assert result.compressed_tokens > 0
        
        summary = dashboard.get_summary()
        assert summary["total_metrics"] > 0
        
        reset_dashboard()


class TestFullWorkflowIntegration:
    """Test full workflow integration."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures."""
        from pyutagent.agent.capability_registry import CapabilityRegistry
        from pyutagent.agent.performance_dashboard import reset_dashboard
        
        CapabilityRegistry.reset_instance()
        reset_dashboard()
        yield
        CapabilityRegistry.reset_instance()
        reset_dashboard()
    
    def test_full_workflow(self):
        """Test full workflow with all Phase 3 components."""
        from pyutagent.agent.planning import SimpleTaskDecomposer, DecompositionContext
        from pyutagent.agent.capability_registry import (
            CapabilityType,
            AgentCapability,
            get_capability_registry,
        )
        from pyutagent.agent.collaboration import (
            CollaborationOrchestrator,
            Task,
            CollaborationPattern,
        )
        from pyutagent.agent.performance_dashboard import (
            get_dashboard,
            time_operation,
        )
        
        registry = get_capability_registry()
        dashboard = get_dashboard()
        
        for i, cap_type in enumerate([
            CapabilityType.CODE_GENERATION,
            CapabilityType.TEST_GENERATION,
            CapabilityType.BUG_FIXING,
        ]):
            cap = AgentCapability(
                name=cap_type.value,
                capability_type=cap_type,
            )
            cap.score.base_score = 0.8 - i * 0.1
            registry.register(f"agent_{i+1}", cap, f"Agent{i+1}")
        
        with time_operation("full_workflow"):
            decomposer = SimpleTaskDecomposer()
            context = DecompositionContext(
                task_description="Fix bugs in UserService",
                task_type="bug_fixing",
            )
            plan = decomposer.decompose(context)
            
            provider = registry.find_best_provider(CapabilityType.BUG_FIXING)
            
            assert len(plan) > 0
            assert provider is not None
        
        report = dashboard.generate_report()
        assert len(report.metrics_summary) > 0


class TestModuleExports:
    """Test that all Phase 3 modules are properly exported."""
    
    def test_planning_exports(self):
        """Test planning module exports."""
        from pyutagent.agent.planning import (
            TaskDecomposer,
            SimpleTaskDecomposer,
            TemplateTaskDecomposer,
            LLMTaskDecomposer,
            DecompositionStrategy,
            DecompositionContext,
            DependencyGraph,
            DependencyAnalyzer,
            AdvancedDependencyAnalyzer,
            ParallelExecutionEngine,
            ResourcePool,
            SubTaskResult,
        )
        
        assert TaskDecomposer is not None
        assert SimpleTaskDecomposer is not None
    
    def test_capability_registry_exports(self):
        """Test capability registry module exports."""
        from pyutagent.agent.capability_registry import (
            CapabilityType,
            CapabilityScore,
            AgentCapability,
            CapabilityProvider,
            CapabilityRegistry,
            get_capability_registry,
        )
        
        assert CapabilityType is not None
        assert CapabilityRegistry is not None
    
    def test_collaboration_exports(self):
        """Test collaboration module exports."""
        from pyutagent.agent.collaboration import (
            CollaborationPattern,
            Task,
            Bid,
            NegotiationProposal,
            ConsensusVote,
            CollaborationResult,
            CollaborationStrategy,
            DelegationStrategy,
            BiddingStrategy,
            NegotiationStrategy,
            ConsensusStrategy,
            CollaborationOrchestrator,
        )
        
        assert CollaborationPattern is not None
        assert CollaborationOrchestrator is not None
    
    def test_context_compression_exports(self):
        """Test context compression module exports."""
        from pyutagent.agent.context_compression import (
            ContentPriority,
            CompressionLevel,
            ContentType,
            ContentBlock,
            CompressionContext,
            CompressionResult,
            ContentAnalyzer,
            PriorityBasedStrategy,
            SemanticStrategy,
            SummarizationStrategy,
            HybridStrategy,
            ContextCompressor,
            create_compressor,
        )
        
        assert ContextCompressor is not None
        assert create_compressor is not None
    
    def test_performance_dashboard_exports(self):
        """Test performance dashboard module exports."""
        from pyutagent.agent.performance_dashboard import (
            MetricCategory,
            AlertType,
            Severity,
            PerformanceMetric,
            PerformanceAlert,
            PerformanceBaseline,
            PerformanceReport,
            MetricsStore,
            AlertManager,
            BaselineManager,
            PerformanceDashboard,
            OperationTimer,
            get_dashboard,
            reset_dashboard,
            time_operation,
        )
        
        assert PerformanceDashboard is not None
        assert get_dashboard is not None
    
    def test_agent_module_exports(self):
        """Test agent module exports Phase 3 components."""
        from pyutagent.agent import (
            CapabilityType,
            CapabilityRegistry,
            CollaborationOrchestrator,
            ContextCompressor,
            PerformanceDashboard,
        )
        
        assert CapabilityType is not None
        assert CapabilityRegistry is not None
        assert CollaborationOrchestrator is not None
        assert ContextCompressor is not None
        assert PerformanceDashboard is not None
