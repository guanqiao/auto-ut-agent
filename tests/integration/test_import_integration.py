"""Integration tests for module imports.

This module tests that all modules can be imported without circular dependency issues.
"""

import pytest


class TestRootImports:
    """Test root package imports."""
    
    def test_root_package_import(self):
        """Test that root package can be imported."""
        import pyutagent
        
        # Verify version
        assert hasattr(pyutagent, "__version__")
        assert pyutagent.__version__ == "0.1.0"
    
    def test_core_agents_import(self):
        """Test core agent classes can be imported from root."""
        from pyutagent import (
            ReActAgent,
            EnhancedAgent,
            EnhancedAgentConfig,
            IntegrationManager,
        )
        
        # Verify classes are accessible
        assert ReActAgent is not None
        assert EnhancedAgent is not None
        assert EnhancedAgentConfig is not None
        assert IntegrationManager is not None
    
    def test_protocols_import(self):
        """Test protocol classes can be imported from root."""
        from pyutagent import (
            AgentState,
            AgentResult,
            ComponentStatus,
            TestResult,
            CoverageResult,
            ClassInfo,
            MethodInfo,
        )
        
        assert AgentState is not None
        assert AgentResult is not None
        assert ComponentStatus is not None
    
    def test_infrastructure_import(self):
        """Test infrastructure classes can be imported from root."""
        from pyutagent import LLMClient, WorkingMemory
        
        assert LLMClient is not None
        assert WorkingMemory is not None


class TestAgentModuleImports:
    """Test agent module imports."""
    
    def test_base_agent_import(self):
        """Test base agent imports."""
        from pyutagent.agent import BaseAgent, StepResult
        assert BaseAgent is not None
        assert StepResult is not None
    
    def test_react_agent_import(self):
        """Test ReAct agent imports."""
        from pyutagent.agent import ReActAgent
        assert ReActAgent is not None
    
    def test_enhanced_agent_import(self):
        """Test Enhanced agent imports."""
        from pyutagent.agent import EnhancedAgent, EnhancedAgentConfig
        assert EnhancedAgent is not None
        assert EnhancedAgentConfig is not None
    
    def test_integration_manager_import(self):
        """Test IntegrationManager imports."""
        from pyutagent.agent import (
            IntegrationManager,
            get_integration_manager,
            reset_integration_manager,
        )
        assert IntegrationManager is not None
        assert get_integration_manager is not None
        assert reset_integration_manager is not None
    
    def test_p0_components_import(self):
        """Test P0 component imports."""
        from pyutagent.agent import (
            ContextManager,
            CompressionStrategy,
            GenerationEvaluator,
            EvaluationResult,
            QualityDimension,
            PartialSuccessHandler,
            PartialTestResult,
        )
        assert ContextManager is not None
        assert CompressionStrategy is not None
        assert GenerationEvaluator is not None
    
    def test_p1_components_import(self):
        """Test P1 component imports."""
        from pyutagent.agent import (
            PromptOptimizer,
            ModelType,
            PromptStrategy,
            ABTest,
        )
        assert PromptOptimizer is not None
        assert ModelType is not None
    
    def test_p2_components_import(self):
        """Test P2 component imports."""
        from pyutagent.agent import (
            AgentCoordinator,
            AgentCapability,
            AgentRole,
            MessageBus,
            SharedKnowledgeBase,
            ExperienceReplay,
        )
        assert AgentCoordinator is not None
        assert MessageBus is not None
        assert SharedKnowledgeBase is not None


class TestCoreModuleImports:
    """Test core module imports."""
    
    def test_protocols_import(self):
        """Test protocols module imports."""
        from pyutagent.core.protocols import (
            AgentState,
            AgentResult,
            ComponentStatus,
            LLMClientProtocol,
            TestResult,
        )
        assert AgentState is not None
        assert ComponentStatus is not None
    
    def test_p3_components_import(self):
        """Test P3 component imports."""
        from pyutagent.core.error_predictor import ErrorPredictor
        from pyutagent.core.adaptive_strategy import AdaptiveStrategyManager
        from pyutagent.core.sandbox_executor import SandboxExecutor
        from pyutagent.core.smart_analyzer import SmartCodeAnalyzer
        
        assert ErrorPredictor is not None
        assert AdaptiveStrategyManager is not None
        assert SandboxExecutor is not None
        assert SmartCodeAnalyzer is not None
    
    def test_metrics_import(self):
        """Test metrics module imports."""
        from pyutagent.core.metrics import MetricsCollector, get_metrics
        assert MetricsCollector is not None
        assert get_metrics is not None


class TestCLIModuleImports:
    """Test CLI module imports."""
    
    def test_cli_import(self):
        """Test CLI module imports."""
        from pyutagent.cli import main, cli
        assert main is not None
        assert cli is not None
    
    def test_cli_commands_import(self):
        """Test CLI commands imports."""
        from pyutagent.cli.commands import (
            GenerateCommand,
            GenerateAllCommand,
            ScanCommand,
            ConfigCommand,
        )
        assert GenerateCommand is not None
        assert ScanCommand is not None


class TestCircularDependencyPrevention:
    """Test that circular dependencies are prevented."""
    
    def test_integration_manager_no_circular_import(self):
        """Test IntegrationManager can be imported without circular import issues."""
        # This test verifies that the fix for circular imports works
        import sys
        
        # Clear any cached modules to ensure fresh import
        modules_to_clear = [
            k for k in sys.modules.keys()
            if k.startswith('pyutagent')
        ]
        for mod in modules_to_clear:
            if mod in sys.modules:
                del sys.modules[mod]
        
        # Import should succeed without circular import errors
        from pyutagent.agent.integration_manager import IntegrationManager
        from pyutagent.agent.enhanced_agent import EnhancedAgent
        
        # Both should be importable
        assert IntegrationManager is not None
        assert EnhancedAgent is not None
    
    def test_component_status_in_protocols(self):
        """Test ComponentStatus is defined in protocols to avoid circular imports."""
        from pyutagent.core.protocols import ComponentStatus
        from pyutagent.agent import ComponentStatus as AgentComponentStatus
        
        # Both should be the same
        assert ComponentStatus is AgentComponentStatus


class TestMemoryModuleImports:
    """Test memory module imports."""
    
    def test_working_memory_import(self):
        """Test WorkingMemory import."""
        from pyutagent.memory.working_memory import WorkingMemory
        assert WorkingMemory is not None
    
    def test_vector_store_import(self):
        """Test VectorStore import."""
        from pyutagent.memory.vector_store import VectorStore
        assert VectorStore is not None


class TestLLMModuleImports:
    """Test LLM module imports."""
    
    def test_llm_client_import(self):
        """Test LLMClient import."""
        from pyutagent.llm.client import LLMClient
        assert LLMClient is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
