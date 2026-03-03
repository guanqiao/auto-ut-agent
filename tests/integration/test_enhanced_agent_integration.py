"""Integration tests for EnhancedAgent with all P0/P1/P2/P3 components.

This module tests the integration of all enhancement layers:
- P0: Context management, quality evaluation, partial success handling
- P1: Prompt optimization, error learning, tool orchestration
- P2: Multi-agent collaboration
- P3: Error prediction, strategy optimization, sandbox execution
"""

import asyncio
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

from pyutagent import (
    EnhancedAgent,
    EnhancedAgentConfig,
    IntegrationManager,
    AgentState,
)
from pyutagent.llm.client import LLMClient
from pyutagent.memory.working_memory import WorkingMemory


@pytest.fixture
def temp_project():
    """Create a temporary project directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_path = Path(tmpdir)
        
        # Create a simple Java file
        src_dir = project_path / "src" / "main" / "java" / "com" / "example"
        src_dir.mkdir(parents=True)
        
        java_file = src_dir / "Calculator.java"
        java_file.write_text("""
package com.example;

public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }
    
    public int subtract(int a, int b) {
        return a - b;
    }
}
""")
        
        # Create pom.xml
        pom_file = project_path / "pom.xml"
        pom_file.write_text("""<?xml version="1.0" encoding="UTF-8"?>
<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>com.example</groupId>
    <artifactId>test-project</artifactId>
    <version>1.0.0</version>
</project>
""")
        
        yield project_path


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client."""
    client = Mock(spec=LLMClient)
    client.model = "gpt-4"
    client.generate = AsyncMock(return_value="""
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class CalculatorTest {
    @Test
    public void testAdd() {
        Calculator calc = new Calculator();
        assertEquals(5, calc.add(2, 3));
    }
}
""")
    client.count_tokens = Mock(return_value=100)
    return client


@pytest.fixture
def mock_working_memory():
    """Create a mock working memory."""
    memory = Mock(spec=WorkingMemory)
    memory.get_context = Mock(return_value={})
    memory.store = Mock()
    memory.retrieve = Mock(return_value=None)
    return memory


class TestEnhancedAgentIntegration:
    """Test EnhancedAgent integration with all components."""
    
    @pytest.mark.asyncio
    async def test_enhanced_agent_initialization(self, temp_project, mock_llm_client, mock_working_memory):
        """Test EnhancedAgent initializes all components correctly."""
        config = EnhancedAgentConfig(
            model_name="gpt-4",
            enable_multi_agent=False,
            enable_error_prediction=True,
            enable_strategy_optimization=True,
            enable_sandbox_execution=True,
            enable_user_interaction=False,  # Disable for testing
            enable_smart_analysis=True,
        )
        
        agent = EnhancedAgent(
            llm_client=mock_llm_client,
            working_memory=mock_working_memory,
            project_path=str(temp_project),
            config=config
        )
        
        # Verify P3 components are initialized
        assert agent.error_predictor is not None
        assert agent.strategy_manager is not None
        assert agent.sandbox_executor is not None
        assert agent.smart_analyzer is not None
        
        # Verify config is applied
        assert agent.config.model_name == "gpt-4"
        assert agent.config.enable_error_prediction is True
    
    @pytest.mark.asyncio
    async def test_error_prediction_integration(self, temp_project, mock_llm_client, mock_working_memory):
        """Test error prediction is called during code generation."""
        config = EnhancedAgentConfig(
            enable_error_prediction=True,
            enable_user_interaction=False,
        )
        
        agent = EnhancedAgent(
            llm_client=mock_llm_client,
            working_memory=mock_working_memory,
            project_path=str(temp_project),
            config=config
        )
        
        # Test code with potential error
        test_code = """
public class Test {
    public void broken() {
        String s = null;
        s.length();  // NullPointerException risk
    }
}
"""
        
        # Call prediction
        result = await agent.predict_and_prevent_errors(test_code, "Test.java")
        
        # Verify prediction ran
        assert result["enabled"] is True
        assert "has_errors" in result
        assert "predicted_errors" in result
    
    @pytest.mark.asyncio
    async def test_strategy_optimization_integration(self, temp_project, mock_llm_client, mock_working_memory):
        """Test strategy optimization is used during error recovery."""
        config = EnhancedAgentConfig(
            enable_strategy_optimization=True,
            enable_user_interaction=False,
        )
        
        agent = EnhancedAgent(
            llm_client=mock_llm_client,
            working_memory=mock_working_memory,
            project_path=str(temp_project),
            config=config
        )
        
        # Verify strategy manager is initialized
        assert agent.strategy_manager is not None
        
        # Test error categorization
        syntax_error = Exception("Syntax error at line 10")
        category = agent._categorize_error(syntax_error)
        assert category is not None
        
        # Test strategy retrieval
        strategies = agent._get_available_strategies(category)
        assert len(strategies) > 0
    
    @pytest.mark.asyncio
    async def test_sandbox_execution_integration(self, temp_project, mock_llm_client, mock_working_memory):
        """Test sandbox execution for Java code."""
        config = EnhancedAgentConfig(
            enable_sandbox_execution=True,
            sandbox_security_level="MODERATE",
        )
        
        agent = EnhancedAgent(
            llm_client=mock_llm_client,
            working_memory=mock_working_memory,
            project_path=str(temp_project),
            config=config
        )
        
        # Verify sandbox executor is initialized
        assert agent.sandbox_executor is not None
        
        # Note: Actual Java execution would require Java installed
        # This test verifies the component is properly integrated
    
    @pytest.mark.asyncio
    async def test_smart_analysis_integration(self, temp_project, mock_llm_client, mock_working_memory):
        """Test smart code analyzer integration."""
        config = EnhancedAgentConfig(
            enable_smart_analysis=True,
        )
        
        agent = EnhancedAgent(
            llm_client=mock_llm_client,
            working_memory=mock_working_memory,
            project_path=str(temp_project),
            config=config
        )
        
        # Verify smart analyzer is initialized
        assert agent.smart_analyzer is not None
        
        # Test semantic analysis (using Python file for testing)
        test_file = temp_project / "test_analysis.py"
        test_file.write_text("""
def add(a, b):
    return a + b

class Calculator:
    def multiply(self, a, b):
        return a * b
""")
        
        result = await agent.analyze_code_semantics(str(test_file))
        
        # Verify analysis ran
        assert result["enabled"] is True
        assert "entities_count" in result
    
    @pytest.mark.asyncio
    async def test_enhanced_stats_collection(self, temp_project, mock_llm_client, mock_working_memory):
        """Test enhanced stats include all component information."""
        config = EnhancedAgentConfig(
            enable_error_prediction=True,
            enable_strategy_optimization=True,
            enable_sandbox_execution=True,
            enable_user_interaction=False,
            enable_smart_analysis=True,
            enable_multi_agent=False,
        )
        
        agent = EnhancedAgent(
            llm_client=mock_llm_client,
            working_memory=mock_working_memory,
            project_path=str(temp_project),
            config=config
        )
        
        stats = agent.get_enhanced_stats()
        
        # Verify all P3 capabilities are reported
        assert "p3_capabilities" in stats
        assert stats["p3_capabilities"]["error_prediction"]["enabled"] is True
        assert stats["p3_capabilities"]["strategy_optimization"]["enabled"] is True
        assert stats["p3_capabilities"]["sandbox_execution"]["enabled"] is True
        assert stats["p3_capabilities"]["smart_analysis"]["enabled"] is True
        
        # Verify config is reported
        assert "config" in stats
        assert stats["config"]["enable_error_prediction"] is True


class TestIntegrationManager:
    """Test IntegrationManager with EnhancedAgent."""
    
    @pytest.mark.asyncio
    async def test_integration_manager_initialization(self, temp_project):
        """Test IntegrationManager initializes without circular imports."""
        from pyutagent.agent.integration_manager import IntegrationManager
        
        manager = IntegrationManager(str(temp_project))
        
        # Verify no circular import issues
        assert manager is not None
        assert manager.project_path == temp_project
    
    @pytest.mark.asyncio
    async def test_component_lifecycle(self, temp_project):
        """Test component initialization and shutdown."""
        from pyutagent.agent.integration_manager import IntegrationManager
        
        manager = IntegrationManager(str(temp_project))
        
        # Initialize all components
        success = await manager.initialize_all()
        assert success is True
        
        # Verify components are registered
        status = manager.get_component_status()
        assert len(status) > 0
        
        # Shutdown
        await manager.shutdown()
        
        # Verify health check is stopped
        assert manager._stop_requested is True
    
    @pytest.mark.asyncio
    async def test_create_enhanced_agent(self, temp_project, mock_llm_client, mock_working_memory):
        """Test creating EnhancedAgent through IntegrationManager."""
        from pyutagent.agent.integration_manager import IntegrationManager
        
        manager = IntegrationManager(str(temp_project))
        await manager.initialize_all()
        
        # Create agent
        agent = manager.create_enhanced_agent(
            llm_client=mock_llm_client,
            working_memory=mock_working_memory
        )
        
        # Verify agent is created and registered
        assert agent is not None
        assert isinstance(agent, EnhancedAgent)
        
        component_status = manager.get_component_status("enhanced_agent")
        assert component_status["status"] == "RUNNING"
        
        await manager.shutdown()


class TestP3ComponentsIntegration:
    """Test P3 components work together."""
    
    @pytest.mark.asyncio
    async def test_error_prediction_to_strategy_selection(self, temp_project, mock_llm_client, mock_working_memory):
        """Test error prediction feeds into strategy selection."""
        config = EnhancedAgentConfig(
            enable_error_prediction=True,
            enable_strategy_optimization=True,
            enable_user_interaction=False,
        )
        
        agent = EnhancedAgent(
            llm_client=mock_llm_client,
            working_memory=mock_working_memory,
            project_path=str(temp_project),
            config=config
        )
        
        # Predict errors in problematic code
        problematic_code = """
public class Broken {
    public void method() {
        int x = 1 / 0;  // Division by zero
    }
}
"""
        
        prediction = await agent.predict_and_prevent_errors(problematic_code, "Broken.java")
        
        # If errors predicted, verify strategies are available
        if prediction["has_errors"]:
            for error in prediction["predicted_errors"]:
                # This would be integrated with strategy selection in real usage
                assert "type" in error
                assert "severity" in error
    
    @pytest.mark.asyncio
    async def test_component_health_monitoring(self, temp_project):
        """Test health monitoring of all components."""
        from pyutagent.agent.integration_manager import IntegrationManager
        
        manager = IntegrationManager(str(temp_project))
        await manager.initialize_all()
        
        # Get health status
        health = manager.get_system_health()
        
        # Verify health report structure
        assert "overall_health" in health
        assert "health_score" in health
        assert "total_components" in health
        assert "components" in health
        
        # Health should be healthy after initialization
        assert health["health_score"] > 0
        
        await manager.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
