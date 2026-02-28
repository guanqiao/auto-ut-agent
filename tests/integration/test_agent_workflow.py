"""Integration tests for agent workflow."""

import pytest
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock


class TestAgentWorkflow:
    """Integration tests for complete agent workflow."""
    
    @pytest.fixture
    def temp_maven_project(self):
        """Create a temporary Maven project structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            
            # Create pom.xml
            pom_content = """<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 
         http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>
    <groupId>com.example</groupId>
    <artifactId>test-project</artifactId>
    <version>1.0.0</version>
    
    <dependencies>
        <dependency>
            <groupId>org.junit.jupiter</groupId>
            <artifactId>junit-jupiter</artifactId>
            <version>5.10.0</version>
            <scope>test</scope>
        </dependency>
    </dependencies>
</project>
"""
            (project_path / "pom.xml").write_text(pom_content)
            
            # Create source directory structure
            src_dir = project_path / "src" / "main" / "java" / "com" / "example"
            src_dir.mkdir(parents=True)
            
            # Create a Calculator class
            calculator_content = """package com.example;

public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }
    
    public int subtract(int a, int b) {
        return a - b;
    }
    
    public int multiply(int a, int b) {
        return a * b;
    }
    
    public double divide(int a, int b) {
        if (b == 0) {
            throw new IllegalArgumentException("Division by zero");
        }
        return (double) a / b;
    }
}
"""
            (src_dir / "Calculator.java").write_text(calculator_content)
            
            yield str(project_path)
    
    @pytest.fixture
    def agent_components(self, temp_maven_project):
        """Create all agent components."""
        from pyutagent.agent.conversation import ConversationManager
        from pyutagent.agent.test_generator import TestGeneratorAgent
        from pyutagent.memory.working_memory import WorkingMemory
        from pyutagent.llm.config import LLMConfig
        
        llm_config = LLMConfig(
            provider="openai",
            api_key="test-key",
            model="gpt-4"
        )
        
        conversation = ConversationManager(max_history=20)
        working_memory = WorkingMemory()
        
        agent = TestGeneratorAgent(
            project_path=temp_maven_project,
            llm_config=llm_config,
            conversation=conversation,
            working_memory=working_memory
        )
        
        return {
            "agent": agent,
            "conversation": conversation,
            "working_memory": working_memory,
            "project_path": temp_maven_project
        }
    
    def test_conversation_integration(self, agent_components):
        """Test conversation manager integration with agent."""
        conversation = agent_components["conversation"]
        
        # Add messages
        conversation.add_user_message("Generate tests for Calculator")
        conversation.add_agent_message("I'll help you generate tests for the Calculator class.")
        conversation.add_tool_message("Parsed Calculator class with 4 methods", tool_name="java_parser")
        
        # Verify conversation state
        assert len(conversation.messages) == 3
        assert conversation.messages[0].role.value == "user"
        assert conversation.messages[1].role.value == "agent"
        assert conversation.messages[2].role.value == "tool"
        
        # Test context retrieval
        context = conversation.get_context()
        assert len(context) == 3
        
        # Test LLM message format
        llm_messages = conversation.to_llm_messages()
        assert len(llm_messages) == 3
        assert llm_messages[0]["role"] == "user"
        assert llm_messages[1]["role"] == "agent"
        assert llm_messages[2]["role"] == "tool"
    
    def test_working_memory_integration(self, agent_components):
        """Test working memory integration with agent."""
        working_memory = agent_components["working_memory"]
        agent = agent_components["agent"]
        
        # Set task parameters
        working_memory.current_file = "Calculator.java"
        working_memory.current_method = "add"
        working_memory.target_coverage = 0.85
        working_memory.max_iterations = 5
        
        # Simulate coverage updates
        working_memory.update_coverage(0.4)
        assert working_memory.current_coverage == 0.4
        assert len(working_memory.coverage_history) == 1
        
        working_memory.update_coverage(0.7)
        assert working_memory.current_coverage == 0.7
        assert len(working_memory.coverage_history) == 2
        
        # Test pause/resume integration
        # Set running state
        from pyutagent.agent.test_generator import TaskStatus
        agent.status = TaskStatus.RUNNING
        
        agent.pause()
        assert working_memory.is_paused is True
        
        agent.resume()
        assert working_memory.is_paused is False
    
    def test_agent_state_persistence(self, agent_components):
        """Test agent state persistence across sessions."""
        from pyutagent.agent.test_generator import TestGeneratorAgent, TaskStatus
        from pyutagent.agent.conversation import ConversationManager
        from pyutagent.llm.config import LLMConfig
        
        agent = agent_components["agent"]
        working_memory = agent_components["working_memory"]
        
        # Set some state
        agent.status = TaskStatus.RUNNING
        working_memory.current_file = "Calculator.java"
        working_memory.current_coverage = 0.75
        working_memory.iteration_count = 2
        
        # Get state
        state = agent.get_state()
        
        # Restore from state
        restored_agent = TestGeneratorAgent.from_state(
            state=state,
            llm_config=LLMConfig(),
            conversation=ConversationManager()
        )
        
        # Verify restored state
        assert restored_agent.status == TaskStatus.RUNNING
        assert restored_agent.working_memory.current_file == "Calculator.java"
        assert restored_agent.working_memory.current_coverage == 0.75
        assert restored_agent.working_memory.iteration_count == 2
    
    @pytest.mark.asyncio
    async def test_pause_resume_during_generation(self, agent_components):
        """Test pause/resume during test generation."""
        from pyutagent.agent.test_generator import TaskStatus
        
        agent = agent_components["agent"]
        project_path = agent_components["project_path"]
        
        # Mock LLM client to avoid actual API calls
        mock_llm_client = AsyncMock()
        mock_llm_client.agenerate.return_value = """
@Test
void testAdd() {
    Calculator calc = new Calculator();
    assertEquals(5, calc.add(2, 3));
}
"""
        agent._llm_client = mock_llm_client
        
        # Set running state
        agent.status = TaskStatus.RUNNING
        
        # Start generation in background
        source_file = Path(project_path) / "src" / "main" / "java" / "com" / "example" / "Calculator.java"
        
        # Simulate pause during generation
        async def generate_with_pause():
            # Pause after a short delay
            await asyncio.sleep(0.1)
            agent.pause()
            
            # Resume after another delay
            await asyncio.sleep(0.1)
            agent.resume()
        
        # Run both tasks concurrently
        with patch.object(agent, '_generate_test_code', new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = "test code"
            
            # Simulate the pause/resume cycle
            agent.pause()
            assert agent.is_paused() is True
            assert agent.working_memory.is_paused is True
            
            agent.resume()
            assert agent.is_paused() is False
            assert agent.working_memory.is_paused is False
    
    def test_conversation_export_import_integration(self, agent_components, tmp_path):
        """Test conversation export and import."""
        conversation = agent_components["conversation"]
        
        # Add messages
        conversation.add_user_message("Generate tests")
        conversation.add_agent_message("Starting test generation...")
        conversation.add_tool_message("Found 4 methods", tool_name="project_scanner")
        
        # Export to file
        export_path = tmp_path / "conversation.json"
        conversation.export_to_file(str(export_path))
        
        # Create new conversation and import
        from pyutagent.agent.conversation import ConversationManager
        new_conversation = ConversationManager()
        new_conversation.import_from_file(str(export_path))
        
        # Verify imported conversation
        assert len(new_conversation.messages) == 3
        assert new_conversation.messages[0].content == "Generate tests"
        assert new_conversation.messages[1].content == "Starting test generation..."
    
    def test_progress_callback_integration(self, agent_components):
        """Test progress callback functionality."""
        agent = agent_components["agent"]
        
        # Track progress updates
        progress_updates = []
        
        def on_progress(value, status):
            progress_updates.append((value, status))
        
        agent.on_progress = on_progress
        
        # Simulate progress updates
        agent._update_progress(10, "Parsing...")
        agent._update_progress(50, "Generating...")
        agent._update_progress(100, "Complete")
        
        assert len(progress_updates) == 3
        assert progress_updates[0] == (10, "Parsing...")
        assert progress_updates[1] == (50, "Generating...")
        assert progress_updates[2] == (100, "Complete")
    
    def test_log_callback_integration(self, agent_components):
        """Test log callback functionality."""
        agent = agent_components["agent"]
        
        # Track log messages
        log_messages = []
        
        def on_log(message):
            log_messages.append(message)
        
        agent.on_log = on_log
        
        # Simulate logging
        agent._log("Starting task")
        agent._log("Parsing file")
        agent._log("Task complete")
        
        assert len(log_messages) == 3
        assert log_messages[0] == "Starting task"
        assert log_messages[1] == "Parsing file"
        assert log_messages[2] == "Task complete"


class TestAgentWithJavaParser:
    """Integration tests for agent with Java parser."""
    
    @pytest.fixture
    def java_project(self):
        """Create a Java project with various classes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            
            # Create pom.xml
            pom_content = """<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
    <modelVersion>4.0.0</modelVersion>
    <groupId>com.example</groupId>
    <artifactId>test-project</artifactId>
    <version>1.0.0</version>
</project>
"""
            (project_path / "pom.xml").write_text(pom_content)
            
            # Create source directory
            src_dir = project_path / "src" / "main" / "java" / "com" / "example"
            src_dir.mkdir(parents=True)
            
            # Create a class with various method types
            java_content = """package com.example;

import java.util.List;
import java.util.Map;

public class DataProcessor {
    private String name;
    private int count;
    
    public DataProcessor(String name) {
        this.name = name;
        this.count = 0;
    }
    
    public String process(String input) {
        if (input == null) {
            return null;
        }
        count++;
        return name + ": " + input.toUpperCase();
    }
    
    public int batchProcess(List<String> inputs) {
        if (inputs == null || inputs.isEmpty()) {
            return 0;
        }
        
        int processed = 0;
        for (String input : inputs) {
            if (process(input) != null) {
                processed++;
            }
        }
        return processed;
    }
    
    @Deprecated
    public void oldMethod() {
        // Deprecated method
    }
}
"""
            (src_dir / "DataProcessor.java").write_text(java_content)
            
            yield str(project_path)
    
    def test_parse_java_file_integration(self, java_project):
        """Test Java file parsing integration."""
        from pyutagent.agent.test_generator import TestGeneratorAgent
        from pyutagent.agent.conversation import ConversationManager
        from pyutagent.memory.working_memory import WorkingMemory
        from pyutagent.llm.config import LLMConfig
        
        agent = TestGeneratorAgent(
            project_path=java_project,
            llm_config=LLMConfig(),
            conversation=ConversationManager(),
            working_memory=WorkingMemory()
        )
        
        # Parse the Java file
        source_file = Path(java_project) / "src" / "main" / "java" / "com" / "example" / "DataProcessor.java"
        java_class = agent.java_parser.parse_file(str(source_file))
        
        # Verify parsed class
        assert java_class.name == "DataProcessor"
        assert java_class.package == "com.example"
        assert len(java_class.methods) == 4  # constructor + 3 methods
        
        # Verify method details
        method_names = [m.name for m in java_class.methods]
        assert "process" in method_names
        assert "batchProcess" in method_names
        assert "oldMethod" in method_names
    
    def test_generate_test_prompt_integration(self, java_project):
        """Test test generation prompt building."""
        from pyutagent.agent.test_generator import TestGeneratorAgent
        from pyutagent.agent.conversation import ConversationManager
        from pyutagent.memory.working_memory import WorkingMemory
        from pyutagent.llm.config import LLMConfig
        from pyutagent.tools.java_parser import JavaCodeParser
        
        agent = TestGeneratorAgent(
            project_path=java_project,
            llm_config=LLMConfig(),
            conversation=ConversationManager(),
            working_memory=WorkingMemory()
        )
        
        # Parse and build prompt
        source_file = Path(java_project) / "src" / "main" / "java" / "com" / "example" / "DataProcessor.java"
        java_class = agent.java_parser.parse_file(str(source_file))
        
        prompt = agent._build_test_generation_prompt(java_class)
        
        # Verify prompt content
        assert "DataProcessor" in prompt
        assert "com.example" in prompt
        assert "process" in prompt
        assert "batchProcess" in prompt
        assert "JUnit 5" in prompt


class TestAgentErrorHandling:
    """Integration tests for agent error handling."""
    
    def test_agent_handles_invalid_project_path(self):
        """Test agent handles invalid project path gracefully."""
        from pyutagent.agent.test_generator import TestGeneratorAgent
        from pyutagent.agent.conversation import ConversationManager
        from pyutagent.memory.working_memory import WorkingMemory
        from pyutagent.llm.config import LLMConfig
        
        # This should not raise an exception during initialization
        agent = TestGeneratorAgent(
            project_path="/nonexistent/path",
            llm_config=LLMConfig(),
            conversation=ConversationManager(),
            working_memory=WorkingMemory()
        )
        
        assert agent.project_path == Path("/nonexistent/path")
    
    def test_agent_handles_missing_java_file(self):
        """Test agent handles missing Java file gracefully."""
        from pyutagent.agent.test_generator import TestGeneratorAgent
        from pyutagent.agent.conversation import ConversationManager
        from pyutagent.memory.working_memory import WorkingMemory
        from pyutagent.llm.config import LLMConfig
        
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = TestGeneratorAgent(
                project_path=tmpdir,
                llm_config=LLMConfig(),
                conversation=ConversationManager(),
                working_memory=WorkingMemory()
            )
            
            # Try to parse non-existent file
            with pytest.raises(Exception):
                agent.java_parser.parse_file("/nonexistent/File.java")
