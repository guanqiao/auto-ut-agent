"""Tests for test generator agent."""

import pytest
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock


class TestTestGeneratorAgent:
    """Test suite for TestGeneratorAgent."""
    
    @pytest.fixture
    def temp_project(self):
        """Create a temporary Maven project."""
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
            
            # Create a Java file
            java_content = """package com.example;

public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }
    
    public int subtract(int a, int b) {
        return a - b;
    }
}
"""
            (src_dir / "Calculator.java").write_text(java_content)
            
            yield str(project_path)
    
    @pytest.fixture
    def agent(self, temp_project):
        """Create a test generator agent."""
        from pyutagent.agent.test_generator import TestGeneratorAgent
        from pyutagent.agent.conversation import ConversationManager
        from pyutagent.memory.working_memory import WorkingMemory
        from pyutagent.llm.config import LLMConfig
        
        llm_config = LLMConfig(
            provider="openai",
            api_key="test-key",
            model="gpt-4"
        )
        
        conversation = ConversationManager()
        working_memory = WorkingMemory()
        
        return TestGeneratorAgent(
            project_path=temp_project,
            llm_config=llm_config,
            conversation=conversation,
            working_memory=working_memory
        )
    
    def test_init(self, agent):
        """Test agent initialization."""
        from pyutagent.agent.test_generator import TaskStatus
        
        assert agent.status == TaskStatus.IDLE
        assert agent.is_paused() is False
        assert agent.is_running() is False
    
    @pytest.mark.asyncio
    async def test_pause_resume(self, agent):
        """Test pause and resume functionality."""
        from pyutagent.agent.test_generator import TaskStatus
        
        # Simulate running state
        agent.status = TaskStatus.RUNNING
        
        # Pause
        agent.pause()
        assert agent.is_paused() is True
        assert agent.working_memory.is_paused is True
        
        # Resume
        agent.resume()
        assert agent.is_paused() is False
        assert agent.working_memory.is_paused is False
    
    @pytest.mark.asyncio
    async def test_check_pause(self, agent):
        """Test check pause functionality."""
        from pyutagent.agent.test_generator import TaskStatus
        
        agent.status = TaskStatus.RUNNING
        
        # Should not block when not paused
        await agent._check_pause()
        assert agent.status == TaskStatus.RUNNING
    
    def test_build_test_generation_prompt(self, agent):
        """Test building test generation prompt."""
        from pyutagent.tools.java_parser import JavaClass, JavaMethod
        
        java_class = JavaClass(
            package="com.example",
            name="Calculator",
            methods=[
                JavaMethod(
                    name="add",
                    return_type="int",
                    parameters=[("int", "a"), ("int", "b")],
                    modifiers=["public"],
                    annotations=[],
                    start_line=1,
                    end_line=3
                )
            ],
            fields=[],
            imports=[],
            annotations=[]
        )
        
        prompt = agent._build_test_generation_prompt(java_class)
        
        assert "Calculator" in prompt
        assert "add" in prompt
        assert "int a" in prompt
        assert "int b" in prompt
    
    def test_generate_basic_test_template(self, agent):
        """Test generating basic test template."""
        from pyutagent.tools.java_parser import JavaClass
        
        java_class = JavaClass(
            package="com.example",
            name="Calculator",
            methods=[],
            fields=[],
            imports=[],
            annotations=[]
        )
        
        template = agent._generate_basic_test_template(java_class)
        
        assert "package com.example" in template
        assert "CalculatorTest" in template
        assert "@Test" in template
        assert "@BeforeEach" in template
    
    def test_save_test_file(self, agent, temp_project):
        """Test saving test file."""
        source_file = Path(temp_project) / "src" / "main" / "java" / "com" / "example" / "Calculator.java"
        test_code = "public class CalculatorTest {}"
        
        test_path = agent._save_test_file(str(source_file), test_code)
        
        assert Path(test_path).exists()
        assert "CalculatorTest.java" in test_path
        assert Path(test_path).read_text() == test_code
    
    def test_append_test_code(self, agent, temp_project):
        """Test appending test code."""
        test_file = Path(temp_project) / "src" / "test" / "java" / "com" / "example" / "CalculatorTest.java"
        test_file.parent.mkdir(parents=True)
        test_file.write_text("public class CalculatorTest { }")
        
        additional_code = "\n    @Test\n    void testAdd() {}\n"
        
        agent._append_test_code(str(test_file), additional_code)
        
        content = test_file.read_text()
        assert "testAdd" in content
    
    def test_get_state(self, agent):
        """Test getting state."""
        from pyutagent.agent.test_generator import TaskStatus
        
        agent.status = TaskStatus.RUNNING
        agent.working_memory.current_file = "Test.java"
        
        state = agent.get_state()
        
        assert state["status"] == "RUNNING"
        assert state["working_memory"]["current_file"] == "Test.java"
        assert "project_path" in state
    
    def test_from_state(self, agent):
        """Test restoring from state."""
        from pyutagent.agent.test_generator import TestGeneratorAgent, TaskStatus
        from pyutagent.agent.conversation import ConversationManager
        from pyutagent.llm.config import LLMConfig
        
        state = {
            "status": "PAUSED",
            "working_memory": {
                "current_file": "Test.java",
                "current_method": "testMethod",
                "iteration_count": 2,
                "max_iterations": 10,
                "target_coverage": 0.8,
                "current_coverage": 0.5,
                "coverage_history": [],
                "is_paused": True,
                "processed_files": [],
                "failed_tests": [],
                "generated_tests": [],
                "llm_context": {}
            },
            "project_path": str(agent.project_path)
        }
        
        restored = TestGeneratorAgent.from_state(
            state=state,
            llm_config=LLMConfig(),
            conversation=ConversationManager()
        )
        
        assert restored.status == TaskStatus.PAUSED
        assert restored.working_memory.current_file == "Test.java"
        assert restored.working_memory.is_paused is True


class TestTaskStatus:
    """Test suite for TaskStatus."""
    
    def test_status_values(self):
        """Test status enum values."""
        from pyutagent.agent.test_generator import TaskStatus
        
        assert TaskStatus.IDLE.name == "IDLE"
        assert TaskStatus.RUNNING.name == "RUNNING"
        assert TaskStatus.PAUSED.name == "PAUSED"
        assert TaskStatus.COMPLETED.name == "COMPLETED"
        assert TaskStatus.FAILED.name == "FAILED"
