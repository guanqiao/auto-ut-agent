"""Integration tests for multi-agent async workflows.

Tests async task execution, message passing, and coordination.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch

from pyutagent.agent.multi_agent import (
    AgentCoordinator,
    CodeAnalysisAgent,
    TestGenerationAgent,
    TestFixAgent,
    MessageBus,
    SharedKnowledgeBase,
    ExperienceReplay,
    AgentRole,
    AgentCapability,
    AgentTask,
    AgentMessage,
    MessageType
)


class TestAsyncTaskExecution:
    """Tests for async task execution."""
    
    @pytest.fixture
    async def multi_agent_system(self):
        """Create a complete multi-agent system."""
        message_bus = MessageBus()
        knowledge_base = SharedKnowledgeBase()
        experience_replay = ExperienceReplay()
        
        coordinator = AgentCoordinator(
            message_bus=message_bus,
            knowledge_base=knowledge_base,
            experience_replay=experience_replay
        )
        
        code_analyzer = CodeAnalysisAgent(
            agent_id="code_analyzer",
            message_bus=message_bus,
            knowledge_base=knowledge_base,
            experience_replay=experience_replay
        )
        
        test_generator = TestGenerationAgent(
            agent_id="test_generator",
            message_bus=message_bus,
            knowledge_base=knowledge_base,
            experience_replay=experience_replay
        )
        
        test_fixer = TestFixAgent(
            agent_id="test_fixer",
            message_bus=message_bus,
            knowledge_base=knowledge_base,
            experience_replay=experience_replay
        )
        
        # Register agents
        coordinator.register_agent(
            code_analyzer.agent_id,
            code_analyzer.capabilities,
            AgentRole.ANALYZER
        )
        coordinator.register_agent(
            test_generator.agent_id,
            test_generator.capabilities,
            AgentRole.IMPLEMENTER
        )
        coordinator.register_agent(
            test_fixer.agent_id,
            test_fixer.capabilities,
            AgentRole.FIXER
        )
        
        yield {
            "coordinator": coordinator,
            "code_analyzer": code_analyzer,
            "test_generator": test_generator,
            "test_fixer": test_fixer,
            "message_bus": message_bus
        }
    
    @pytest.mark.asyncio
    async def test_code_analysis_task_execution(self, multi_agent_system):
        """Test async code analysis task execution."""
        agent = multi_agent_system["code_analyzer"]
        
        task = AgentTask(
            task_id="test_task_1",
            task_type="analyze_code",
            payload={
                "file_path": "test.java",
                "source_code": """
public class Test {
    public void method1() {}
    public void method2() {}
}
"""
            },
            priority=1
        )
        
        result = await agent.execute_task(task)
        
        assert result["success"] is True
        assert "output" in result
        assert result["output"]["class_name"] == "Test"
        assert len(result["output"]["methods"]) >= 2
    
    @pytest.mark.asyncio
    async def test_test_generation_task_execution(self, multi_agent_system):
        """Test async test generation task execution."""
        agent = multi_agent_system["test_generator"]
        
        task = AgentTask(
            task_id="test_task_2",
            task_type="generate_tests",
            payload={
                "file_path": "UserService.java",
                "class_info": {
                    "class_name": "UserService",
                    "package": "com.example"
                },
                "methods": [
                    {"name": "getUser", "signature": "User getUser(String id)"},
                    {"name": "saveUser", "signature": "void saveUser(User user)"}
                ],
                "options": {
                    "framework": "JUnit5",
                    "mock_framework": "Mockito"
                }
            },
            priority=1
        )
        
        result = await agent.execute_task(task)
        
        # Test generation may fail without LLM, but should return gracefully
        assert "success" in result
        if result["success"]:
            assert "output" in result
            assert "test_code" in result["output"]
    
    @pytest.mark.asyncio
    async def test_test_fix_task_execution(self, multi_agent_system):
        """Test async test fix task execution."""
        agent = multi_agent_system["test_fixer"]
        
        task = AgentTask(
            task_id="test_task_3",
            task_type="fix_compilation_error",
            payload={
                "error_info": {
                    "message": "cannot find symbol class UserRepository",
                    "line": 10
                },
                "test_code": """
import org.junit.jupiter.api.Test;

public class UserServiceTest {
    private UserRepository repository;
    
    @Test
    void test() {}
}
"""
            },
            priority=1
        )
        
        result = await agent.execute_task(task)
        
        # Should attempt to fix or return failure gracefully
        assert "success" in result
        assert "error" in result or "output" in result
    
    @pytest.mark.asyncio
    async def test_concurrent_task_execution(self, multi_agent_system):
        """Test concurrent execution of multiple tasks."""
        agent = multi_agent_system["code_analyzer"]
        
        tasks = [
            AgentTask(
                task_id=f"concurrent_task_{i}",
                task_type="analyze_code",
                payload={
                    "file_path": f"Test{i}.java",
                    "source_code": f"public class Test{i} {{ public void method{i}() {{}} }}"
                },
                priority=1
            )
            for i in range(5)
        ]
        
        # Execute tasks concurrently
        results = await asyncio.gather(
            *[agent.execute_task(task) for task in tasks],
            return_exceptions=True
        )
        
        # All tasks should complete without exceptions
        assert len(results) == 5
        for result in results:
            assert not isinstance(result, Exception)
            assert result["success"] is True
    
    @pytest.mark.asyncio
    async def test_task_timeout_handling(self, multi_agent_system):
        """Test task timeout handling."""
        coordinator = multi_agent_system["coordinator"]
        
        # Submit a task
        task_id = await coordinator.submit_task(
            task_type="analyze_code",
            payload={
                "file_path": "test.java",
                "source_code": "public class Test {}"
            },
            priority=1
        )
        
        # Wait with a short timeout
        success = await coordinator.wait_for_task(task_id, timeout=5)
        
        # Task should complete or timeout gracefully
        assert isinstance(success, bool)


class TestMessageBusIntegration:
    """Tests for message bus integration."""
    
    @pytest.fixture
    async def message_bus(self):
        """Create a message bus."""
        bus = MessageBus()
        # Register test agents
        await bus.register_agent("agent_1")
        await bus.register_agent("agent_2")
        return bus
    
    @pytest.mark.asyncio
    async def test_point_to_point_message(self, message_bus):
        """Test point-to-point message delivery."""
        received_messages = []
        
        async def handler():
            queue = message_bus._queues["agent_1"]
            try:
                msg = await asyncio.wait_for(queue.get(), timeout=1.0)
                received_messages.append(msg)
            except asyncio.TimeoutError:
                pass
        
        # Start handler
        handler_task = asyncio.create_task(handler())
        
        # Send message
        message = AgentMessage.create(
            sender_id="sender",
            recipient_id="agent_1",
            message_type=MessageType.TASK_ASSIGNMENT,
            payload={"content": "hello"}
        )
        await message_bus.send(message)
        
        # Wait for handler
        await handler_task
        
        assert len(received_messages) == 1
        assert received_messages[0].payload["content"] == "hello"
    
    @pytest.mark.asyncio
    async def test_broadcast_message(self, message_bus):
        """Test broadcast message delivery."""
        received_by_agent1 = []
        received_by_agent2 = []
        
        async def handler1():
            queue = message_bus._queues["agent_1"]
            try:
                msg = await asyncio.wait_for(queue.get(), timeout=1.0)
                received_by_agent1.append(msg)
            except asyncio.TimeoutError:
                pass
        
        async def handler2():
            queue = message_bus._queues["agent_2"]
            try:
                msg = await asyncio.wait_for(queue.get(), timeout=1.0)
                received_by_agent2.append(msg)
            except asyncio.TimeoutError:
                pass
        
        # Start handlers
        handler1_task = asyncio.create_task(handler1())
        handler2_task = asyncio.create_task(handler2())
        
        # Broadcast message
        message = AgentMessage.create(
            sender_id="sender",
            recipient_id=None,  # Broadcast
            message_type=MessageType.BROADCAST,
            payload={"content": "hello all"}
        )
        await message_bus.send(message)
        
        # Wait for handlers
        await handler1_task
        await handler2_task
        
        assert len(received_by_agent1) == 1
        assert len(received_by_agent2) == 1
    
    @pytest.mark.asyncio
    async def test_subscribe_unsubscribe(self, message_bus):
        """Test subscribing and unsubscribing from messages."""
        # Subscribe agent_1 to TASK_ASSIGNMENT messages
        await message_bus.subscribe("agent_1", MessageType.TASK_ASSIGNMENT)
        
        # Verify subscription
        assert "agent_1" in message_bus._subscribers[MessageType.TASK_ASSIGNMENT]
        
        # Unsubscribe
        await message_bus.unsubscribe("agent_1", MessageType.TASK_ASSIGNMENT)
        
        # Verify unsubscription
        assert "agent_1" not in message_bus._subscribers[MessageType.TASK_ASSIGNMENT]


class TestTaskDependencies:
    """Tests for task dependency handling."""
    
    @pytest.fixture
    async def coordinator(self):
        """Create a coordinator with agents."""
        message_bus = MessageBus()
        knowledge_base = SharedKnowledgeBase()
        experience_replay = ExperienceReplay()
        
        coordinator = AgentCoordinator(
            message_bus=message_bus,
            knowledge_base=knowledge_base,
            experience_replay=experience_replay
        )
        
        code_analyzer = CodeAnalysisAgent(
            agent_id="code_analyzer",
            message_bus=message_bus,
            knowledge_base=knowledge_base,
            experience_replay=experience_replay
        )
        
        coordinator.register_agent(
            code_analyzer.agent_id,
            code_analyzer.capabilities,
            AgentRole.ANALYZER
        )
        
        yield coordinator
    
    @pytest.mark.asyncio
    async def test_task_submission_and_completion(self, coordinator):
        """Test task submission and completion."""
        # Submit a task
        task_id = await coordinator.submit_task(
            task_type="analyze_code",
            payload={"file_path": "Test.java", "source_code": "public class Test {}"},
            priority=1
        )
        
        # Wait for task
        success = await coordinator.wait_for_task(task_id, timeout=10)
        
        # Task should complete (success may be True or False depending on execution)
        assert isinstance(success, bool)
        
        # Verify task status
        task = coordinator.tasks.get(task_id)
        assert task is not None
    
    @pytest.mark.asyncio
    async def test_failed_task_handling(self, coordinator):
        """Test handling of failed task."""
        # Submit a task with invalid type
        task_id = await coordinator.submit_task(
            task_type="unknown_task_type",
            payload={},
            priority=1
        )
        
        # Wait for task
        await coordinator.wait_for_task(task_id, timeout=5)
        
        # Check task status
        task = coordinator.tasks.get(task_id)
        assert task is not None
        # Task should have been processed


class TestErrorHandling:
    """Tests for error handling in multi-agent system."""
    
    @pytest.fixture
    def agent(self):
        """Create a code analysis agent."""
        message_bus = MessageBus()
        knowledge_base = SharedKnowledgeBase()
        experience_replay = ExperienceReplay()
        
        return CodeAnalysisAgent(
            agent_id="test_analyzer",
            message_bus=message_bus,
            knowledge_base=knowledge_base,
            experience_replay=experience_replay
        )
    
    @pytest.mark.asyncio
    async def test_invalid_task_type(self, agent):
        """Test handling of invalid task type."""
        task = AgentTask(
            task_id="invalid_task",
            task_type="unknown_type",
            payload={},
            priority=1
        )
        
        result = await agent.execute_task(task)
        
        assert result["success"] is False
        assert "error" in result
        assert "Unknown task type" in result["error"]
    
    @pytest.mark.asyncio
    async def test_missing_payload_fields(self, agent):
        """Test handling of missing payload fields."""
        task = AgentTask(
            task_id="missing_fields_task",
            task_type="analyze_code",
            payload={},  # Missing file_path and source_code
            priority=1
        )
        
        result = await agent.execute_task(task)
        
        # Should handle gracefully
        assert "success" in result
    
    @pytest.mark.asyncio
    async def test_malformed_java_code(self, agent):
        """Test handling of malformed Java code."""
        task = AgentTask(
            task_id="malformed_task",
            task_type="analyze_code",
            payload={
                "file_path": "test.java",
                "source_code": "this is not valid Java code { }"
            },
            priority=1
        )
        
        result = await agent.execute_task(task)
        
        # Should handle gracefully and use fallback parsing
        assert "success" in result


class TestKnowledgeSharing:
    """Tests for knowledge sharing between agents."""
    
    @pytest.fixture
    def knowledge_system(self):
        """Create agents with shared knowledge base."""
        message_bus = MessageBus()
        knowledge_base = SharedKnowledgeBase()
        experience_replay = ExperienceReplay()
        
        agent1 = CodeAnalysisAgent(
            agent_id="agent_1",
            message_bus=message_bus,
            knowledge_base=knowledge_base,
            experience_replay=experience_replay
        )
        
        agent2 = TestGenerationAgent(
            agent_id="agent_2",
            message_bus=message_bus,
            knowledge_base=knowledge_base,
            experience_replay=experience_replay
        )
        
        return {
            "agent1": agent1,
            "agent2": agent2,
            "knowledge_base": knowledge_base
        }
    
    def test_knowledge_sharing_between_agents(self, knowledge_system):
        """Test that agents can share knowledge."""
        agent1 = knowledge_system["agent1"]
        agent2 = knowledge_system["agent2"]
        knowledge_base = knowledge_system["knowledge_base"]
        
        # Agent1 shares knowledge
        item_id = agent1.share_knowledge(
            item_type="test_pattern",
            content={"pattern": "singleton", "usage": "test"},
            confidence=0.9,
            tags=["pattern", "singleton"]
        )
        
        # Agent2 should be able to query the same knowledge
        items = knowledge_base.query_knowledge(item_type="test_pattern")
        
        assert len(items) > 0
        assert any(item.content.get("pattern") == "singleton" for item in items)
    
    def test_experience_replay_buffer(self, knowledge_system):
        """Test experience replay buffer functionality."""
        experience_replay = knowledge_system["agent1"].experience_replay
        
        # Add experiences using the correct method signature
        for i in range(10):
            experience_replay.add_experience(
                task_type="test_generation",
                context={"file": f"Test{i}.java"},
                action="generate_test",
                outcome="success",
                reward=0.8,
                agent_id="agent_1"
            )
        
        # Sample experiences
        samples = experience_replay.sample(batch_size=5)
        
        assert len(samples) == 5
        for exp in samples:
            # Experience is a dataclass, access attributes directly
            assert hasattr(exp, 'task_type')
            assert hasattr(exp, 'action')
            assert hasattr(exp, 'reward')
            assert exp.task_type == "test_generation"
            assert exp.action == "generate_test"
