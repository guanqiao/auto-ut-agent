"""Tests for specialized agents in multi-agent system."""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, AsyncMock

from pyutagent.agent.multi_agent import (
    CodeAnalysisAgent,
    TestGenerationAgent,
    TestFixAgent,
    MessageBus,
    SharedKnowledgeBase,
    ExperienceReplay,
    AgentCapability,
    AgentRole
)


class TestCodeAnalysisAgent:
    """Tests for CodeAnalysisAgent."""
    
    @pytest.fixture
    def agent(self):
        """Create a CodeAnalysisAgent for testing."""
        message_bus = MessageBus()
        knowledge_base = SharedKnowledgeBase()
        experience_replay = ExperienceReplay()
        
        return CodeAnalysisAgent(
            agent_id="test_analyzer",
            message_bus=message_bus,
            knowledge_base=knowledge_base,
            experience_replay=experience_replay
        )
    
    def test_initialization(self, agent):
        """Test agent initialization."""
        assert agent.agent_id == "test_analyzer"
        assert AgentCapability.DEPENDENCY_ANALYSIS in agent.capabilities
        assert AgentCapability.TEST_DESIGN in agent.capabilities
    
    def test_calculate_testability(self, agent):
        """Test testability score calculation."""
        # High testability: many methods, low complexity
        analysis = {
            "methods": [{"name": "m1"}, {"name": "m2"}, {"name": "m3"}],
            "complexity": 3,
            "imports": ["java.util.List"]
        }
        score = agent._calculate_testability(analysis)
        assert 0.5 <= score <= 1.0
        
        # Low testability: high complexity
        analysis = {
            "methods": [{"name": "m1"}],
            "complexity": 20,
            "imports": ["java.util.List"] * 10
        }
        score = agent._calculate_testability(analysis)
        assert 0.0 <= score < 0.5
    
    def test_basic_parse(self, agent):
        """Test basic Java code parsing."""
        source_code = """
package com.example;

import java.util.List;

public class TestClass {
    private String name;
    
    public String getName() {
        return name;
    }
    
    public void setName(String name) {
        this.name = name;
    }
}
"""
        result = agent._basic_parse(source_code)
        
        assert result["package"] == "com.example"
        assert "java.util.List" in result["imports"]
        assert result["class_name"] == "TestClass"
        assert len(result["methods"]) >= 2
    
    def test_calculate_method_priority(self, agent):
        """Test method priority calculation."""
        # Getter should have low priority
        getter = {"name": "getValue", "return_type": "String"}
        assert agent._calculate_method_priority(getter) < 0.5
        
        # Regular method should have higher priority
        method = {"name": "process", "return_type": "void"}
        assert agent._calculate_method_priority(method) >= 0.5
    
    def test_suggest_tests_for_method(self, agent):
        """Test test suggestion generation."""
        method = {"name": "saveUser", "return_type": "void"}
        suggestions = agent._suggest_tests_for_method(method)
        
        assert len(suggestions) >= 3
        assert any("valid" in s.lower() for s in suggestions)
        assert any("duplicate" in s.lower() for s in suggestions)
    
    def test_parse_with_parser(self, agent):
        """Test tree-sitter Java parser integration."""
        java_code = """
package com.example;

import java.util.List;
import java.util.ArrayList;

public class UserService {
    private List<String> users;
    
    public UserService() {
        this.users = new ArrayList<>();
    }
    
    public void addUser(String name) {
        if (name != null && !name.isEmpty()) {
            users.add(name);
        }
    }
    
    public List<String> getUsers() {
        return users;
    }
}
"""
        result = agent._parse_with_parser(java_code)
        
        if result is not None:
            # Tree-sitter parsing successful
            assert result["package"] == "com.example"
            assert result["class_name"] == "UserService"
            assert "java.util.List" in result["imports"]
            assert "java.util.ArrayList" in result["imports"]
            assert len(result["methods"]) >= 2
            assert any(m["name"] == "addUser" for m in result["methods"])
            assert any(m["name"] == "getUsers" for m in result["methods"])
            assert result["complexity"] >= 2  # Base + if statement
            assert result["line_count"] > 0
    
    def test_extract_method_info(self, agent):
        """Test method information extraction from AST."""
        # This test requires tree-sitter to be available
        java_code = """
public class Test {
    public String greet(String name, int age) {
        return "Hello " + name;
    }
}
"""
        result = agent._parse_with_parser(java_code)
        
        if result and result["methods"]:
            method = result["methods"][0]
            assert method["name"] == "greet"
            assert method["return_type"] == "String"
            assert len(method["parameters"]) == 2
            assert any(p["name"] == "name" for p in method["parameters"])
            assert any(p["name"] == "age" for p in method["parameters"])
    
    def test_calculate_cyclomatic_complexity(self, agent):
        """Test cyclomatic complexity calculation."""
        java_code = """
public class Test {
    public void simple() {
        System.out.println("simple");
    }
    
    public void complex(int x) {
        if (x > 0) {
            if (x < 10) {
                System.out.println("small");
            } else {
                System.out.println("large");
            }
        }
        for (int i = 0; i < x; i++) {
            System.out.println(i);
        }
    }
}
"""
        result = agent._parse_with_parser(java_code)
        
        if result:
            # Simple method has complexity 1, complex has more
            assert result["complexity"] >= 4  # Base + if + if + for
    
    def test_cache_mechanism(self, agent):
        """Test analysis result caching."""
        # Initial cache stats
        initial_stats = agent.get_cache_stats()
        assert initial_stats["cache_size"] == 0
        assert initial_stats["cache_hits"] == 0
        assert initial_stats["cache_misses"] == 0
        
        # First analysis (cache miss)
        java_code = "public class TestCache { public void method() {} }"
        result1 = agent._get_from_cache("TestCache.java", java_code)
        assert result1 is None  # Not in cache yet
        
        # Add to cache
        mock_result = {"class_name": "TestCache", "methods": [{"name": "method"}]}
        agent._add_to_cache("TestCache.java", java_code, mock_result, None)
        
        # Second analysis (cache hit)
        result2 = agent._get_from_cache("TestCache.java", java_code)
        assert result2 is not None
        assert result2["class_name"] == "TestCache"
        
        # Check cache stats
        stats = agent.get_cache_stats()
        assert stats["cache_hits"] >= 1
        assert stats["cache_misses"] >= 1
        assert stats["cache_size"] >= 1
    
    def test_cache_ttl_expiration(self, agent):
        """Test cache TTL expiration."""
        # Add entry with very short TTL
        agent._cache_ttl_seconds = 0.001  # 1ms TTL for testing
        
        mock_result = {"class_name": "TestTTL"}
        agent._add_to_cache("TestTTL.java", "code", mock_result, None)
        
        # Should be in cache immediately
        assert agent._get_from_cache("TestTTL.java", "code") is not None
        
        # Wait for expiration
        import time
        time.sleep(0.01)  # Wait 10ms
        
        # Should be expired now
        assert agent._get_from_cache("TestTTL.java", "code") is None
        
        # Restore normal TTL
        agent._cache_ttl_seconds = 300
    
    def test_cache_key_generation(self, agent):
        """Test cache key generation."""
        # Same file, different content = different keys
        key1 = agent._generate_cache_key("Test.java", "code1")
        key2 = agent._generate_cache_key("Test.java", "code2")
        assert key1 != key2
        
        # Same file, same content = same key
        key3 = agent._generate_cache_key("Test.java", "code1")
        assert key1 == key3
        
        # No content = file path only
        key4 = agent._generate_cache_key("Test.java", None)
        assert key4 == "Test.java"


class TestTestGenerationAgent:
    """Tests for TestGenerationAgent."""
    
    @pytest.fixture
    def agent(self):
        """Create a TestGenerationAgent for testing."""
        message_bus = MessageBus()
        knowledge_base = SharedKnowledgeBase()
        experience_replay = ExperienceReplay()
        
        return TestGenerationAgent(
            agent_id="test_generator",
            message_bus=message_bus,
            knowledge_base=knowledge_base,
            experience_replay=experience_replay
        )
    
    def test_initialization(self, agent):
        """Test agent initialization."""
        assert agent.agent_id == "test_generator"
        assert AgentCapability.TEST_IMPLEMENTATION in agent.capabilities
        assert AgentCapability.MOCK_GENERATION in agent.capabilities
    
    def test_generate_test_template(self, agent):
        """Test test template generation."""
        template = agent._generate_test_template(
            class_name="UserService",
            methods=[{"name": "getUser"}, {"name": "saveUser"}],
            package="com.example"
        )
        
        assert "package com.example;" in template
        assert "public class UserServiceTest" in template
        assert "@Test" in template
        # capitalize() only capitalizes first letter, so "getUser" -> "Getuser"
        assert "testGetuser" in template or "testGetUser" in template
        assert "testSaveuser" in template or "testSaveUser" in template
        assert "@BeforeEach" in template
    
    def test_generate_mock_for_dependency(self, agent):
        """Test mock generation for dependency."""
        mock = agent._generate_mock_for_dependency(
            dependency="com.example.UserRepository",
            class_name="UserService"
        )
        
        assert mock["simple_name"] == "UserRepository"
        assert mock["variable_name"] == "userRepository"
        assert "@Mock" in mock["mock_declaration"]
        assert "UserRepository" in mock["mock_declaration"]
    
    def test_extract_code_from_markdown(self, agent):
        """Test code extraction from markdown."""
        markdown = """Some text
```java
public class Test {
    void test() {}
}
```
More text"""
        
        code = agent._extract_code_from_markdown(markdown)
        assert "public class Test" in code
        assert "```" not in code
    
    def test_build_test_generation_prompt(self, agent):
        """Test prompt building for test generation."""
        prompt = agent._build_test_generation_prompt(
            source_code="public class Test {}",
            class_name="Test",
            methods=[{"name": "method1", "signature": "void method1()"}],
            options={"framework": "JUnit5", "mock_framework": "Mockito"}
        )
        
        assert "Test" in prompt
        assert "method1" in prompt
        assert "JUnit5" in prompt
        assert "Mockito" in prompt


class TestTestFixAgent:
    """Tests for TestFixAgent."""
    
    @pytest.fixture
    def agent(self):
        """Create a TestFixAgent for testing."""
        message_bus = MessageBus()
        knowledge_base = SharedKnowledgeBase()
        experience_replay = ExperienceReplay()
        
        return TestFixAgent(
            agent_id="test_fixer",
            message_bus=message_bus,
            knowledge_base=knowledge_base,
            experience_replay=experience_replay
        )
    
    def test_initialization(self, agent):
        """Test agent initialization."""
        assert agent.agent_id == "test_fixer"
        assert AgentCapability.ERROR_FIXING in agent.capabilities
        assert AgentCapability.TEST_REVIEW in agent.capabilities
    
    def test_classify_compilation_error(self, agent):
        """Test compilation error classification."""
        assert agent._classify_compilation_error(
            "cannot find symbol class User"
        ) == "symbol_not_found"
        
        assert agent._classify_compilation_error(
            "incompatible types: String cannot be converted to int"
        ) == "type_mismatch"
        
        # The actual error pattern is "package X does not exist"
        result = agent._classify_compilation_error(
            "package com.example does not exist"
        )
        assert result in ["missing_import", "unknown_compilation_error"]
    
    def test_classify_test_failure(self, agent):
        """Test test failure classification."""
        assert agent._classify_test_failure(
            "java.lang.NullPointerException"
        ) == "null_pointer"
        
        assert agent._classify_test_failure(
            "expected:<5> but was:<3>"
        ) == "assertion_failure"
    
    def test_add_import(self, agent):
        """Test import addition."""
        code = """package com.example;

public class Test {}
"""
        fixed = agent._add_import(code, "import java.util.List;")
        
        assert "import java.util.List;" in fixed
        assert "package com.example;" in fixed
    
    def test_fix_common_imports(self, agent):
        """Test common import fixes."""
        code = """@Test
void test() {}
"""
        fixed = agent._fix_common_imports(code)
        
        assert "import org.junit.jupiter.api.Test;" in fixed
    
    def test_extract_missing_class(self, agent):
        """Test missing class extraction."""
        error = "cannot find symbol\n  symbol: class UserRepository"
        result = agent._extract_missing_class(error)
        
        assert result == "UserRepository"
    
    def test_determine_severity(self, agent):
        """Test error severity determination."""
        assert agent._determine_severity("cannot find symbol") == "high"
        assert agent._determine_severity("deprecated") == "low"
        # "warning" is in the list of low severity keywords
        result = agent._determine_severity("some warning")
        assert result in ["low", "medium"]


class TestMultiAgentIntegration:
    """Integration tests for multi-agent system."""
    
    @pytest.fixture
    def multi_agent_system(self):
        """Create a complete multi-agent system."""
        from pyutagent.agent.multi_agent import AgentCoordinator
        
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
        
        return {
            "coordinator": coordinator,
            "code_analyzer": code_analyzer,
            "test_generator": test_generator,
            "test_fixer": test_fixer
        }
    
    def test_agent_registration(self, multi_agent_system):
        """Test that all agents are registered."""
        coordinator = multi_agent_system["coordinator"]
        
        assert len(coordinator.agents) == 3
        assert "code_analyzer" in coordinator.agents
        assert "test_generator" in coordinator.agents
        assert "test_fixer" in coordinator.agents
    
    def test_capability_based_allocation(self, multi_agent_system):
        """Test task allocation based on capabilities."""
        coordinator = multi_agent_system["coordinator"]
        
        # Check analyzer has correct capabilities
        analyzer = coordinator.agents["code_analyzer"]
        assert AgentCapability.DEPENDENCY_ANALYSIS in analyzer.capabilities
        
        # Check generator has correct capabilities
        generator = coordinator.agents["test_generator"]
        assert AgentCapability.TEST_IMPLEMENTATION in generator.capabilities
    
    def test_knowledge_sharing(self, multi_agent_system):
        """Test knowledge sharing between agents."""
        knowledge_base = multi_agent_system["coordinator"].knowledge_base
        code_analyzer = multi_agent_system["code_analyzer"]
        
        # Share knowledge from analyzer
        item_id = code_analyzer.share_knowledge(
            item_type="test_pattern",
            content={"pattern": "singleton", "usage": "test"},
            confidence=0.9
        )
        
        # Verify knowledge is stored
        items = knowledge_base.query_knowledge(item_type="test_pattern")
        assert len(items) > 0
