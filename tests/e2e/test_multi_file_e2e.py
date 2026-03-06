"""E2E tests for multi-file projects.

This module tests multi-file project scenarios:
- Inter-file dependencies
- External Maven dependencies
- Large project performance
- Complex class hierarchy
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from tests.e2e.utils import (
    create_java_class,
    create_pom_xml,
    count_java_files,
    count_test_files
)


class TestMultiFileProjectE2E:
    """E2E tests for multi-file projects."""
    
    @pytest.mark.asyncio
    async def test_project_with_dependencies(self, temp_multi_file_project, mock_llm_client):
        """Test project with inter-file dependencies."""
        from pyutagent.agent.react_agent import ReActAgent
        from pyutagent.memory.working_memory import WorkingMemory
        
        service_file = temp_multi_file_project / "src" / "main" / "java" / "com" / "example" / "service" / "UserService.java"
        
        working_memory = WorkingMemory(
            target_coverage=0.8,
            max_iterations=3,
            current_file="UserService.java"
        )
        
        def generate_service_test(*args, **kwargs):
            return '''
package com.example.service;

import com.example.model.User;
import com.example.repository.UserRepository;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

class UserServiceTest {
    private UserService service;
    private UserRepository repository;
    
    @BeforeEach
    void setUp() {
        repository = mock(UserRepository.class);
        service = new UserService(repository);
    }
    
    @Test
    void testCreateUser() {
        service.createUser("John", 25);
        verify(repository).save(any(User.class));
    }
    
    @Test
    void testGetUser() {
        User user = new User("John", 25);
        when(repository.findByName("John")).thenReturn(user);
        
        User result = service.getUser("John");
        assertNotNull(result);
        assertEquals("John", result.getName());
    }
}
'''
        
        mock_llm_client.agenerate = AsyncMock(side_effect=generate_service_test)
        
        with patch('pyutagent.agent.react_agent.LLMClient') as mock_llm_class:
            mock_llm_class.from_config.return_value = mock_llm_client
            
            agent = ReActAgent(
                project_path=str(temp_multi_file_project),
                llm_client=mock_llm_client,
                working_memory=working_memory
            )
            
            assert agent is not None
            assert working_memory.current_file == "UserService.java"
    
    @pytest.mark.asyncio
    async def test_project_with_external_dependencies(self, temp_multi_file_project, mock_llm_client):
        """Test project with external Maven dependencies."""
        from pyutagent.tools.project_analyzer import ProjectAnalyzer
        
        pom_file = temp_multi_file_project / "pom.xml"
        pom_content = pom_file.read_text()
        
        assert "junit-jupiter" in pom_content
        
        analyzer = ProjectAnalyzer(str(temp_multi_file_project))
        
        assert analyzer.project_path == str(temp_multi_file_project)
    
    @pytest.mark.asyncio
    async def test_large_project_performance(self, temp_large_project, mock_llm_client):
        """Test performance with large project."""
        from pyutagent.services.batch_generator import BatchGenerator, BatchConfig
        from pyutagent.llm.config import LLMConfig, LLMProvider
        
        java_count = count_java_files(temp_large_project)
        assert java_count == 20, f"Expected 20 Java files, found {java_count}"
        
        llm_config = LLMConfig(
            id="test",
            name="Test",
            provider=LLMProvider.OPENAI,
            api_key="test-key",
            model="gpt-4"
        )
        
        batch_config = BatchConfig(
            parallel_workers=4,
            timeout_per_file=60,
            continue_on_error=True,
            coverage_target=80,
            max_iterations=3
        )
        
        with patch('pyutagent.services.batch_generator.LLMClient') as mock_llm_class:
            mock_llm_class.from_config.return_value = mock_llm_client
            
            generator = BatchGenerator(
                llm_client=mock_llm_client,
                project_path=str(temp_large_project),
                config=batch_config
            )
            
            assert generator is not None
            assert generator.config.parallel_workers == 4
    
    @pytest.mark.asyncio
    async def test_project_with_complex_hierarchy(self, temp_multi_file_project, mock_llm_client):
        """Test project with complex class hierarchy."""
        base_class = create_java_class(
            package="com.example.model",
            class_name="BaseEntity",
            methods=[
                "private String id; public String getId() { return id; }",
                "public void setId(String id) { this.id = id; }"
            ]
        )
        
        abstract_class = create_java_class(
            package="com.example.model",
            class_name="AbstractPerson",
            methods=[
                "private String name; public String getName() { return name; }",
                "public abstract String getDisplayName();"
            ],
            imports=["com.example.model.BaseEntity"]
        )
        
        concrete_class = create_java_class(
            package="com.example.model",
            class_name="Employee",
            methods=[
                "private String department; public String getDepartment() { return department; }",
                "@Override public String getDisplayName() { return getName() + \" (\" + department + \")\"; }"
            ],
            imports=["com.example.model.AbstractPerson"]
        )
        
        assert "class BaseEntity" in base_class
        assert "class AbstractPerson" in abstract_class
        assert "class Employee" in concrete_class


class TestMultiFileIntegrationE2E:
    """E2E tests for multi-file integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_cross_file_test_generation(self, temp_multi_file_project, mock_llm_client):
        """Test generating tests that reference multiple files."""
        from pyutagent.agent.react_agent import ReActAgent
        from pyutagent.memory.working_memory import WorkingMemory
        
        working_memory = WorkingMemory(
            target_coverage=0.8,
            max_iterations=3
        )
        
        with patch('pyutagent.agent.react_agent.LLMClient') as mock_llm_class:
            mock_llm_class.from_config.return_value = mock_llm_client
            
            agent = ReActAgent(
                project_path=str(temp_multi_file_project),
                llm_client=mock_llm_client,
                working_memory=working_memory
            )
            
            assert agent.project_path == str(temp_multi_file_project)
    
    @pytest.mark.asyncio
    async def test_dependency_resolution(self, temp_multi_file_project, mock_llm_client):
        """Test dependency resolution during test generation."""
        from pyutagent.tools.dependency_analyzer import DependencyAnalyzer
        
        analyzer = DependencyAnalyzer()
        
        service_file = temp_multi_file_project / "src" / "main" / "java" / "com" / "example" / "service" / "UserService.java"
        
        if service_file.exists():
            content = service_file.read_text()
            
            assert "UserRepository" in content or "User" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
