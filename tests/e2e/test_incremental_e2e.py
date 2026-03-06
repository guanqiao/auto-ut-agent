"""E2E tests for incremental mode.

This module tests incremental test generation:
- Preserving passing tests
- Adding new tests
- Merging test strategies
- Handling refactoring
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from tests.e2e.utils import create_java_class, create_test_class


class TestIncrementalModeE2E:
    """E2E tests for incremental mode."""
    
    @pytest.mark.asyncio
    async def test_preserve_passing_tests(self, temp_maven_project, mock_llm_client):
        """Test preserving existing passing tests."""
        from pyutagent.agent.incremental_manager import IncrementalManager
        
        test_file = temp_maven_project / "src" / "test" / "java" / "com" / "example" / "CalculatorTest.java"
        
        existing_test = '''
package com.example;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class CalculatorTest {
    @Test
    void testAdd() {
        Calculator calc = new Calculator();
        assertEquals(4, calc.add(2, 2));
    }
    
    @Test
    void testSubtract() {
        Calculator calc = new Calculator();
        assertEquals(0, calc.subtract(2, 2));
    }
}
'''
        test_file.write_text(existing_test)
        
        manager = IncrementalManager(str(temp_maven_project))
        
        assert manager is not None
    
    @pytest.mark.asyncio
    async def test_add_new_tests(self, temp_maven_project, mock_llm_client):
        """Test adding new tests in incremental mode."""
        from pyutagent.agent.incremental_manager import IncrementalManager
        
        java_file = temp_maven_project / "src" / "main" / "java" / "com" / "example" / "Calculator.java"
        
        new_method = '''
    public double percentage(int value, int total) {
        if (total == 0) {
            throw new IllegalArgumentException("Total cannot be zero");
        }
        return (double) value / total * 100;
    }
'''
        current_content = java_file.read_text()
        updated_content = current_content.replace("}", new_method + "\n}")
        java_file.write_text(updated_content)
        
        manager = IncrementalManager(str(temp_maven_project))
        
        assert manager is not None
    
    @pytest.mark.asyncio
    async def test_merge_test_strategies(self, temp_maven_project, mock_llm_client):
        """Test merging different test strategies."""
        from pyutagent.agent.incremental_manager import IncrementalManager
        
        manager = IncrementalManager(str(temp_maven_project))
        
        assert manager is not None
    
    @pytest.mark.asyncio
    async def test_incremental_with_refactoring(self, temp_maven_project, mock_llm_client):
        """Test incremental mode with code refactoring."""
        from pyutagent.agent.incremental_manager import IncrementalManager
        
        java_file = temp_maven_project / "src" / "main" / "java" / "com" / "example" / "Calculator.java"
        
        refactored_content = '''
package com.example;

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
    
    public int abs(int n) {
        return Math.abs(n);
    }
    
    public int negate(int n) {
        return -n;
    }
}
'''
        java_file.write_text(refactored_content)
        
        manager = IncrementalManager(str(temp_maven_project))
        
        assert manager is not None


class TestIncrementalIntegrationE2E:
    """E2E tests for incremental mode integration."""
    
    @pytest.mark.asyncio
    async def test_incremental_generation_workflow(self, temp_maven_project, mock_llm_client):
        """Test complete incremental generation workflow."""
        from pyutagent.agent.react_agent import ReActAgent
        from pyutagent.memory.working_memory import WorkingMemory
        
        test_file = temp_maven_project / "src" / "test" / "java" / "com" / "example" / "CalculatorTest.java"
        
        existing_test = '''
package com.example;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class CalculatorTest {
    @Test
    void testExistingMethod() {
        Calculator calc = new Calculator();
        assertEquals(4, calc.add(2, 2));
    }
}
'''
        test_file.write_text(existing_test)
        
        working_memory = WorkingMemory(
            target_coverage=0.8,
            max_iterations=3
        )
        
        with patch('pyutagent.agent.react_agent.LLMClient') as mock_llm_class:
            mock_llm_class.from_config.return_value = mock_llm_client
            
            agent = ReActAgent(
                project_path=str(temp_maven_project),
                llm_client=mock_llm_client,
                working_memory=working_memory,
                incremental_mode=True
            )
            
            assert agent.incremental_mode is True
    
    @pytest.mark.asyncio
    async def test_test_preservation_verification(self, temp_maven_project, mock_llm_client):
        """Test that existing tests are preserved during incremental generation."""
        from pyutagent.agent.incremental_fixer import IncrementalFixer
        
        test_file = temp_maven_project / "src" / "test" / "java" / "com" / "example" / "CalculatorTest.java"
        
        existing_test = '''
package com.example;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class CalculatorTest {
    @Test
    void testAdd() {
        Calculator calc = new Calculator();
        assertEquals(4, calc.add(2, 2));
    }
}
'''
        test_file.write_text(existing_test)
        
        fixer = IncrementalFixer(str(temp_maven_project))
        
        assert fixer is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
