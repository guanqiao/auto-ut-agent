"""Tests for working memory."""

import pytest
from datetime import datetime


class TestWorkingMemory:
    """Test suite for WorkingMemory."""
    
    def test_init_default_values(self):
        """Test initialization with default values."""
        from pyutagent.memory.working_memory import WorkingMemory
        
        mem = WorkingMemory()
        
        assert mem.current_file is None
        assert mem.current_method is None
        assert mem.iteration_count == 0
        assert mem.max_iterations == 10
        assert mem.target_coverage == 0.8
        assert mem.current_coverage == 0.0
        assert mem.is_paused is False
        assert mem.processed_files == []
        assert mem.failed_tests == []
        assert mem.generated_tests == []
    
    def test_init_custom_values(self):
        """Test initialization with custom values."""
        from pyutagent.memory.working_memory import WorkingMemory
        
        mem = WorkingMemory(
            current_file="UserService.java",
            current_method="getUser",
            iteration_count=3,
            max_iterations=15,
            target_coverage=0.9
        )
        
        assert mem.current_file == "UserService.java"
        assert mem.current_method == "getUser"
        assert mem.iteration_count == 3
        assert mem.max_iterations == 15
        assert mem.target_coverage == 0.9
    
    def test_update_coverage(self):
        """Test updating coverage."""
        from pyutagent.memory.working_memory import WorkingMemory
        
        mem = WorkingMemory()
        
        mem.update_coverage(0.5)
        assert mem.current_coverage == 0.5
        assert len(mem.coverage_history) == 1
        
        mem.update_coverage(0.7)
        assert mem.current_coverage == 0.7
        assert len(mem.coverage_history) == 2
    
    def test_add_processed_file(self):
        """Test adding processed file."""
        from pyutagent.memory.working_memory import WorkingMemory
        
        mem = WorkingMemory()
        
        mem.add_processed_file("UserService.java")
        assert "UserService.java" in mem.processed_files
        
        # Duplicate should not be added
        mem.add_processed_file("UserService.java")
        assert mem.processed_files.count("UserService.java") == 1
    
    def test_add_failed_test(self):
        """Test adding failed test."""
        from pyutagent.memory.working_memory import WorkingMemory
        
        mem = WorkingMemory()
        
        mem.add_failed_test("testGetUser", "NullPointerException")
        assert len(mem.failed_tests) == 1
        assert mem.failed_tests[0]["test_name"] == "testGetUser"
        assert mem.failed_tests[0]["error"] == "NullPointerException"
    
    def test_add_generated_test(self):
        """Test adding generated test."""
        from pyutagent.memory.working_memory import WorkingMemory
        
        mem = WorkingMemory()
        
        test_code = "public void testAdd() { assertEquals(2, calc.add(1,1)); }"
        mem.add_generated_test("CalculatorTest.java", "testAdd", test_code)
        
        assert len(mem.generated_tests) == 1
        assert mem.generated_tests[0]["file"] == "CalculatorTest.java"
        assert mem.generated_tests[0]["method"] == "testAdd"
        assert mem.generated_tests[0]["code"] == test_code
    
    def test_pause_resume(self):
        """Test pause and resume."""
        from pyutagent.memory.working_memory import WorkingMemory
        
        mem = WorkingMemory()
        
        assert mem.is_paused is False
        
        mem.pause()
        assert mem.is_paused is True
        
        mem.resume()
        assert mem.is_paused is False
    
    def test_increment_iteration(self):
        """Test incrementing iteration count."""
        from pyutagent.memory.working_memory import WorkingMemory
        
        mem = WorkingMemory()
        
        assert mem.iteration_count == 0
        
        mem.increment_iteration()
        assert mem.iteration_count == 1
        
        mem.increment_iteration()
        assert mem.iteration_count == 2
    
    def test_is_complete(self):
        """Test completion check."""
        from pyutagent.memory.working_memory import WorkingMemory
        
        mem = WorkingMemory(target_coverage=0.8, max_iterations=5)
        
        # Not complete initially
        assert mem.is_complete() is False
        
        # Complete when target coverage reached
        mem.update_coverage(0.85)
        assert mem.is_complete() is True
        
        # Reset and check max iterations
        mem = WorkingMemory(target_coverage=0.8, max_iterations=5)
        mem.iteration_count = 5
        assert mem.is_complete() is True
    
    def test_to_dict(self):
        """Test serialization to dict."""
        from pyutagent.memory.working_memory import WorkingMemory
        
        mem = WorkingMemory(
            current_file="Test.java",
            iteration_count=2,
            current_coverage=0.6
        )
        
        data = mem.to_dict()
        
        assert data["current_file"] == "Test.java"
        assert data["iteration_count"] == 2
        assert data["current_coverage"] == 0.6
        assert "coverage_history" in data
    
    def test_from_dict(self):
        """Test deserialization from dict."""
        from pyutagent.memory.working_memory import WorkingMemory
        
        data = {
            "current_file": "Test.java",
            "current_method": "testMethod",
            "iteration_count": 3,
            "max_iterations": 15,
            "target_coverage": 0.85,
            "current_coverage": 0.7,
            "coverage_history": [0.5, 0.6, 0.7],
            "is_paused": True,
            "processed_files": ["A.java", "B.java"],
            "failed_tests": [],
            "generated_tests": []
        }
        
        mem = WorkingMemory.from_dict(data)
        
        assert mem.current_file == "Test.java"
        assert mem.current_method == "testMethod"
        assert mem.iteration_count == 3
        assert mem.is_paused is True
        assert mem.processed_files == ["A.java", "B.java"]
