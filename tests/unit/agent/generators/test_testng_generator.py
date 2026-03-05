"""Tests for TestNG test generator."""

import pytest
import asyncio
from pathlib import Path
import tempfile
import shutil

from pyutagent.agent.generators.testng_generator import TestNGTestGenerator
from pyutagent.core.project_config import MockFramework


class TestTestNGTestGenerator:
    """Test cases for TestNGTestGenerator."""
    
    @pytest.fixture
    def temp_project(self):
        """Create a temporary project directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def generator(self, temp_project):
        """Create a TestNGTestGenerator instance."""
        return TestNGTestGenerator(temp_project, MockFramework.MOCKITO)
    
    @pytest.mark.asyncio
    async def test_generate_initial_test_basic(self, generator):
        """Test generating basic TestNG test."""
        class_info = {
            'package': 'com.example',
            'name': 'PaymentService',
            'methods': [
                {
                    'name': 'processPayment',
                    'return_type': 'boolean',
                    'parameters': [
                        {'name': 'amount', 'type': 'BigDecimal'},
                        {'name': 'currency', 'type': 'String'}
                    ],
                    'line_number': 10,
                    'end_line': 25,
                    'throws_exceptions': []
                }
            ],
            'dependencies': [
                {'name': 'paymentGateway', 'type': 'PaymentGateway'}
            ],
            'fields': []
        }
        
        test_code = await generator.generate_initial_test(class_info)
        
        # Verify TestNG annotations
        assert '@Test' in test_code
        assert '@BeforeMethod' in test_code
        assert '@AfterMethod' in test_code
        
        # Verify imports
        assert 'import org.testng.annotations' in test_code
        assert 'import static org.testng.Assert' in test_code
        
        # Verify class name
        assert 'public class PaymentServiceTest' in test_code
        
        # Verify test method
        assert 'testProcessPayment' in test_code
    
    @pytest.mark.asyncio
    async def test_generate_initial_test_with_mockito(self, generator):
        """Test generating TestNG test with Mockito."""
        class_info = {
            'package': 'com.example',
            'name': 'OrderService',
            'methods': [
                {
                    'name': 'createOrder',
                    'return_type': 'Order',
                    'parameters': [],
                    'line_number': 15,
                    'end_line': 30,
                    'throws_exceptions': []
                }
            ],
            'dependencies': [
                {'name': 'orderRepository', 'type': 'OrderRepository'},
                {'name': 'emailService', 'type': 'EmailService'}
            ],
            'fields': []
        }
        
        test_code = await generator.generate_initial_test(class_info)
        
        # Verify Mockito imports
        assert 'import org.mockito' in test_code
        assert '@Mock' in test_code
        assert 'mock(' in test_code
        
        # Verify mock initialization in setup
        assert '@BeforeMethod' in test_code
        assert 'setUp()' in test_code
    
    @pytest.mark.asyncio
    async def test_generate_parameterized_test(self, generator):
        """Test generating parameterized test with DataProvider."""
        class_info = {
            'package': 'com.example',
            'name': 'CalculatorService',
            'methods': [
                {
                    'name': 'add',
                    'return_type': 'int',
                    'parameters': [
                        {'name': 'a', 'type': 'int'},
                        {'name': 'b', 'type': 'int'}
                    ],
                    'line_number': 10,
                    'end_line': 15,
                    'throws_exceptions': []
                }
            ],
            'dependencies': [],
            'fields': []
        }
        
        test_code = await generator.generate_initial_test(class_info)
        
        # Verify DataProvider
        assert '@DataProvider' in test_code
        assert 'Object[][]' in test_code
        
        # Verify parameterized test
        assert 'dataProvider' in test_code
    
    @pytest.mark.asyncio
    async def test_generate_exception_test(self, generator):
        """Test generating exception test."""
        class_info = {
            'package': 'com.example',
            'name': 'ValidationService',
            'methods': [
                {
                    'name': 'validateEmail',
                    'return_type': 'boolean',
                    'parameters': [
                        {'name': 'email', 'type': 'String'}
                    ],
                    'line_number': 20,
                    'end_line': 35,
                    'throws_exceptions': [
                        {'name': 'IllegalArgumentException', 'full_name': 'java.lang.IllegalArgumentException'}
                    ]
                }
            ],
            'dependencies': [],
            'fields': []
        }
        
        test_code = await generator.generate_initial_test(class_info)
        
        # Verify exception test
        assert '@Test(expectedExceptions' in test_code
        assert 'IllegalArgumentException' in test_code
    
    @pytest.mark.asyncio
    async def test_generate_additional_tests(self, generator):
        """Test generating additional tests for uncovered lines."""
        class_info = {
            'package': 'com.example',
            'name': 'UserService',
            'methods': [
                {
                    'name': 'getUserById',
                    'return_type': 'User',
                    'parameters': [
                        {'name': 'id', 'type': 'Long'}
                    ],
                    'line_number': 25,
                    'end_line': 40,
                    'throws_exceptions': []
                }
            ],
            'dependencies': [],
            'fields': []
        }
        
        uncovered_lines = [28, 30, 35]
        
        additional_tests = await generator.generate_additional_tests(
            class_info,
            uncovered_lines
        )
        
        # Verify additional tests are generated
        assert '@Test' in additional_tests
        assert 'coverage' in additional_tests.lower() or 'uncovered' in additional_tests.lower()
    
    def test_generate_test_method_name(self, generator):
        """Test test method name generation."""
        assert generator._generate_test_method_name('calculate') == 'testCalculate'
        assert generator._generate_test_method_name('processPayment') == 'testProcessPayment'
        assert generator._generate_test_method_name('') == 'testBasic'
    
    def test_get_default_value(self, generator):
        """Test default value generation for parameter types."""
        assert generator._get_default_value('int') == '0'
        assert generator._get_default_value('String') == '""'
        assert generator._get_default_value('boolean') == 'false'
        assert generator._get_default_value('double') == '0.0'
        assert generator._get_default_value('Unknown') == 'null'
    
    def test_generate_sample_test_data(self, generator):
        """Test sample test data generation."""
        parameters = [
            {'name': 'value', 'type': 'int'},
            {'name': 'name', 'type': 'String'}
        ]
        
        test_data = generator._generate_sample_test_data(parameters)
        
        assert len(test_data) == 3  # Should generate 3 rows
        assert len(test_data[0]) == 2  # Each row should have 2 values
        
        # Verify values vary across rows
        assert test_data[0][0] != test_data[1][0]  # int values should differ
        assert test_data[0][1] != test_data[1][1]  # String values should differ
    
    def test_build_imports(self, generator):
        """Test import statement generation."""
        imports = generator._build_imports()
        
        assert 'import org.testng.annotations' in imports
        assert 'import static org.testng.Assert' in imports
        assert 'import org.mockito' in imports  # Mockito is default
        assert 'import java.util' in imports
        assert 'import java.io' in imports
    
    @pytest.mark.asyncio
    async def test_fallback_to_basic_template(self, generator):
        """Test fallback to basic template when generation fails."""
        # Empty class info should still generate valid TestNG template
        class_info = {
            'package': '',
            'name': 'EmptyClass',
            'methods': [],
            'dependencies': [],
            'fields': []
        }
        
        test_code = await generator.generate_initial_test(class_info)
        
        # Should generate valid TestNG structure
        assert 'public class EmptyClassTest' in test_code
        assert '@BeforeMethod' in test_code
        assert '@AfterMethod' in test_code
        assert '@Test' in test_code


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
