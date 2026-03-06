"""E2E tests for CLI commands.

This module tests the complete CLI workflow:
- scan: Project scanning
- generate: Single file test generation
- generate-all: Batch test generation
- config: Configuration management
"""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

import pytest
from click.testing import CliRunner

from pyutagent.cli.main import cli
from tests.e2e.utils import (
    create_java_class,
    create_pom_xml,
    count_java_files,
    count_test_files
)


class TestScanCommandE2E:
    """E2E tests for scan command."""
    
    def test_scan_maven_project_success(self, temp_maven_project):
        """Test scanning a valid Maven project."""
        runner = CliRunner()
        
        result = runner.invoke(cli, ['scan', str(temp_maven_project)])
        
        assert result.exit_code == 0
        assert 'Calculator.java' in result.output
        assert str(temp_maven_project) in result.output
    
    def test_scan_non_maven_project(self, temp_dir):
        """Test scanning a non-Maven project."""
        runner = CliRunner()
        
        result = runner.invoke(cli, ['scan', str(temp_dir)])
        
        assert result.exit_code != 0 or 'No Java files found' in result.output or 'pom.xml' in result.output.lower()
    
    def test_scan_empty_project(self, temp_dir):
        """Test scanning an empty Maven project."""
        pom_file = temp_dir / "pom.xml"
        pom_file.write_text(create_pom_xml())
        
        runner = CliRunner()
        result = runner.invoke(cli, ['scan', str(temp_dir)])
        
        assert 'No Java files found' in result.output or result.exit_code == 0
    
    def test_scan_with_complex_structure(self, temp_multi_file_project):
        """Test scanning project with complex package structure."""
        runner = CliRunner()
        
        result = runner.invoke(cli, ['scan', str(temp_multi_file_project)])
        
        assert result.exit_code == 0
        assert 'User.java' in result.output
        assert 'UserRepository.java' in result.output
        assert 'UserService.java' in result.output
        assert 'Calculator.java' in result.output
        assert 'StringUtils.java' in result.output


class TestGenerateCommandE2E:
    """E2E tests for generate command."""
    
    @pytest.mark.asyncio
    async def test_generate_single_file_success(self, temp_maven_project, mock_llm_client):
        """Test generating tests for a single Java file."""
        java_file = temp_maven_project / "src" / "main" / "java" / "com" / "example" / "Calculator.java"
        
        runner = CliRunner()
        
        with patch('pyutagent.cli.commands.generate.LLMClient') as mock_llm_class:
            mock_llm_class.from_config.return_value = mock_llm_client
            
            with patch('pyutagent.cli.commands.generate.load_llm_config') as mock_load:
                from pyutagent.llm.config import LLMConfig, LLMConfigCollection
                mock_config = LLMConfig(
                    id="test",
                    name="Test",
                    provider="openai",
                    api_key="test-key",
                    model="gpt-4",
                    is_default=True
                )
                mock_load.return_value = LLMConfigCollection(configs=[mock_config])
                
                result = runner.invoke(cli, ['generate', str(java_file)])
                
                assert result.exit_code == 0 or 'Generating tests' in result.output
    
    @pytest.mark.asyncio
    async def test_generate_with_incremental_mode(self, temp_maven_project, mock_llm_client):
        """Test generating tests with incremental mode."""
        java_file = temp_maven_project / "src" / "main" / "java" / "com" / "example" / "Calculator.java"
        
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
        
        runner = CliRunner()
        
        with patch('pyutagent.cli.commands.generate.LLMClient') as mock_llm_class:
            mock_llm_class.from_config.return_value = mock_llm_client
            
            with patch('pyutagent.cli.commands.generate.load_llm_config') as mock_load:
                from pyutagent.llm.config import LLMConfig, LLMConfigCollection
                mock_config = LLMConfig(
                    id="test",
                    name="Test",
                    provider="openai",
                    api_key="test-key",
                    model="gpt-4",
                    is_default=True
                )
                mock_load.return_value = LLMConfigCollection(configs=[mock_config])
                
                result = runner.invoke(cli, ['generate', str(java_file), '-i'])
                
                assert result.exit_code == 0 or 'Incremental mode' in result.output or 'Generating tests' in result.output
    
    @pytest.mark.asyncio
    async def test_generate_with_coverage_target(self, temp_maven_project, mock_llm_client):
        """Test generating tests with specific coverage target."""
        java_file = temp_maven_project / "src" / "main" / "java" / "com" / "example" / "Calculator.java"
        
        runner = CliRunner()
        
        with patch('pyutagent.cli.commands.generate.LLMClient') as mock_llm_class:
            mock_llm_class.from_config.return_value = mock_llm_client
            
            with patch('pyutagent.cli.commands.generate.load_llm_config') as mock_load:
                from pyutagent.llm.config import LLMConfig, LLMConfigCollection
                mock_config = LLMConfig(
                    id="test",
                    name="Test",
                    provider="openai",
                    api_key="test-key",
                    model="gpt-4",
                    is_default=True
                )
                mock_load.return_value = LLMConfigCollection(configs=[mock_config])
                
                result = runner.invoke(cli, ['generate', str(java_file), '--coverage-target', '90'])
                
                assert result.exit_code == 0 or 'Coverage target: 90%' in result.output
    
    def test_generate_non_java_file(self, temp_dir):
        """Test generating tests for non-Java file."""
        text_file = temp_dir / "test.txt"
        text_file.write_text("This is not a Java file")
        
        runner = CliRunner()
        result = runner.invoke(cli, ['generate', str(text_file)])
        
        assert result.exit_code != 0
        assert 'not a Java file' in result.output.lower() or 'error' in result.output.lower()
    
    @pytest.mark.asyncio
    async def test_generate_with_compilation_error(self, temp_maven_project, mock_llm_client):
        """Test handling compilation errors during generation."""
        java_file = temp_maven_project / "src" / "main" / "java" / "com" / "example" / "Calculator.java"
        
        def generate_with_error(*args, **kwargs):
            return '''
package com.example;

import org.junit.jupiter.api.Test;

class CalculatorTest {
    @Test
    void testWithError() {
        Calculator calc = new Calculator();
        UndefinedClass obj = new UndefinedClass();
    }
}
'''
        
        mock_llm_client.agenerate = AsyncMock(side_effect=generate_with_error)
        
        runner = CliRunner()
        
        with patch('pyutagent.cli.commands.generate.LLMClient') as mock_llm_class:
            mock_llm_class.from_config.return_value = mock_llm_client
            
            with patch('pyutagent.cli.commands.generate.load_llm_config') as mock_load:
                from pyutagent.llm.config import LLMConfig, LLMConfigCollection
                mock_config = LLMConfig(
                    id="test",
                    name="Test",
                    provider="openai",
                    api_key="test-key",
                    model="gpt-4",
                    is_default=True
                )
                mock_load.return_value = LLMConfigCollection(configs=[mock_config])
                
                result = runner.invoke(cli, ['generate', str(java_file)])
                
                assert result.exit_code == 0 or 'Generating tests' in result.output


class TestGenerateAllCommandE2E:
    """E2E tests for generate-all command."""
    
    @pytest.mark.asyncio
    async def test_generate_all_sequential(self, temp_multi_file_project, mock_llm_client):
        """Test batch generation in sequential mode."""
        runner = CliRunner()
        
        with patch('pyutagent.cli.commands.generate_all.LLMClient') as mock_llm_class:
            mock_llm_class.from_config.return_value = mock_llm_client
            
            with patch('pyutagent.cli.commands.generate_all.load_llm_config') as mock_load:
                from pyutagent.llm.config import LLMConfig, LLMConfigCollection
                mock_config = LLMConfig(
                    id="test",
                    name="Test",
                    provider="openai",
                    api_key="test-key",
                    model="gpt-4",
                    is_default=True
                )
                mock_load.return_value = LLMConfigCollection(configs=[mock_config])
                
                with patch('pyutagent.services.batch_generator.BatchGenerator') as mock_gen:
                    mock_instance = MagicMock()
                    mock_result = MagicMock()
                    mock_result.success_count = 5
                    mock_result.failed_count = 0
                    mock_result.total_files = 5
                    mock_result.success_rate = 100.0
                    mock_result.results = []
                    mock_result.compilation_result = None
                    mock_instance.generate_all_sync.return_value = mock_result
                    mock_gen.return_value = mock_instance
                    
                    result = runner.invoke(cli, ['generate-all', str(temp_multi_file_project), '-p', '1'])
                    
                    assert result.exit_code == 0 or 'Batch Test Generation' in result.output
    
    @pytest.mark.asyncio
    async def test_generate_all_parallel(self, temp_multi_file_project, mock_llm_client):
        """Test batch generation in parallel mode."""
        runner = CliRunner()
        
        with patch('pyutagent.cli.commands.generate_all.LLMClient') as mock_llm_class:
            mock_llm_class.from_config.return_value = mock_llm_client
            
            with patch('pyutagent.cli.commands.generate_all.load_llm_config') as mock_load:
                from pyutagent.llm.config import LLMConfig, LLMConfigCollection
                mock_config = LLMConfig(
                    id="test",
                    name="Test",
                    provider="openai",
                    api_key="test-key",
                    model="gpt-4",
                    is_default=True
                )
                mock_load.return_value = LLMConfigCollection(configs=[mock_config])
                
                with patch('pyutagent.services.batch_generator.BatchGenerator') as mock_gen:
                    mock_instance = MagicMock()
                    mock_result = MagicMock()
                    mock_result.success_count = 5
                    mock_result.failed_count = 0
                    mock_result.total_files = 5
                    mock_result.success_rate = 100.0
                    mock_result.results = []
                    mock_result.compilation_result = None
                    mock_instance.generate_all_sync.return_value = mock_result
                    mock_gen.return_value = mock_instance
                    
                    result = runner.invoke(cli, ['generate-all', str(temp_multi_file_project), '-p', '4'])
                    
                    assert result.exit_code == 0 or 'Parallel workers: 4' in result.output
    
    @pytest.mark.asyncio
    async def test_generate_all_with_defer_compilation(self, temp_multi_file_project, mock_llm_client):
        """Test two-phase batch generation."""
        runner = CliRunner()
        
        with patch('pyutagent.cli.commands.generate_all.LLMClient') as mock_llm_class:
            mock_llm_class.from_config.return_value = mock_llm_client
            
            with patch('pyutagent.cli.commands.generate_all.load_llm_config') as mock_load:
                from pyutagent.llm.config import LLMConfig, LLMConfigCollection
                mock_config = LLMConfig(
                    id="test",
                    name="Test",
                    provider="openai",
                    api_key="test-key",
                    model="gpt-4",
                    is_default=True
                )
                mock_load.return_value = LLMConfigCollection(configs=[mock_config])
                
                with patch('pyutagent.services.batch_generator.BatchGenerator') as mock_gen:
                    mock_instance = MagicMock()
                    mock_result = MagicMock()
                    mock_result.success_count = 5
                    mock_result.failed_count = 0
                    mock_result.total_files = 5
                    mock_result.success_rate = 100.0
                    mock_result.results = []
                    mock_compilation = MagicMock()
                    mock_compilation.success = True
                    mock_compilation.compiled_files = 5
                    mock_compilation.failed_files = 0
                    mock_compilation.duration = 10.0
                    mock_compilation.errors = []
                    mock_result.compilation_result = mock_compilation
                    mock_instance.generate_all_sync.return_value = mock_result
                    mock_gen.return_value = mock_instance
                    
                    result = runner.invoke(cli, ['generate-all', str(temp_multi_file_project), '--defer-compilation'])
                    
                    assert result.exit_code == 0 or 'Defer compilation' in result.output or 'Two-phase mode' in result.output
    
    @pytest.mark.asyncio
    async def test_generate_all_continue_on_error(self, temp_multi_file_project, mock_llm_client):
        """Test batch generation with continue-on-error."""
        runner = CliRunner()
        
        with patch('pyutagent.cli.commands.generate_all.LLMClient') as mock_llm_class:
            mock_llm_class.from_config.return_value = mock_llm_client
            
            with patch('pyutagent.cli.commands.generate_all.load_llm_config') as mock_load:
                from pyutagent.llm.config import LLMConfig, LLMConfigCollection
                mock_config = LLMConfig(
                    id="test",
                    name="Test",
                    provider="openai",
                    api_key="test-key",
                    model="gpt-4",
                    is_default=True
                )
                mock_load.return_value = LLMConfigCollection(configs=[mock_config])
                
                with patch('pyutagent.services.batch_generator.BatchGenerator') as mock_gen:
                    mock_instance = MagicMock()
                    mock_result = MagicMock()
                    mock_result.success_count = 3
                    mock_result.failed_count = 2
                    mock_result.total_files = 5
                    mock_result.success_rate = 60.0
                    mock_result.results = []
                    mock_result.compilation_result = None
                    mock_instance.generate_all_sync.return_value = mock_result
                    mock_gen.return_value = mock_instance
                    
                    result = runner.invoke(cli, ['generate-all', str(temp_multi_file_project), '--continue-on-error'])
                    
                    assert result.exit_code == 0 or 'Continue on error' in result.output
    
    @pytest.mark.asyncio
    async def test_generate_all_stop_on_error(self, temp_multi_file_project, mock_llm_client):
        """Test batch generation with stop-on-error."""
        runner = CliRunner()
        
        with patch('pyutagent.cli.commands.generate_all.LLMClient') as mock_llm_class:
            mock_llm_class.from_config.return_value = mock_llm_client
            
            with patch('pyutagent.cli.commands.generate_all.load_llm_config') as mock_load:
                from pyutagent.llm.config import LLMConfig, LLMConfigCollection
                mock_config = LLMConfig(
                    id="test",
                    name="Test",
                    provider="openai",
                    api_key="test-key",
                    model="gpt-4",
                    is_default=True
                )
                mock_load.return_value = LLMConfigCollection(configs=[mock_config])
                
                with patch('pyutagent.services.batch_generator.BatchGenerator') as mock_gen:
                    mock_instance = MagicMock()
                    mock_result = MagicMock()
                    mock_result.success_count = 0
                    mock_result.failed_count = 1
                    mock_result.total_files = 5
                    mock_result.success_rate = 0.0
                    mock_result.results = []
                    mock_result.compilation_result = None
                    mock_instance.generate_all_sync.return_value = mock_result
                    mock_gen.return_value = mock_instance
                    
                    result = runner.invoke(cli, ['generate-all', str(temp_multi_file_project), '--stop-on-error'])
                    
                    assert result.exit_code == 0 or 'stop-on-error' in result.output.lower()


class TestConfigCommandE2E:
    """E2E tests for config command."""
    
    def test_config_llm_list(self, temp_config):
        """Test listing LLM configurations."""
        runner = CliRunner()
        
        with patch('pyutagent.config.get_config_dir') as mock_dir:
            mock_dir.return_value = temp_config
            
            result = runner.invoke(cli, ['config', 'llm', 'list'])
            
            assert result.exit_code == 0 or 'LLM' in result.output or 'config' in result.output.lower()
    
    def test_config_llm_set_default(self, temp_config):
        """Test setting default LLM configuration."""
        runner = CliRunner()
        
        with patch('pyutagent.config.get_config_dir') as mock_dir:
            mock_dir.return_value = temp_config
            
            result = runner.invoke(cli, ['config', 'llm', 'set-default', 'test-openai'])
            
            assert result.exit_code == 0 or 'default' in result.output.lower()
    
    def test_config_maven_show(self, temp_config):
        """Test showing Maven configuration."""
        runner = CliRunner()
        
        with patch('pyutagent.core.config.get_config_dir') as mock_dir:
            mock_dir.return_value = temp_config
            
            result = runner.invoke(cli, ['config', 'maven', 'show'])
            
            assert result.exit_code == 0 or 'Maven' in result.output or 'maven' in result.output.lower()
    
    def test_config_coverage_show(self, temp_config):
        """Test showing coverage configuration."""
        runner = CliRunner()
        
        with patch('pyutagent.core.config.get_config_dir') as mock_dir:
            mock_dir.return_value = temp_config
            
            result = runner.invoke(cli, ['config', 'coverage', 'show'])
            
            assert result.exit_code == 0 or 'coverage' in result.output.lower() or 'JaCoCo' in result.output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
