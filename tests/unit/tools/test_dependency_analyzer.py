"""Unit tests for DependencyAnalyzer."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from pyutagent.tools.dependency_analyzer import DependencyAnalyzer


class TestDependencyAnalyzer:
    """Test cases for DependencyAnalyzer."""
    
    def test_init(self):
        """Test DependencyAnalyzer initialization."""
        analyzer = DependencyAnalyzer()
        
        assert analyzer.llm_client is None
        assert analyzer.prompt_builder is None
    
    def test_init_with_clients(self):
        """Test DependencyAnalyzer initialization with LLM client."""
        mock_client = MagicMock()
        mock_builder = MagicMock()
        
        analyzer = DependencyAnalyzer(mock_client, mock_builder)
        
        assert analyzer.llm_client == mock_client
        assert analyzer.prompt_builder == mock_builder
    
    @pytest.mark.asyncio
    async def test_analyze_missing_dependencies_no_llm(self):
        """Test analyzing dependencies without LLM (rule-based)."""
        analyzer = DependencyAnalyzer()
        
        compiler_output = """
        [ERROR] /path/to/Test.java:[10,5] cannot find symbol
        [ERROR]   symbol:   class Test
        [ERROR]   location: package org.junit.jupiter.api
        [ERROR] package org.junit.jupiter.api does not exist
        """
        
        pom_content = "<dependencies></dependencies>"
        
        result = await analyzer.analyze_missing_dependencies(compiler_output, pom_content)
        
        assert "missing_dependencies" in result
        assert "confidence" in result
        assert len(result["missing_dependencies"]) > 0
        
        dep = result["missing_dependencies"][0]
        assert dep["group_id"] == "org.junit.jupiter"
        assert dep["artifact_id"] == "junit-jupiter"
        assert dep["scope"] == "test"
    
    @pytest.mark.asyncio
    async def test_analyze_missing_dependencies_with_llm(self):
        """Test analyzing dependencies with LLM."""
        mock_client = AsyncMock()
        mock_builder = MagicMock()
        
        llm_response = """{
            "missing_dependencies": [
                {
                    "group_id": "org.junit.jupiter",
                    "artifact_id": "junit-jupiter",
                    "version": "5.10.0",
                    "scope": "test",
                    "reason": "JUnit 5 testing framework"
                }
            ],
            "confidence": 0.95,
            "analysis": "Found JUnit 5 dependency missing",
            "suggested_fixes": []
        }"""
        
        mock_client.agenerate = AsyncMock(return_value=llm_response)
        mock_builder.build_dependency_analysis_prompt = MagicMock(return_value="test prompt")
        
        analyzer = DependencyAnalyzer(mock_client, mock_builder)
        
        compiler_output = "package org.junit.jupiter.api does not exist"
        pom_content = "<dependencies></dependencies>"
        
        result = await analyzer.analyze_missing_dependencies(compiler_output, pom_content)
        
        assert result["confidence"] == 0.95
        assert len(result["missing_dependencies"]) == 1
        assert result["missing_dependencies"][0]["group_id"] == "org.junit.jupiter"
    
    def test_extract_missing_packages(self):
        """Test extracting missing packages from compiler output."""
        analyzer = DependencyAnalyzer()
        
        compiler_output = """
        [ERROR] package org.junit.jupiter.api does not exist
        [ERROR] package org.mockito does not exist
        [ERROR] import org.assertj.core.api.Assertions;
        """
        
        packages = analyzer._extract_missing_packages(compiler_output)
        
        assert "org.junit.jupiter.api" in packages
        assert "org.mockito" in packages
        assert "org.assertj.core.api.Assertions" in packages
    
    def test_map_package_to_dependency_junit(self):
        """Test mapping JUnit package to dependency."""
        analyzer = DependencyAnalyzer()
        
        dep = analyzer._map_package_to_dependency("org.junit.jupiter.api.Test")
        
        assert dep is not None
        assert dep["group_id"] == "org.junit.jupiter"
        assert dep["artifact_id"] == "junit-jupiter"
        assert dep["scope"] == "test"
    
    def test_map_package_to_dependency_mockito(self):
        """Test mapping Mockito package to dependency."""
        analyzer = DependencyAnalyzer()
        
        dep = analyzer._map_package_to_dependency("org.mockito.Mock")
        
        assert dep is not None
        assert dep["group_id"] == "org.mockito"
        assert dep["artifact_id"] == "mockito-core"
        assert dep["scope"] == "test"
    
    def test_map_package_to_dependency_unknown(self):
        """Test mapping unknown package to dependency."""
        analyzer = DependencyAnalyzer()
        
        dep = analyzer._map_package_to_dependency("com.unknown.package")
        
        assert dep is None
    
    def test_build_dependency_analysis_prompt(self):
        """Test building dependency analysis prompt."""
        analyzer = DependencyAnalyzer()
        
        compiler_output = "package org.junit.jupiter.api does not exist"
        pom_content = "<dependencies></dependencies>"
        
        prompt = analyzer.build_dependency_analysis_prompt(compiler_output, pom_content)
        
        assert "Maven dependency expert" in prompt
        assert compiler_output in prompt
        assert pom_content in prompt
        assert "JSON format" in prompt
    
    def test_parse_llm_response_valid_json(self):
        """Test parsing valid LLM JSON response."""
        analyzer = DependencyAnalyzer()
        
        response = """Some text before
        {
            "missing_dependencies": [
                {
                    "group_id": "org.junit.jupiter",
                    "artifact_id": "junit-jupiter",
                    "version": "5.10.0",
                    "scope": "test",
                    "reason": "JUnit 5"
                }
            ],
            "confidence": 0.9,
            "analysis": "Test analysis"
        }
        Some text after"""
        
        result = analyzer._parse_llm_response(response)
        
        assert result["confidence"] == 0.9
        assert len(result["missing_dependencies"]) == 1
        assert result["analysis"] == "Test analysis"
    
    def test_parse_llm_response_invalid_json(self):
        """Test parsing invalid LLM JSON response."""
        analyzer = DependencyAnalyzer()
        
        response = "This is not JSON"
        
        result = analyzer._parse_llm_response(response)
        
        assert result["confidence"] == 0.3
        assert len(result["missing_dependencies"]) == 0
    
    def test_validate_dependency_valid(self):
        """Test validating a valid dependency."""
        analyzer = DependencyAnalyzer()
        
        dependency = {
            "group_id": "org.junit.jupiter",
            "artifact_id": "junit-jupiter",
            "version": "5.10.0",
            "scope": "test"
        }
        
        is_valid, errors = analyzer.validate_dependency(dependency)
        
        assert is_valid is True
        assert len(errors) == 0
    
    def test_validate_dependency_missing_field(self):
        """Test validating a dependency with missing field."""
        analyzer = DependencyAnalyzer()
        
        dependency = {
            "group_id": "org.junit.jupiter",
            "artifact_id": "junit-jupiter"
        }
        
        is_valid, errors = analyzer.validate_dependency(dependency)
        
        assert is_valid is False
        assert any("Missing or empty required field" in e for e in errors)
    
    def test_validate_dependency_invalid_scope(self):
        """Test validating a dependency with invalid scope."""
        analyzer = DependencyAnalyzer()
        
        dependency = {
            "group_id": "org.junit.jupiter",
            "artifact_id": "junit-jupiter",
            "version": "5.10.0",
            "scope": "invalid_scope"
        }
        
        is_valid, errors = analyzer.validate_dependency(dependency)
        
        assert is_valid is False
        assert any("Invalid scope" in e for e in errors)
    
    def test_deduplicate_dependencies(self):
        """Test deduplicating dependencies."""
        analyzer = DependencyAnalyzer()
        
        dependencies = [
            {
                "group_id": "org.junit.jupiter",
                "artifact_id": "junit-jupiter",
                "version": "5.10.0"
            },
            {
                "group_id": "org.junit.jupiter",
                "artifact_id": "junit-jupiter",
                "version": "5.9.0"
            },
            {
                "group_id": "org.mockito",
                "artifact_id": "mockito-core",
                "version": "5.8.0"
            }
        ]
        
        unique_deps = analyzer.deduplicate_dependencies(dependencies)
        
        assert len(unique_deps) == 2
        assert unique_deps[0]["group_id"] == "org.junit.jupiter"
        assert unique_deps[1]["group_id"] == "org.mockito"
    
    @pytest.mark.asyncio
    async def test_llm_analysis_fallback_to_rule_based(self):
        """Test LLM analysis falling back to rule-based on error."""
        mock_client = AsyncMock()
        mock_builder = MagicMock()
        
        mock_client.agenerate = AsyncMock(side_effect=Exception("LLM error"))
        mock_builder.build_dependency_analysis_prompt = MagicMock(return_value="test prompt")
        
        analyzer = DependencyAnalyzer(mock_client, mock_builder)
        
        compiler_output = "package org.junit.jupiter.api does not exist"
        pom_content = "<dependencies></dependencies>"
        
        result = await analyzer.analyze_missing_dependencies(compiler_output, pom_content)
        
        assert "missing_dependencies" in result
        assert len(result["missing_dependencies"]) > 0
