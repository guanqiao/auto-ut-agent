"""Tests for enhanced error classification with dependency detection."""

import pytest
from pyutagent.core.error_classification import (
    ErrorSubCategory,
    detect_missing_dependencies,
    detect_compilation_error_type,
    detect_maven_error_type,
    get_error_classification_service,
    get_detailed_error_info,
)


class TestDetectMissingDependencies:
    """Tests for detect_missing_dependencies function."""
    
    def test_detect_junit_missing(self):
        compiler_output = """
error: package org.junit.jupiter.api does not exist
import org.junit.jupiter.api.Test;
"""
        result = detect_missing_dependencies(compiler_output)
        
        assert result["sub_category"] == ErrorSubCategory.MISSING_DEPENDENCY
        assert "org.junit.jupiter.api" in result["missing_packages"]
        assert result["is_test_dependency"] is True
        assert "junit-jupiter" in result["suggested_dependencies"]
    
    def test_detect_mockito_missing(self):
        compiler_output = """
error: package org.mockito does not exist
import org.mockito.Mockito;
"""
        result = detect_missing_dependencies(compiler_output)
        
        assert result["sub_category"] == ErrorSubCategory.MISSING_DEPENDENCY
        assert "org.mockito" in result["missing_packages"]
        assert result["is_test_dependency"] is True
        assert "mockito-core" in result["suggested_dependencies"]
    
    def test_detect_assertj_missing(self):
        compiler_output = """
error: package org.assertj.core.api does not exist
import static org.assertj.core.api.Assertions.assertThat;
"""
        result = detect_missing_dependencies(compiler_output)
        
        assert result["sub_category"] == ErrorSubCategory.MISSING_DEPENDENCY
        assert "org.assertj.core.api" in result["missing_packages"]
        assert result["is_test_dependency"] is True
        assert "assertj-core" in result["suggested_dependencies"]
    
    def test_detect_multiple_missing_dependencies(self):
        compiler_output = """
error: package org.junit.jupiter.api does not exist
error: package org.mockito does not exist
error: package org.assertj.core.api does not exist
"""
        result = detect_missing_dependencies(compiler_output)
        
        assert result["sub_category"] == ErrorSubCategory.MISSING_DEPENDENCY
        assert len(result["missing_packages"]) == 3
        assert result["is_test_dependency"] is True
        assert len(result["suggested_dependencies"]) == 3
    
    def test_detect_non_test_dependency(self):
        compiler_output = """
error: package com.example.unknown does not exist
import com.example.unknown.SomeClass;
"""
        result = detect_missing_dependencies(compiler_output)
        
        assert result["sub_category"] == ErrorSubCategory.MISSING_DEPENDENCY
        assert "com.example.unknown" in result["missing_packages"]
        assert result["is_test_dependency"] is False
    
    def test_no_missing_dependencies(self):
        compiler_output = """
error: incompatible types: String cannot be converted to int
"""
        result = detect_missing_dependencies(compiler_output)
        
        assert result["sub_category"] == ErrorSubCategory.UNKNOWN
        assert len(result["missing_packages"]) == 0


class TestDetectCompilationErrorType:
    """Tests for detect_compilation_error_type function."""
    
    def test_detect_missing_dependency(self):
        compiler_output = "error: package org.junit does not exist"
        result = detect_compilation_error_type(compiler_output)
        assert result == ErrorSubCategory.MISSING_DEPENDENCY
    
    def test_detect_symbol_not_found(self):
        compiler_output = "error: cannot find symbol: class UnknownClass"
        result = detect_compilation_error_type(compiler_output)
        assert result == ErrorSubCategory.SYMBOL_NOT_FOUND
    
    def test_detect_incompatible_types(self):
        compiler_output = "error: incompatible types: String cannot be converted to int"
        result = detect_compilation_error_type(compiler_output)
        assert result == ErrorSubCategory.INCOMPATIBLE_TYPES
    
    def test_detect_syntax_error(self):
        compiler_output = "error: ';' expected"
        result = detect_compilation_error_type(compiler_output)
        assert result == ErrorSubCategory.SYNTAX_ERROR
    
    def test_detect_method_not_found(self):
        compiler_output = "error: method someMethod cannot be applied to given types"
        result = detect_compilation_error_type(compiler_output)
        assert result == ErrorSubCategory.METHOD_NOT_FOUND


class TestDetectMavenErrorType:
    """Tests for detect_maven_error_type function."""
    
    def test_detect_dependency_error(self):
        maven_output = "Failed to execute goal: could not resolve dependencies"
        result = detect_maven_error_type(maven_output)
        assert result == ErrorSubCategory.MAVEN_DEPENDENCY_ERROR
    
    def test_detect_network_error(self):
        maven_output = "Connection timeout while downloading artifact"
        result = detect_maven_error_type(maven_output)
        assert result == ErrorSubCategory.MAVEN_NETWORK_ERROR
    
    def test_detect_test_error(self):
        maven_output = "Failed to execute goal: test failed"
        result = detect_maven_error_type(maven_output)
        assert result == ErrorSubCategory.MAVEN_TEST_ERROR
    
    def test_detect_build_error(self):
        maven_output = "Failed to execute goal: build failed"
        result = detect_maven_error_type(maven_output)
        assert result == ErrorSubCategory.MAVEN_BUILD_ERROR


class TestErrorClassificationService:
    """Tests for ErrorClassificationService enhanced methods."""
    
    def test_get_detailed_error_info_missing_dependency(self):
        service = get_error_classification_service()
        error = Exception("Compilation failed: package org.junit.jupiter.api does not exist")
        context = {
            "step": "compilation",
            "compiler_output": "error: package org.junit.jupiter.api does not exist"
        }
        
        result = service.get_detailed_error_info(error, context)
        
        assert result["sub_category"] == "MISSING_DEPENDENCY"
        assert result["is_environment_issue"] is True
        assert result["needs_dependency_resolution"] is True
        assert "dependency_info" in result
    
    def test_get_detailed_error_info_code_error(self):
        service = get_error_classification_service()
        error = Exception("Compilation failed: incompatible types")
        context = {
            "step": "compilation",
            "compiler_output": "error: incompatible types: String cannot be converted to int"
        }
        
        result = service.get_detailed_error_info(error, context)
        
        assert result["sub_category"] == "INCOMPATIBLE_TYPES"
        assert result["is_environment_issue"] is False
        assert result["needs_dependency_resolution"] is False
    
    def test_get_strategy_for_sub_category(self):
        service = get_error_classification_service()
        
        assert service.get_strategy_for_sub_category(ErrorSubCategory.MISSING_DEPENDENCY) == "INSTALL_DEPENDENCIES"
        assert service.get_strategy_for_sub_category(ErrorSubCategory.MAVEN_DEPENDENCY_ERROR) == "RESOLVE_DEPENDENCIES"
        assert service.get_strategy_for_sub_category(ErrorSubCategory.SYNTAX_ERROR) == "ANALYZE_AND_FIX"
        assert service.get_strategy_for_sub_category(ErrorSubCategory.UNKNOWN) == "ANALYZE_AND_FIX"


class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_get_detailed_error_info_function(self):
        error = Exception("package org.junit does not exist")
        context = {"step": "compilation", "compiler_output": "error: package org.junit does not exist"}
        
        result = get_detailed_error_info(error, context)
        
        assert "sub_category" in result
        assert "is_environment_issue" in result
