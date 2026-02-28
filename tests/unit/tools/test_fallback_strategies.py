"""Unit tests for fallback_strategies module.

This module provides comprehensive tests for fallback strategies,
including rule-based fixes, template-based generation, and fallback management.
"""

import pytest
import allure
from unittest.mock import Mock, AsyncMock

from pyutagent.tools.fallback_strategies import (
    FallbackManager, FallbackLevel, FallbackResult,
    RuleBasedFixStrategy, TemplateBasedStrategy,
    create_fallback_manager, apply_fallback
)
from pyutagent.tools.error_analyzer import ErrorAnalysis, CompilationError, ErrorType
from pyutagent.tools.failure_analyzer import FailureAnalysis, TestFailure, FailureType


def display_description(description: str):
    """Decorator to add display description for test cases."""
    def decorator(func):
        func.display_description = description
        return allure.description(description)(func)
    return decorator


@allure.feature("Fallback Strategies")
@allure.story("Rule Based Fix")
class TestRuleBasedFixStrategy:
    """Tests for RuleBasedFixStrategy."""
    
    @pytest.fixture
    def rule_strategy(self):
        return RuleBasedFixStrategy()
    
    @display_description("验证修复缺少导入错误")
    @allure.title("Test fix missing import")
    @pytest.mark.asyncio
    async def test_fix_missing_import(self, rule_strategy):
        code = """public class Test { @Test void test() {} }"""
        error = CompilationError(
            error_type=ErrorType.IMPORT_ERROR,
            message="cannot find symbol: class Test",
            error_token="Test",
            fix_hint="Add import"
        )
        analysis = ErrorAnalysis(
            errors=[error], summary="1 error", fix_strategy="Fix imports", priority=1
        )
        
        result = await rule_strategy.execute(code, error_analysis=analysis)
        
        assert result.success
        assert "import org.junit.jupiter.api.Test" in result.code
    
    @display_description("验证修复缺少分号错误")
    @allure.title("Test fix missing semicolon")
    @pytest.mark.asyncio
    async def test_fix_missing_semicolon(self, rule_strategy):
        code = """public class Test { void method() { int x = 1 } }"""
        error = CompilationError(
            error_type=ErrorType.SYNTAX_ERROR,
            message="';' expected",
            line_number=1,
            fix_hint="Add semicolon"
        )
        analysis = ErrorAnalysis(
            errors=[error], summary="1 error", fix_strategy="Fix syntax", priority=1
        )
        
        result = await rule_strategy.execute(code, error_analysis=analysis)
        
        assert result.success


@allure.feature("Fallback Strategies")
@allure.story("Template Based")
class TestTemplateBasedStrategy:
    """Tests for TemplateBasedStrategy."""
    
    @pytest.fixture
    def template_strategy(self):
        return TemplateBasedStrategy()
    
    @display_description("验证基本测试模板生成")
    @allure.title("Test basic test template generation")
    @pytest.mark.asyncio
    async def test_basic_template(self, template_strategy):
        context = {
            'class_name': 'Calculator',
            'method_name': 'add',
            'return_type': 'int',
            'parameters': '1, 2',
            'assertions': 'assertEquals(3, result)'
        }
        
        result = await template_strategy.execute("", context=context)
        
        assert result.success
        assert "CalculatorTest" in result.code
        assert "testAdd" in result.code


@allure.feature("Fallback Strategies")
@allure.story("Fallback Manager")
class TestFallbackManager:
    """Tests for FallbackManager."""
    
    @pytest.fixture
    def fallback_manager(self):
        return FallbackManager()
    
    @display_description("验证降级管理器执行链")
    @allure.title("Test fallback manager execution chain")
    @pytest.mark.asyncio
    async def test_fallback_chain(self, fallback_manager):
        code = "public class Test { }"
        
        result = await fallback_manager.execute_with_fallback(code)
        
        assert isinstance(result, FallbackResult)
    
    @display_description("验证注册自定义策略")
    @allure.title("Test register custom strategy")
    def test_register_strategy(self, fallback_manager):
        custom_strategy = Mock()
        
        fallback_manager.register_strategy(FallbackLevel.RULE_BASED_FIX, custom_strategy)
        
        assert fallback_manager.get_strategy(FallbackLevel.RULE_BASED_FIX) is custom_strategy


@allure.feature("Fallback Strategies")
@allure.story("Utility Functions")
class TestUtilityFunctions:
    """Tests for utility functions."""
    
    @display_description("验证创建降级管理器")
    @allure.title("Test create fallback manager")
    def test_create_fallback_manager(self):
        manager = create_fallback_manager()
        assert isinstance(manager, FallbackManager)
