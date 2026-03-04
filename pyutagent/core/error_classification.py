"""Unified error classification service.

This module provides a single point for all error classification logic,
eliminating duplication across the codebase.
"""

import logging
import re
from enum import Enum, auto
from typing import Any, Dict, List, Optional

from pyutagent.core.error_recovery import ErrorCategory, ErrorClassifier

logger = logging.getLogger(__name__)


class ErrorSubCategory(Enum):
    """错误子类型细分，用于更精确的错误诊断和恢复策略选择"""
    # 编译错误子类型
    MISSING_DEPENDENCY = auto()       # 缺少依赖包
    MISSING_IMPORT = auto()           # 缺少导入语句
    TYPE_MISMATCH = auto()            # 类型不匹配
    SYNTAX_ERROR = auto()             # 语法错误
    SYMBOL_NOT_FOUND = auto()         # 符号未找到
    INCOMPATIBLE_TYPES = auto()       # 类型不兼容
    METHOD_NOT_FOUND = auto()         # 方法未找到
    
    # 工具执行错误子类型
    MAVEN_DEPENDENCY_ERROR = auto()   # Maven依赖解析错误
    MAVEN_BUILD_ERROR = auto()        # Maven构建错误
    MAVEN_TEST_ERROR = auto()         # Maven测试错误
    MAVEN_NETWORK_ERROR = auto()      # Maven网络错误
    
    # 测试错误子类型
    ASSERTION_FAILED = auto()         # 断言失败
    NULL_POINTER = auto()             # 空指针异常
    MOCK_INVOCATION_ERROR = auto()    # Mock调用错误
    TEST_TIMEOUT = auto()             # 测试超时
    
    # 环境错误子类型
    NETWORK_UNREACHABLE = auto()      # 网络不可达
    PERMISSION_DENIED = auto()        # 权限被拒绝
    RESOURCE_EXHAUSTED = auto()       # 资源耗尽
    
    # 未知子类型
    UNKNOWN = auto()


DEPENDENCY_ERROR_PATTERNS = [
    (r"package\s+([\w.]+)\s+does not exist", "missing_package"),
    (r"cannot find symbol.*package\s+([\w.]+)", "missing_package"),
    (r"could not resolve dependencies", "resolve_failed"),
    (r"Failed to execute goal.*dependency", "maven_dependency"),
    (r"Failed to collect dependencies", "collect_failed"),
    (r"Could not find artifact\s+([\w.:-]+)", "artifact_not_found"),
    (r"Cannot resolve\s+([\w.:-]+)", "cannot_resolve"),
]

TEST_DEPENDENCY_PACKAGES = {
    "org.junit": "junit-jupiter",
    "org.junit.jupiter": "junit-jupiter",
    "org.mockito": "mockito-core",
    "org.assertj": "assertj-core",
    "org.hamcrest": "hamcrest",
    "org.powermock": "powermock",
}


def detect_missing_dependencies(compiler_output: str) -> Dict[str, Any]:
    """检测编译输出中缺失的依赖包
    
    Args:
        compiler_output: 编译器输出字符串
        
    Returns:
        包含缺失依赖信息的字典：
        - missing_packages: 缺失的包名列表
        - suggested_dependencies: 建议添加的Maven依赖
        - is_test_dependency: 是否为测试依赖
        - sub_category: 错误子类型
    """
    missing_packages = []
    suggested_dependencies = []
    is_test_dependency = False
    
    for pattern, error_type in DEPENDENCY_ERROR_PATTERNS:
        matches = re.findall(pattern, compiler_output, re.IGNORECASE | re.MULTILINE)
        for match in matches:
            if isinstance(match, tuple):
                match = match[0] if match else ""
            
            match = match.strip()
            if match and match not in missing_packages:
                missing_packages.append(match)
                
                for pkg_prefix, dep_name in TEST_DEPENDENCY_PACKAGES.items():
                    if match.startswith(pkg_prefix):
                        is_test_dependency = True
                        if dep_name not in suggested_dependencies:
                            suggested_dependencies.append(dep_name)
                        break
    
    sub_category = ErrorSubCategory.UNKNOWN
    if missing_packages:
        sub_category = ErrorSubCategory.MISSING_DEPENDENCY
    elif "could not resolve dependencies" in compiler_output.lower():
        sub_category = ErrorSubCategory.MAVEN_DEPENDENCY_ERROR
    
    return {
        "missing_packages": missing_packages,
        "suggested_dependencies": suggested_dependencies,
        "is_test_dependency": is_test_dependency,
        "sub_category": sub_category,
        "error_count": len(missing_packages),
    }


def detect_compilation_error_type(compiler_output: str) -> ErrorSubCategory:
    """检测编译错误的具体类型
    
    Args:
        compiler_output: 编译器输出字符串
        
    Returns:
        ErrorSubCategory 枚举值
    """
    output_lower = compiler_output.lower()
    
    if "package" in output_lower and "does not exist" in output_lower:
        return ErrorSubCategory.MISSING_DEPENDENCY
    
    if "cannot find symbol" in output_lower:
        if "package" in output_lower:
            return ErrorSubCategory.MISSING_DEPENDENCY
        return ErrorSubCategory.SYMBOL_NOT_FOUND
    
    if "incompatible types" in output_lower:
        return ErrorSubCategory.INCOMPATIBLE_TYPES
    
    if "method" in output_lower and "cannot be applied" in output_lower:
        return ErrorSubCategory.METHOD_NOT_FOUND
    
    if "expected" in output_lower and (";" in output_lower or "{" in output_lower or "}" in output_lower):
        return ErrorSubCategory.SYNTAX_ERROR
    
    if "type mismatch" in output_lower:
        return ErrorSubCategory.TYPE_MISMATCH
    
    return ErrorSubCategory.UNKNOWN


def detect_maven_error_type(maven_output: str) -> ErrorSubCategory:
    """检测Maven错误的具体类型
    
    Args:
        maven_output: Maven输出字符串
        
    Returns:
        ErrorSubCategory 枚举值
    """
    output_lower = maven_output.lower()
    
    if "dependency" in output_lower or "resolve" in output_lower:
        if "could not" in output_lower or "failed" in output_lower:
            return ErrorSubCategory.MAVEN_DEPENDENCY_ERROR
    
    if "connection" in output_lower or "network" in output_lower or "timeout" in output_lower:
        return ErrorSubCategory.MAVEN_NETWORK_ERROR
    
    if "test" in output_lower and "failed" in output_lower:
        return ErrorSubCategory.MAVEN_TEST_ERROR
    
    if "build" in output_lower and "failed" in output_lower:
        return ErrorSubCategory.MAVEN_BUILD_ERROR
    
    return ErrorSubCategory.UNKNOWN


class ErrorClassificationService:
    """Unified service for error classification.
    
    This service provides a single point for all error classification
    logic, eliminating duplication across the codebase.
    
    Features:
    - Error classification by type and message
    - Retryability checking
    - Recovery strategy recommendation
    - Singleton pattern for global access
    
    Example:
        >>> service = get_error_classification_service()
        >>> category = service.classify(ValueError("test"))
        >>> print(category)
        ErrorCategory.VALIDATION
    """
    
    _instance: Optional['ErrorClassificationService'] = None
    
    def __new__(cls) -> 'ErrorClassificationService':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def classify(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> ErrorCategory:
        """Classify an error into a category.
        
        Args:
            error: The error to classify
            context: Optional context information (step name, etc.)
            
        Returns:
            ErrorCategory enum value
        """
        category = ErrorClassifier.classify(error)
        
        if context:
            step = context.get("step", "").lower()
            if "compile" in step:
                return ErrorCategory.COMPILATION_ERROR
            elif "test" in step and "fail" in str(error).lower():
                return ErrorCategory.TEST_FAILURE
        
        return category
    
    def categorize_by_message(self, error_message: str, error_details: Dict[str, Any]) -> ErrorCategory:
        """Categorize error by message and details.
        
        Args:
            error_message: Error message string
            error_details: Additional error details
            
        Returns:
            ErrorCategory enum value
        """
        return ErrorClassifier.categorize_error(error_message, error_details)
    
    def is_retryable(self, error: Exception) -> bool:
        """Check if an error is retryable.
        
        Args:
            error: The error to check
            
        Returns:
            True if the error is retryable
        """
        return ErrorClassifier.is_retryable(error)
    
    def is_retryable_category(self, category: ErrorCategory) -> bool:
        """Check if an error category is retryable.
        
        Args:
            category: Error category to check
            
        Returns:
            True if the category is retryable
        """
        retryable_categories = {
            ErrorCategory.TRANSIENT,
            ErrorCategory.NETWORK,
            ErrorCategory.TIMEOUT,
            ErrorCategory.RESOURCE,
            ErrorCategory.COMPILATION_ERROR,
            ErrorCategory.TEST_FAILURE,
            ErrorCategory.LLM_API_ERROR,
        }
        return category in retryable_categories
    
    def get_recovery_strategy(self, error: Exception, attempt_count: int = 0) -> str:
        """Get recommended recovery strategy for an error.
        
        Args:
            error: The error
            attempt_count: Number of previous attempts
            
        Returns:
            Strategy name string
        """
        category = self.classify(error)
        return self.get_strategy_for_category(category, attempt_count)
    
    def get_strategy_for_category(self, category: ErrorCategory, attempt_count: int = 0) -> str:
        """Get recommended recovery strategy for an error category.
        
        Args:
            category: Error category
            attempt_count: Number of previous attempts
            
        Returns:
            Strategy name string
        """
        if category == ErrorCategory.NETWORK:
            return "RETRY_WITH_BACKOFF"
        elif category == ErrorCategory.TIMEOUT:
            return "RETRY_WITH_BACKOFF"
        elif category == ErrorCategory.COMPILATION_ERROR:
            return "ANALYZE_AND_FIX"
        elif category == ErrorCategory.TEST_FAILURE:
            if attempt_count < 3:
                return "ANALYZE_AND_FIX"
            else:
                return "RESET_AND_REGENERATE"
        elif category == ErrorCategory.LLM_API_ERROR:
            if attempt_count < 2:
                return "RETRY_WITH_BACKOFF"
            else:
                return "FALLBACK_ALTERNATIVE"
        elif category in (ErrorCategory.TRANSIENT, ErrorCategory.RESOURCE):
            return "RETRY_IMMEDIATE"
        elif category == ErrorCategory.PARSING_ERROR:
            return "RESET_AND_REGENERATE"
        elif category == ErrorCategory.FILE_IO_ERROR:
            return "SKIP_AND_CONTINUE"
        else:
            return "ANALYZE_AND_FIX"
    
    def get_error_severity(self, error: Exception) -> str:
        """Get severity level for an error.
        
        Args:
            error: The error
            
        Returns:
            Severity level string: "low", "medium", "high", "critical"
        """
        category = self.classify(error)
        
        severity_mapping = {
            ErrorCategory.TRANSIENT: "low",
            ErrorCategory.NETWORK: "medium",
            ErrorCategory.TIMEOUT: "medium",
            ErrorCategory.RESOURCE: "high",
            ErrorCategory.VALIDATION: "medium",
            ErrorCategory.SYNTAX: "high",
            ErrorCategory.LOGIC: "medium",
            ErrorCategory.COMPILATION_ERROR: "medium",
            ErrorCategory.TEST_FAILURE: "medium",
            ErrorCategory.TOOL_EXECUTION_ERROR: "high",
            ErrorCategory.PARSING_ERROR: "high",
            ErrorCategory.GENERATION_ERROR: "high",
            ErrorCategory.FILE_IO_ERROR: "high",
            ErrorCategory.LLM_API_ERROR: "high",
            ErrorCategory.PERMANENT: "critical",
            ErrorCategory.UNKNOWN: "medium",
        }
        
        return severity_mapping.get(category, "medium")
    
    def should_skip_recovery(self, error: Exception) -> bool:
        """Check if recovery should be skipped for this error.
        
        Args:
            error: The error
            
        Returns:
            True if recovery should be skipped
        """
        error_message = str(error).lower()
        
        skip_patterns = [
            "no compilation errors",
            "no test failures",
            "all tests passed",
            "compilation successful",
            "tests passed",
        ]
        
        for pattern in skip_patterns:
            if pattern in error_message:
                return True
        
        return False
    
    def get_error_info(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get comprehensive error information.
        
        Args:
            error: The error
            context: Optional context information
            
        Returns:
            Dictionary with error information
        """
        category = self.classify(error, context)
        
        return {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "category": category.name,
            "is_retryable": self.is_retryable_category(category),
            "recommended_strategy": self.get_strategy_for_category(category),
            "severity": self.get_error_severity(error),
            "should_skip_recovery": self.should_skip_recovery(error),
        }
    
    def get_detailed_error_info(
        self, 
        error: Exception, 
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """获取详细的错误信息，包括子类型和依赖检测
        
        Args:
            error: 错误对象
            context: 上下文信息，可包含 compiler_output, step 等
            
        Returns:
            包含详细错误信息的字典
        """
        base_info = self.get_error_info(error, context)
        category = self.classify(error, context)
        
        sub_category = ErrorSubCategory.UNKNOWN
        dependency_info = {}
        
        compiler_output = context.get("compiler_output", "") if context else ""
        if not compiler_output:
            compiler_output = str(error)
        
        if category == ErrorCategory.COMPILATION_ERROR:
            sub_category = detect_compilation_error_type(compiler_output)
            if sub_category == ErrorSubCategory.MISSING_DEPENDENCY:
                dependency_info = detect_missing_dependencies(compiler_output)
        
        elif category == ErrorCategory.TOOL_EXECUTION_ERROR:
            tool_name = context.get("tool", "") if context else ""
            if "maven" in tool_name.lower() or "mvn" in tool_name.lower():
                sub_category = detect_maven_error_type(compiler_output)
                if sub_category == ErrorSubCategory.MAVEN_DEPENDENCY_ERROR:
                    dependency_info = detect_missing_dependencies(compiler_output)
        
        elif category == ErrorCategory.TEST_FAILURE:
            failures = context.get("failures", []) if context else []
            if failures:
                first_failure = failures[0] if failures else {}
                error_msg = first_failure.get("error", "")
                if "assertion" in error_msg.lower():
                    sub_category = ErrorSubCategory.ASSERTION_FAILED
                elif "nullpointer" in error_msg.lower():
                    sub_category = ErrorSubCategory.NULL_POINTER
                elif "mock" in error_msg.lower():
                    sub_category = ErrorSubCategory.MOCK_INVOCATION_ERROR
        
        result = {
            **base_info,
            "sub_category": sub_category.name if sub_category != ErrorSubCategory.UNKNOWN else None,
            "is_environment_issue": self._is_environment_issue(category, sub_category),
            "needs_dependency_resolution": sub_category in (
                ErrorSubCategory.MISSING_DEPENDENCY,
                ErrorSubCategory.MAVEN_DEPENDENCY_ERROR,
            ),
        }
        
        if dependency_info:
            result["dependency_info"] = dependency_info
        
        return result
    
    def _is_environment_issue(self, category: ErrorCategory, sub_category: ErrorSubCategory) -> bool:
        """判断是否为环境问题（非代码问题）
        
        Args:
            category: 错误类别
            sub_category: 错误子类别
            
        Returns:
            如果是环境问题返回 True
        """
        environment_sub_categories = {
            ErrorSubCategory.MISSING_DEPENDENCY,
            ErrorSubCategory.MAVEN_DEPENDENCY_ERROR,
            ErrorSubCategory.MAVEN_NETWORK_ERROR,
            ErrorSubCategory.NETWORK_UNREACHABLE,
            ErrorSubCategory.PERMISSION_DENIED,
            ErrorSubCategory.RESOURCE_EXHAUSTED,
        }
        
        if sub_category in environment_sub_categories:
            return True
        
        if category in (ErrorCategory.NETWORK, ErrorCategory.RESOURCE):
            return True
        
        return False
    
    def get_strategy_for_sub_category(
        self, 
        sub_category: ErrorSubCategory,
        attempt_count: int = 0
    ) -> str:
        """根据错误子类型获取恢复策略
        
        Args:
            sub_category: 错误子类型
            attempt_count: 尝试次数
            
        Returns:
            策略名称字符串
        """
        strategy_mapping = {
            ErrorSubCategory.MISSING_DEPENDENCY: "INSTALL_DEPENDENCIES",
            ErrorSubCategory.MAVEN_DEPENDENCY_ERROR: "RESOLVE_DEPENDENCIES",
            ErrorSubCategory.MAVEN_NETWORK_ERROR: "RETRY_WITH_BACKOFF",
            ErrorSubCategory.NETWORK_UNREACHABLE: "RETRY_WITH_BACKOFF",
            ErrorSubCategory.PERMISSION_DENIED: "ESCALATE_TO_USER",
            ErrorSubCategory.RESOURCE_EXHAUSTED: "BACKOFF",
            ErrorSubCategory.SYNTAX_ERROR: "ANALYZE_AND_FIX",
            ErrorSubCategory.TYPE_MISMATCH: "ANALYZE_AND_FIX",
            ErrorSubCategory.INCOMPATIBLE_TYPES: "ANALYZE_AND_FIX",
            ErrorSubCategory.SYMBOL_NOT_FOUND: "ANALYZE_AND_FIX",
            ErrorSubCategory.METHOD_NOT_FOUND: "ANALYZE_AND_FIX",
            ErrorSubCategory.ASSERTION_FAILED: "ANALYZE_AND_FIX",
            ErrorSubCategory.NULL_POINTER: "ANALYZE_AND_FIX",
            ErrorSubCategory.MOCK_INVOCATION_ERROR: "ANALYZE_AND_FIX",
            ErrorSubCategory.TEST_TIMEOUT: "RETRY_WITH_BACKOFF",
        }
        
        return strategy_mapping.get(sub_category, "ANALYZE_AND_FIX")


error_classification_service = ErrorClassificationService()


def get_error_classification_service() -> ErrorClassificationService:
    """Get the global error classification service instance.
    
    Returns:
        ErrorClassificationService singleton instance
    """
    return error_classification_service


def classify_error(error: Exception, context: Optional[Dict[str, Any]] = None) -> ErrorCategory:
    """Convenience function to classify an error.
    
    Args:
        error: The error to classify
        context: Optional context information
        
    Returns:
        ErrorCategory enum value
    """
    return get_error_classification_service().classify(error, context)


def is_retryable_error(error: Exception) -> bool:
    """Convenience function to check if an error is retryable.
    
    Args:
        error: The error to check
        
    Returns:
        True if the error is retryable
    """
    return get_error_classification_service().is_retryable(error)


def get_recovery_strategy(error: Exception, attempt_count: int = 0) -> str:
    """Convenience function to get recovery strategy.
    
    Args:
        error: The error
        attempt_count: Number of previous attempts
        
    Returns:
        Strategy name string
    """
    return get_error_classification_service().get_recovery_strategy(error, attempt_count)


def get_detailed_error_info(error: Exception, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Convenience function to get detailed error info.
    
    Args:
        error: The error
        context: Optional context information
        
    Returns:
        Dictionary with detailed error information
    """
    return get_error_classification_service().get_detailed_error_info(error, context)
