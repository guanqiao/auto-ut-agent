"""错误类型定义"""
from enum import Enum


class CompilationErrorType(Enum):
    """编译错误类型"""
    SYNTAX_ERROR = "syntax_error"
    IMPORT_ERROR = "import_error"
    SYMBOL_NOT_FOUND = "symbol_not_found"
    TYPE_MISMATCH = "type_mismatch"
    ACCESS_ERROR = "access_error"
    GENERIC_ERROR = "generic_error"


class RuntimeErrorType(Enum):
    """运行时错误类型"""
    NULL_POINTER = "null_pointer"
    INDEX_OUT_OF_BOUNDS = "index_out_of_bounds"
    CLASS_CAST = "class_cast"
    ARITHMETIC = "arithmetic"
    ILLEGAL_STATE = "illegal_state"
    ILLEGAL_ARGUMENT = "illegal_argument"
    IO_ERROR = "io_error"


class LogicErrorType(Enum):
    """逻辑错误类型"""
    ASSERTION_FAILED = "assertion_failed"
    EXPECTATION_FAILED = "expectation_failed"
    INCORRECT_RESULT = "incorrect_result"
    MISSING_ASSERTION = "missing_assertion"


class NetworkErrorType(Enum):
    """网络错误类型"""
    CONNECTION_ERROR = "connection_error"
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    API_ERROR = "api_error"
    AUTHENTICATION_ERROR = "authentication_error"


class ConfigurationErrorType(Enum):
    """配置错误类型"""
    MISSING_CONFIG = "missing_config"
    INVALID_CONFIG = "invalid_config"
    MISSING_DEPENDENCY = "missing_dependency"
    VERSION_MISMATCH = "version_mismatch"


class ValidationErrorType(Enum):
    """验证错误类型"""
    INVALID_INPUT = "invalid_input"
    MISSING_REQUIRED_FIELD = "missing_required_field"
    FORMAT_ERROR = "format_error"
    CONSTRAINT_VIOLATION = "constraint_violation"
