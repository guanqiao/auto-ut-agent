"""Internationalization (i18n) support for PyUT Agent.

This module provides multi-language support for the application.
Currently supports English (default) and Chinese.
"""

import logging
from enum import Enum, auto
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class Language(Enum):
    """Supported languages."""
    ENGLISH = auto()
    CHINESE = auto()


class I18n:
    """Internationalization manager."""
    
    _instance: Optional['I18n'] = None
    _current_language = Language.ENGLISH
    
    # Translation dictionary
    _translations: Dict[Language, Dict[str, str]] = {
        Language.ENGLISH: {
            # Common
            "app_name": "PyUT Agent",
            "ok": "OK",
            "cancel": "Cancel",
            "save": "Save",
            "delete": "Delete",
            "add": "Add",
            "edit": "Edit",
            "close": "Close",
            "browse": "Browse...",
            "test": "Test",
            
            # Status
            "status_idle": "Idle",
            "status_running": "Running",
            "status_paused": "Paused",
            "status_completed": "Completed",
            "status_failed": "Failed",
            "status_compiling": "Compiling",
            "status_testing": "Testing",
            "status_fixing": "Fixing",
            "status_analyzing": "Analyzing",
            "status_optimizing": "Optimizing",
            
            # Task messages
            "task_paused": "Task paused, waiting to resume...",
            "task_resumed": "Task resumed",
            "task_stopped": "Task stopped",
            "pause_requested": "Pause request sent",
            "resume_requested": "Resume request sent",
            
            # Generation messages
            "parsing_java_file": "Parsing Java file...",
            "parsing_file": "Parsing: {file}",
            "found_class": "Found class: {class_name}, methods: {method_count}",
            "generating_tests": "Generating initial tests...",
            "running_tests": "Running tests...",
            "analyzing_coverage": "Analyzing coverage...",
            "current_coverage": "Current coverage: {coverage:.1%}",
            "optimization_iteration": "Optimization iteration {current}/{total}...",
            "iteration_coverage": "Iteration {iteration}: coverage {coverage:.1%}",
            "all_lines_covered": "All lines covered",
            "test_run_failed": "Test run failed, stopping optimization",
            "completed": "Completed!",
            
            # Aider messages
            "fixing_with_aider": "Fixing with Aider-style editing...",
            "no_errors_detected": "No errors detected",
            "errors_detected": "Detected {count} errors",
            "fix_success": "Fix successful (attempt {attempts})",
            "fix_failed": "Fix failed: {error}",
            "aider_fix_failed": "Aider fix failed: {error}",
            "additional_tests_added": "Additional test code added",
            "coverage_improvement_failed": "Coverage improvement failed: {error}",
            "coverage_optimization_error": "Coverage optimization error: {error}",
            
            # File messages
            "test_file_saved": "Test file saved: {path}",
            "test_code_appended": "Test code appended to: {path}",
            
            # Compilation messages
            "compiling_tests": "Compiling tests...",
            "compilation_failed_retry": "Compilation failed, retrying with Aider (attempt {attempt}/3)...",
            
            # Test failure messages
            "tests_failed_fixing": "Tests failed, attempting to fix with Aider...",
            
            # Error messages
            "error": "Error",
            "warning": "Warning",
            "info": "Info",
            "success": "Success",
            "failed": "Failed",
            
            # Config dialog
            "llm_config_management": "LLM Configuration Management",
            "config_name": "Configuration Name",
            "provider": "Provider",
            "model": "Model",
            "api_key": "API Key",
            "endpoint": "Endpoint",
            "timeout": "Timeout",
            "retry_count": "Retry Count",
            "test_connection": "Test Connection",
            "testing": "Testing...",
            "testing_connection": "Testing connection...",
            "connection_success": "Connection successful",
            "connection_failed": "Connection failed: {error}",
            "config_incomplete": "Configuration incomplete",
            "please_fill_required": "Please fill in API Key and Endpoint",
            "confirm_delete": "Confirm Delete",
            "delete_confirm_message": 'Are you sure you want to delete configuration "{name}"?',
            "cannot_delete": "Cannot Delete",
            "at_least_one_config": "At least one configuration must be kept",
            "new_config": "New Config {number}",
            "set_as_default": "Set as Default",
            "config_list": "Configuration List",
            "seconds": "seconds",
            
            # Aider config dialog
            "aider_advanced_config": "Aider Advanced Configuration",
            "general": "General",
            "architect_editor": "Architect/Editor",
            "multi_file_edit": "Multi-file Edit",
            "edit_format": "Edit Format",
            "core_settings": "Core Settings",
            "max_attempts": "Max Attempts",
            "enable_degradation": "Enable Degradation Strategy",
            "enable_circuit_breaker": "Enable Circuit Breaker",
            "track_cost": "Track Cost",
            "enable_architect_editor": "Enable Architect/Editor Dual Model Mode",
            "architect_model": "Architect Model",
            "editor_model": "Editor Model",
            "work_mode": "Work Mode",
            "dual_model_mode": "Dual Model Mode",
            "single_model_mode": "Single Model Mode",
            "enable_multi_file": "Enable Multi-file Batch Edit",
            "max_files": "Max Files",
            "preferred_format": "Preferred Format",
            "auto_detect": "Auto Detect",
            "select_model": "-- Select Model --",
            "architect_editor_description": "Architect/Editor dual model provides smarter code editing capabilities...",
            "multi_file_description": "Multi-file editing allows AI to modify multiple related files simultaneously...",
            "edit_format_description": "Edit format determines how AI generates code modifications...",
            
            # Chat widget
            "chat": "Chat",
            "generate_tests": "Generate Tests",
            "pause": "Pause",
            "status": "Status",
            "clear": "Clear",
            "input_placeholder": "Enter message... (e.g., generate tests for UserService)",
            "send": "Send",
            
            # Aider integration
            "multi_file_not_implemented": "Multi-file editing not fully implemented, falling back to single-file",
        },
        Language.CHINESE: {
            # Common
            "app_name": "PyUT Agent",
            "ok": "确定",
            "cancel": "取消",
            "save": "保存",
            "delete": "删除",
            "add": "添加",
            "edit": "编辑",
            "close": "关闭",
            "browse": "浏览...",
            "test": "测试",
            
            # Status
            "status_idle": "空闲",
            "status_running": "运行中",
            "status_paused": "已暂停",
            "status_completed": "已完成",
            "status_failed": "失败",
            "status_compiling": "编译中",
            "status_testing": "测试中",
            "status_fixing": "修复中",
            "status_analyzing": "分析中",
            "status_optimizing": "优化中",
            
            # Task messages
            "task_paused": "任务已暂停，等待恢复...",
            "task_resumed": "任务已恢复",
            "task_stopped": "任务已停止",
            "pause_requested": "暂停请求已发送",
            "resume_requested": "恢复请求已发送",
            
            # Generation messages
            "parsing_java_file": "解析 Java 文件...",
            "parsing_file": "正在解析: {file}",
            "found_class": "找到类: {class_name}, 方法数: {method_count}",
            "generating_tests": "生成初始测试...",
            "running_tests": "运行测试...",
            "analyzing_coverage": "分析覆盖率...",
            "current_coverage": "当前覆盖率: {coverage:.1%}",
            "optimization_iteration": "优化迭代 {current}/{total}...",
            "iteration_coverage": "迭代 {iteration}: 覆盖率 {coverage:.1%}",
            "all_lines_covered": "所有行已覆盖",
            "test_run_failed": "测试运行失败，停止优化",
            "completed": "完成!",
            
            # Aider messages
            "fixing_with_aider": "使用 Aider 风格编辑修复...",
            "no_errors_detected": "未检测到错误",
            "errors_detected": "检测到 {count} 个错误",
            "fix_success": "修复成功 (尝试 {attempts} 次)",
            "fix_failed": "修复失败: {error}",
            "aider_fix_failed": "Aider 修复失败: {error}",
            "additional_tests_added": "已添加额外测试代码",
            "coverage_improvement_failed": "覆盖率优化失败: {error}",
            "coverage_optimization_error": "覆盖率优化异常: {error}",
            
            # File messages
            "test_file_saved": "测试文件已保存: {path}",
            "test_code_appended": "已追加测试代码到: {path}",
            
            # Compilation messages
            "compiling_tests": "编译测试...",
            "compilation_failed_retry": "编译失败，尝试使用 Aider 修复 (尝试 {attempt}/3)...",
            
            # Test failure messages
            "tests_failed_fixing": "测试失败，尝试使用 Aider 修复...",
            
            # Error messages
            "error": "错误",
            "warning": "警告",
            "info": "信息",
            "success": "成功",
            "failed": "失败",
            
            # Config dialog
            "llm_config_management": "LLM 配置管理",
            "config_name": "配置名称",
            "provider": "提供商",
            "model": "模型",
            "api_key": "API 密钥",
            "endpoint": "端点",
            "timeout": "超时",
            "retry_count": "重试次数",
            "test_connection": "测试连接",
            "testing": "测试中...",
            "testing_connection": "正在测试连接...",
            "connection_success": "连接成功",
            "connection_failed": "连接失败: {error}",
            "config_incomplete": "配置不完整",
            "please_fill_required": "请先填写 API Key 和 Endpoint",
            "confirm_delete": "确认删除",
            "delete_confirm_message": '确定要删除配置 "{name}" 吗？',
            "cannot_delete": "无法删除",
            "at_least_one_config": "至少需要保留一个配置",
            "new_config": "新配置 {number}",
            "set_as_default": "设为默认",
            "config_list": "配置列表",
            "seconds": "秒",
            
            # Aider config dialog
            "aider_advanced_config": "Aider 高级配置",
            "general": "常规",
            "architect_editor": "Architect/Editor",
            "multi_file_edit": "多文件编辑",
            "edit_format": "编辑格式",
            "core_settings": "核心设置",
            "max_attempts": "最大尝试次数",
            "enable_degradation": "启用降级策略",
            "enable_circuit_breaker": "启用熔断保护",
            "track_cost": "跟踪成本",
            "enable_architect_editor": "启用 Architect/Editor 双模型模式",
            "architect_model": "Architect 模型",
            "editor_model": "Editor 模型",
            "work_mode": "工作模式",
            "dual_model_mode": "双模型模式",
            "single_model_mode": "单模型模式",
            "enable_multi_file": "启用多文件批量编辑",
            "max_files": "最大文件数",
            "preferred_format": "首选格式",
            "auto_detect": "自动检测",
            "select_model": "-- 选择模型 --",
            "architect_editor_description": "Architect/Editor 双模型提供更智能的代码编辑能力...",
            "multi_file_description": "多文件编辑允许 AI 同时修改多个相关文件...",
            "edit_format_description": "编辑格式决定 AI 如何生成代码修改...",
            
            # Chat widget
            "chat": "对话",
            "generate_tests": "生成测试",
            "pause": "暂停",
            "status": "状态",
            "clear": "清空",
            "input_placeholder": "输入消息... (例如: 生成 UserService 的测试)",
            "send": "发送",
            
            # Aider integration
            "multi_file_not_implemented": "多文件编辑功能未完全实现，回退到单文件编辑",
        }
    }
    
    @classmethod
    def get_instance(cls) -> 'I18n':
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    def set_language(cls, language: Language):
        """Set current language."""
        cls._current_language = language
        logger.info(f"Language set to: {language.name}")
    
    @classmethod
    def get_language(cls) -> Language:
        """Get current language."""
        return cls._current_language
    
    @classmethod
    def t(cls, key: str, **kwargs) -> str:
        """Translate a key.
        
        Args:
            key: Translation key
            **kwargs: Format arguments
            
        Returns:
            Translated string
        """
        translations = cls._translations.get(cls._current_language, {})
        text = translations.get(key, key)
        
        if kwargs:
            try:
                return text.format(**kwargs)
            except KeyError as e:
                logger.warning(f"Missing format argument {e} for key: {key}")
                return text
        
        return text
    
    @classmethod
    def add_translations(cls, language: Language, translations: Dict[str, str]):
        """Add custom translations for a language.
        
        Args:
            language: Target language
            translations: Dictionary of key-value translations
        """
        if language not in cls._translations:
            cls._translations[language] = {}
        cls._translations[language].update(translations)


# Convenience function
def t(key: str, **kwargs) -> str:
    """Translate a key.
    
    Args:
        key: Translation key
        **kwargs: Format arguments
        
    Returns:
        Translated string
    """
    return I18n.t(key, **kwargs)


def set_language(language: Language):
    """Set current language.
    
    Args:
        language: Language to set
    """
    I18n.set_language(language)


def get_language() -> Language:
    """Get current language.
    
    Returns:
        Current language
    """
    return I18n.get_language()
