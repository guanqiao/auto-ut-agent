"""Hooks System - 生命周期钩子系统

参考 Claude Code Hooks 实现，允许在特定生命周期事件中注入自定义逻辑。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Callable, Any, Optional, Set
from enum import Enum, auto
from abc import ABC, abstractmethod
import asyncio
import logging
import time
from functools import wraps

logger = logging.getLogger(__name__)


class HookType(Enum):
    """钩子类型"""
    USER_PROMPT_SUBMIT = auto()    # 用户提交提示后
    PRE_TOOL_USE = auto()          # 工具执行前
    POST_TOOL_USE = auto()         # 工具执行后
    PRE_SUBTASK = auto()           # 子任务执行前
    POST_SUBTASK = auto()          # 子任务执行后
    ON_ERROR = auto()              # 发生错误时
    ON_SUCCESS = auto()            # 任务成功完成时
    ON_STOP = auto()               # Agent 停止时
    ON_PLAN_CREATED = auto()       # 计划创建后
    ON_PLAN_ADJUSTED = auto()      # 计划调整后
    ON_TASK_START = auto()         # 任务开始时
    ON_TASK_COMPLETE = auto()      # 任务完成时


@dataclass
class HookContext:
    """钩子上下文"""
    hook_type: HookType
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    def get(self, key: str, default=None):
        """获取数据"""
        return self.data.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """设置数据"""
        self.data[key] = value
    
    def update(self, data: Dict[str, Any]) -> None:
        """批量更新数据"""
        self.data.update(data)


@dataclass
class HookResult:
    """钩子执行结果"""
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    should_abort: bool = False
    modified_context: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: float = 0.0


class Hook(ABC):
    """钩子基类"""
    
    def __init__(
        self,
        name: str,
        hook_type: HookType,
        priority: int = 0,
        condition: Optional[Callable[[HookContext], bool]] = None,
        enabled: bool = True
    ):
        self.name = name
        self.hook_type = hook_type
        self.priority = priority
        self.condition = condition
        self.enabled = enabled
        self.execution_count = 0
        self.total_execution_time = 0.0
        self.success_count = 0
        self.failure_count = 0
    
    async def execute(self, context: HookContext) -> HookResult:
        """执行钩子"""
        if not self.enabled:
            return HookResult(success=True, data={})
        
        # 检查条件
        if self.condition and not self.condition(context):
            return HookResult(success=True, data={})
        
        start_time = time.time()
        self.execution_count += 1
        
        try:
            result = await self._execute_impl(context)
            result.execution_time = time.time() - start_time
            
            if result.success:
                self.success_count += 1
            else:
                self.failure_count += 1
            
            self.total_execution_time += result.execution_time
            return result
            
        except Exception as e:
            self.failure_count += 1
            logger.error(f"Hook {self.name} failed: {e}")
            return HookResult(
                success=False,
                error=str(e),
                execution_time=time.time() - start_time
            )
    
    @abstractmethod
    async def _execute_impl(self, context: HookContext) -> HookResult:
        """钩子具体实现"""
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'name': self.name,
            'hook_type': self.hook_type.name,
            'execution_count': self.execution_count,
            'success_count': self.success_count,
            'failure_count': self.failure_count,
            'average_execution_time': self.total_execution_time / max(self.execution_count, 1),
            'enabled': self.enabled
        }


class FunctionHook(Hook):
    """函数钩子 - 使用函数作为处理逻辑"""
    
    def __init__(
        self,
        name: str,
        hook_type: HookType,
        handler: Callable[[HookContext], Any],
        priority: int = 0,
        condition: Optional[Callable[[HookContext], bool]] = None,
        enabled: bool = True
    ):
        super().__init__(name, hook_type, priority, condition, enabled)
        self.handler = handler
    
    async def _execute_impl(self, context: HookContext) -> HookResult:
        """执行函数钩子"""
        if asyncio.iscoroutinefunction(self.handler):
            result = await self.handler(context)
        else:
            result = self.handler(context)
        
        # 统一返回格式
        if isinstance(result, HookResult):
            return result
        elif isinstance(result, dict):
            return HookResult(success=True, data=result)
        elif result is None:
            return HookResult(success=True, data={})
        else:
            return HookResult(success=True, data={'result': result})


class HookRegistry:
    """钩子注册表"""
    
    def __init__(self):
        self._hooks: Dict[HookType, List[Hook]] = {hook_type: [] for hook_type in HookType}
        self._hook_names: Set[str] = set()
    
    def register(self, hook: Hook) -> bool:
        """注册钩子
        
        Returns:
            bool: 是否注册成功
        """
        if hook.name in self._hook_names:
            logger.warning(f"Hook {hook.name} already registered, skipping")
            return False
        
        self._hooks[hook.hook_type].append(hook)
        self._hook_names.add(hook.name)
        
        # 按优先级排序（高优先级在前）
        self._hooks[hook.hook_type].sort(key=lambda h: h.priority, reverse=True)
        
        logger.info(f"Registered hook: {hook.name} for {hook.hook_type.name} (priority={hook.priority})")
        return True
    
    def unregister(self, hook_name: str) -> bool:
        """注销钩子"""
        if hook_name not in self._hook_names:
            return False
        
        for hooks in self._hooks.values():
            for i, hook in enumerate(hooks):
                if hook.name == hook_name:
                    hooks.pop(i)
                    self._hook_names.discard(hook_name)
                    logger.info(f"Unregistered hook: {hook_name}")
                    return True
        return False
    
    def get_hook(self, hook_name: str) -> Optional[Hook]:
        """获取钩子"""
        for hooks in self._hooks.values():
            for hook in hooks:
                if hook.name == hook_name:
                    return hook
        return None
    
    def get_hooks(self, hook_type: HookType) -> List[Hook]:
        """获取指定类型的所有钩子"""
        return self._hooks.get(hook_type, []).copy()
    
    async def execute_hooks(
        self,
        hook_type: HookType,
        context: HookContext
    ) -> HookResult:
        """执行指定类型的所有钩子"""
        hooks = self._hooks.get(hook_type, [])
        combined_data: Dict[str, Any] = {}
        
        for hook in hooks:
            result = await hook.execute(context)
            combined_data.update(result.data)
            
            if result.should_abort:
                logger.info(f"Hook {hook.name} requested abort")
                return HookResult(
                    success=result.success,
                    data=combined_data,
                    should_abort=True,
                    execution_time=result.execution_time
                )
            
            if result.modified_context:
                context.data.update(result.modified_context)
        
        return HookResult(success=True, data=combined_data)
    
    def get_all_stats(self) -> Dict[str, Any]:
        """获取所有钩子的统计信息"""
        stats = {}
        for hook_type, hooks in self._hooks.items():
            if hooks:
                stats[hook_type.name] = [hook.get_stats() for hook in hooks]
        return stats
    
    def clear(self) -> None:
        """清空所有钩子"""
        self._hooks = {hook_type: [] for hook_type in HookType}
        self._hook_names.clear()
        logger.info("Cleared all hooks")


class HookManager:
    """钩子管理器"""
    
    def __init__(self):
        self.registry = HookRegistry()
        self._builtin_hooks_registered = False
        self._execution_history: List[Dict[str, Any]] = []
    
    def register_builtin_hooks(self) -> None:
        """注册内置钩子"""
        if self._builtin_hooks_registered:
            return
        
        # 1. 代码格式化钩子
        self.registry.register(FunctionHook(
            name="auto_format",
            hook_type=HookType.POST_TOOL_USE,
            handler=self._auto_format_handler,
            priority=10,
            condition=lambda ctx: ctx.get('tool_name') == 'file_write' and 
                                 str(ctx.get('file_path', '')).endswith('.java')
        ))
        
        # 2. 操作日志钩子
        self.registry.register(FunctionHook(
            name="operation_logger",
            hook_type=HookType.POST_TOOL_USE,
            handler=self._operation_log_handler,
            priority=5
        ))
        
        # 3. 敏感操作确认钩子
        self.registry.register(FunctionHook(
            name="sensitive_operation_confirm",
            hook_type=HookType.PRE_TOOL_USE,
            handler=self._sensitive_operation_handler,
            priority=100,
            condition=lambda ctx: ctx.get('tool_name') in ['file_delete', 'git_push', 'mvn_deploy']
        ))
        
        # 4. 错误恢复钩子
        self.registry.register(FunctionHook(
            name="error_recovery",
            hook_type=HookType.ON_ERROR,
            handler=self._error_recovery_handler,
            priority=50
        ))
        
        # 5. 任务开始日志钩子
        self.registry.register(FunctionHook(
            name="task_start_logger",
            hook_type=HookType.ON_TASK_START,
            handler=self._task_start_handler,
            priority=1
        ))
        
        # 6. 任务完成日志钩子
        self.registry.register(FunctionHook(
            name="task_complete_logger",
            hook_type=HookType.ON_TASK_COMPLETE,
            handler=self._task_complete_handler,
            priority=1
        ))
        
        self._builtin_hooks_registered = True
        logger.info("Registered built-in hooks")
    
    def _auto_format_handler(self, context: HookContext) -> Dict[str, Any]:
        """自动格式化代码"""
        file_path = context.get('file_path')
        if file_path:
            logger.info(f"Auto-formatting: {file_path}")
            # 实际实现可以调用代码格式化工具
        return {'formatted': True, 'file_path': file_path}
    
    def _operation_log_handler(self, context: HookContext) -> Dict[str, Any]:
        """记录操作日志"""
        tool_name = context.get('tool_name')
        logger.info(f"Tool executed: {tool_name}")
        return {'logged': True, 'tool_name': tool_name}
    
    def _sensitive_operation_handler(self, context: HookContext) -> HookResult:
        """敏感操作确认"""
        tool_name = context.get('tool_name')
        operation_details = context.get('details', {})
        
        logger.warning(f"Sensitive operation detected: {tool_name}")
        
        # 这里可以实现用户确认逻辑
        # 例如：弹出确认对话框或记录审计日志
        
        return HookResult(
            success=True,
            data={'confirmed': True, 'tool_name': tool_name},
            modified_context={'sensitive_operation_logged': True}
        )
    
    def _error_recovery_handler(self, context: HookContext) -> Dict[str, Any]:
        """错误恢复处理"""
        error = context.get('error')
        operation = context.get('operation', 'unknown')
        
        logger.error(f"Error in {operation}: {error}")
        
        # 可以在这里实现自动恢复逻辑
        return {
            'error_logged': True,
            'operation': operation,
            'timestamp': time.time()
        }
    
    def _task_start_handler(self, context: HookContext) -> Dict[str, Any]:
        """任务开始处理"""
        task_id = context.get('task_id')
        task_type = context.get('task_type')
        
        logger.info(f"Task started: {task_id} ({task_type})")
        
        return {
            'task_started': True,
            'task_id': task_id,
            'start_time': time.time()
        }
    
    def _task_complete_handler(self, context: HookContext) -> Dict[str, Any]:
        """任务完成处理"""
        task_id = context.get('task_id')
        success = context.get('success', False)
        
        logger.info(f"Task completed: {task_id} (success={success})")
        
        return {
            'task_completed': True,
            'task_id': task_id,
            'success': success,
            'end_time': time.time()
        }
    
    async def trigger(
        self,
        hook_type: HookType,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> HookResult:
        """触发钩子"""
        context = HookContext(
            hook_type=hook_type,
            data=data.copy(),
            metadata=metadata or {}
        )
        
        result = await self.registry.execute_hooks(hook_type, context)
        
        # 记录执行历史
        self._execution_history.append({
            'hook_type': hook_type.name,
            'timestamp': time.time(),
            'success': result.success,
            'should_abort': result.should_abort
        })
        
        return result
    
    def register_hook(
        self,
        name: str,
        hook_type: HookType,
        handler: Callable,
        priority: int = 0,
        condition: Optional[Callable] = None
    ) -> bool:
        """注册自定义钩子"""
        hook = FunctionHook(
            name=name,
            hook_type=hook_type,
            handler=handler,
            priority=priority,
            condition=condition
        )
        return self.registry.register(hook)
    
    def unregister_hook(self, hook_name: str) -> bool:
        """注销钩子"""
        return self.registry.unregister(hook_name)
    
    def enable_hook(self, hook_name: str) -> bool:
        """启用钩子"""
        hook = self.registry.get_hook(hook_name)
        if hook:
            hook.enabled = True
            return True
        return False
    
    def disable_hook(self, hook_name: str) -> bool:
        """禁用钩子"""
        hook = self.registry.get_hook(hook_name)
        if hook:
            hook.enabled = False
            return True
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'registry_stats': self.registry.get_all_stats(),
            'execution_history_count': len(self._execution_history),
            'builtin_hooks_registered': self._builtin_hooks_registered
        }


# 装饰器方式注册钩子
def hook(
    hook_type: HookType,
    priority: int = 0,
    condition: Optional[Callable[[HookContext], bool]] = None,
    name: Optional[str] = None
):
    """钩子装饰器
    
    使用示例:
        @hook(HookType.POST_TOOL_USE, priority=10)
        def my_handler(context: HookContext) -> HookResult:
            ...
    """
    def decorator(func: Callable) -> Callable:
        func._hook_type = hook_type
        func._hook_priority = priority
        func._hook_condition = condition
        func._hook_name = name or func.__name__
        return func
    return decorator


class HookMixin:
    """钩子混入类 - 为其他类提供钩子支持"""
    
    def __init__(self):
        self._hook_manager: Optional[HookManager] = None
    
    def set_hook_manager(self, hook_manager: HookManager) -> None:
        """设置钩子管理器"""
        self._hook_manager = hook_manager
    
    async def trigger_hook(
        self,
        hook_type: HookType,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> HookResult:
        """触发钩子"""
        if self._hook_manager:
            return await self._hook_manager.trigger(hook_type, data, metadata)
        return HookResult(success=True, data={})


# 全局钩子管理器实例
_global_hook_manager: Optional[HookManager] = None


def get_global_hook_manager() -> HookManager:
    """获取全局钩子管理器"""
    global _global_hook_manager
    if _global_hook_manager is None:
        _global_hook_manager = HookManager()
    return _global_hook_manager


def set_global_hook_manager(hook_manager: HookManager) -> None:
    """设置全局钩子管理器"""
    global _global_hook_manager
    _global_hook_manager = hook_manager
