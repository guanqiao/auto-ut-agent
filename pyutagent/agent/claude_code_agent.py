"""
Claude Code Style Agent - Claude Code 风格的智能编程助手

集成所有参考 Claude Code 设计的新功能：
- 通用任务规划器
- Hooks 生命周期系统
- 项目配置系统
- 专业化 Subagents
- 智能上下文压缩
"""

import asyncio
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable, Union
from pathlib import Path
import logging
import time
import json

from pyutagent.agent.universal_planner import (
    UniversalTaskPlanner,
    TaskType,
    TaskUnderstanding,
    ExecutionPlan,
    ExecutionResult,
    TaskHandler
)
from pyutagent.core.hooks import (
    HookManager,
    HookType,
    HookContext,
    HookResult,
    HookMixin
)
from pyutagent.core.project_config import (
    ProjectConfigManager,
    ProjectContext,
    get_config_manager
)
from pyutagent.agent.subagents import (
    SubagentRouter,
    create_default_router,
    SubagentResult
)
from pyutagent.core.context_compactor import (
    AutoCompactManager,
    CompactedContext
)

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Agent 配置"""
    # LLM 配置
    model_name: str = "claude-3-5-sonnet"
    max_tokens: int = 128000
    temperature: float = 0.7
    
    # 上下文压缩配置
    enable_auto_compact: bool = True
    compact_threshold: float = 0.85
    
    # Hooks 配置
    enable_hooks: bool = True
    
    # Subagents 配置
    enable_subagents: bool = True
    
    # 执行配置
    max_retries: int = 3
    timeout_seconds: int = 300


@dataclass
class AgentState:
    """Agent 状态"""
    session_id: str
    project_root: Path
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    current_task: Optional[str] = None
    current_plan: Optional[ExecutionPlan] = None
    compacted_contexts: List[CompactedContext] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)


class ClaudeCodeAgent:
    """
    Claude Code 风格的智能编程助手
    
    这是 PyUT Agent 的进化版，参考 Claude Code 的核心设计哲学：
    1. 计划-执行闭环
    2. 工具生态的"连接-使用"分离
    3. 分层上下文管理
    """
    
    def __init__(
        self,
        llm_client: Any,
        tool_registry: Any,
        config: Optional[AgentConfig] = None,
        project_root: Optional[Union[str, Path]] = None
    ):
        self.llm = llm_client
        self.tools = tool_registry
        self.config = config or AgentConfig()
        self.project_root = Path(project_root) if project_root else Path.cwd()
        
        # 初始化组件
        self._init_components()
        
        # 状态
        self.state: Optional[AgentState] = None
        
        logger.info(f"ClaudeCodeAgent initialized with project: {self.project_root}")
    
    def _init_components(self) -> None:
        """初始化所有组件"""
        # 1. 通用任务规划器
        self.planner = UniversalTaskPlanner(
            llm_client=self.llm,
            project_analyzer=None,  # 可以传入项目分析器
            tool_registry=self.tools
        )
        
        # 2. Hooks 系统
        self.hook_manager = HookManager()
        if self.config.enable_hooks:
            self.hook_manager.register_builtin_hooks()
        
        # 3. 项目配置管理器
        self.config_manager = get_config_manager(self.project_root)
        
        # 4. Subagents 路由器
        self.subagent_router = create_default_router(self.llm, self.tools)
        
        # 5. 上下文压缩管理器
        self.compact_manager = AutoCompactManager(
            llm_client=self.llm,
            max_tokens=self.config.max_tokens,
            threshold=self.config.compact_threshold,
            enable_auto_compact=self.config.enable_auto_compact
        )
    
    async def start_session(self, session_id: Optional[str] = None) -> AgentState:
        """开始新会话"""
        import uuid
        
        self.state = AgentState(
            session_id=session_id or str(uuid.uuid4())[:8],
            project_root=self.project_root
        )
        
        # 触发会话开始钩子
        await self.hook_manager.trigger(
            HookType.ON_TASK_START,
            data={
                'session_id': self.state.session_id,
                'project_root': str(self.project_root)
            }
        )
        
        logger.info(f"Session started: {self.state.session_id}")
        return self.state
    
    async def process_request(
        self,
        user_request: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        处理用户请求 - 主入口
        
        这是 Claude Code 风格的核心流程：
        1. 理解任务
        2. 制定计划
        3. 执行计划
        4. 观察结果
        5. 反馈调整
        """
        if self.state is None:
            await self.start_session()
        
        context = context or {}
        start_time = time.time()
        
        try:
            # 1. 触发用户输入钩子
            hook_result = await self.hook_manager.trigger(
                HookType.USER_PROMPT_SUBMIT,
                data={'prompt': user_request, 'context': context}
            )
            
            if hook_result.should_abort:
                return {'success': False, 'error': 'Request aborted by hook'}
            
            # 2. 获取项目上下文
            project_context = self._get_project_context()
            
            # 3. 理解任务
            understanding = await self.planner.understand_task(
                user_request,
                project_context
            )
            
            logger.info(f"Task understood: {understanding.task_type.value} - {understanding.description}")
            
            # 4. 分解任务
            plan = await self.planner.decompose_task(understanding, project_context)
            self.state.current_plan = plan
            self.state.current_task = understanding.description
            
            # 触发计划创建钩子
            await self.hook_manager.trigger(
                HookType.ON_PLAN_CREATED,
                data={'plan': plan}
            )
            
            # 5. 检查是否需要使用 Subagent
            if self.config.enable_subagents and len(plan.subtasks) == 1:
                # 简单任务直接使用 Subagent
                result = await self._execute_with_subagent(
                    understanding.description,
                    understanding,
                    context
                )
            else:
                # 复杂任务使用规划器执行
                result = await self._execute_plan(plan, context)
            
            # 6. 更新会话历史
            self._update_conversation_history(user_request, result)
            
            # 7. 检查是否需要压缩上下文
            await self._check_and_compact_context()
            
            execution_time = time.time() - start_time
            
            # 触发任务完成钩子
            await self.hook_manager.trigger(
                HookType.ON_TASK_COMPLETE,
                data={
                    'success': result.get('success', False),
                    'execution_time': execution_time
                }
            )
            
            return {
                'success': result.get('success', False),
                'result': result,
                'execution_time': execution_time,
                'session_id': self.state.session_id
            }
            
        except Exception as e:
            logger.error(f"Request processing failed: {e}")
            
            # 触发错误钩子
            await self.hook_manager.trigger(
                HookType.ON_ERROR,
                data={'error': str(e), 'operation': 'process_request'}
            )
            
            return {
                'success': False,
                'error': str(e),
                'execution_time': time.time() - start_time
            }
    
    async def _execute_plan(
        self,
        plan: ExecutionPlan,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """执行计划"""
        logger.info(f"Executing plan: {plan.task_id} with {len(plan.subtasks)} subtasks")
        
        # 使用规划器的闭环执行
        execution_result = await self.planner.execute_with_feedback(
            plan,
            context,
            progress_callback=self._on_subtask_progress
        )
        
        return {
            'success': execution_result.success,
            'plan': execution_result.plan,
            'subtask_results': [
                {
                    'subtask_id': r.subtask_id,
                    'success': r.success,
                    'error': r.error,
                    'execution_time': r.execution_time
                }
                for r in execution_result.subtask_results
            ],
            'completed_subtasks': list(execution_result.completed_subtasks),
            'failed_subtasks': list(execution_result.failed_subtasks),
            'total_execution_time': execution_result.execution_time
        }
    
    async def _execute_with_subagent(
        self,
        task: str,
        understanding: TaskUnderstanding,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """使用 Subagent 执行任务"""
        logger.info(f"Executing with subagent: {task}")
        
        # 构建上下文
        subagent_context = {
            **context,
            'project_root': str(self.project_root),
            'task_type': understanding.task_type.value,
            'target_files': understanding.target_files
        }
        
        # 路由到合适的 Subagent
        result = await self.subagent_router.route(task, subagent_context)
        
        return {
            'success': result.success,
            'summary': result.summary,
            'data': result.data,
            'artifacts': result.artifacts
        }
    
    async def _on_subtask_progress(
        self,
        subtask: Any,
        result: Any
    ) -> None:
        """子任务进度回调"""
        logger.info(f"Subtask {subtask.id}: {'success' if result.success else 'failed'}")
        
        # 触发子任务完成钩子
        await self.hook_manager.trigger(
            HookType.POST_SUBTASK,
            data={
                'subtask_id': subtask.id,
                'success': result.success,
                'result': result.data
            }
        )
    
    def _get_project_context(self) -> Dict[str, Any]:
        """获取项目上下文"""
        context = self.config_manager.load_context()
        
        if context:
            return {
                'name': context.name,
                'language': context.language,
                'build_tool': context.build_tool.value,
                'java_version': context.java_version,
                'test_framework': context.test_preferences.test_framework.value,
                'mock_framework': context.test_preferences.mock_framework.value,
                'key_modules': context.key_modules,
                'source_dirs': context.source_dirs,
                'test_dirs': context.test_dirs
            }
        
        return {
            'language': 'java',
            'build_tool': 'maven',
            'project_root': str(self.project_root)
        }
    
    def _update_conversation_history(
        self,
        user_request: str,
        result: Dict[str, Any]
    ) -> None:
        """更新会话历史"""
        if self.state:
            self.state.conversation_history.append({
                'timestamp': time.time(),
                'user_request': user_request,
                'result': result,
                'success': result.get('success', False)
            })
            self.state.last_activity = time.time()
    
    async def _check_and_compact_context(self) -> None:
        """检查并压缩上下文"""
        if not self.config.enable_auto_compact or not self.state:
            return
        
        compacted = await self.compact_manager.check_and_compact(
            self.state.conversation_history,
            current_task=self.state.current_task
        )
        
        if compacted:
            self.state.compacted_contexts.append(compacted)
            logger.info(f"Context compacted: {compacted.get_compression_ratio():.1%} reduction")
    
    async def init_project_config(self, force: bool = False) -> bool:
        """初始化项目配置"""
        return self.config_manager.init_config(force=force)
    
    def get_session_stats(self) -> Dict[str, Any]:
        """获取会话统计"""
        if not self.state:
            return {'error': 'No active session'}
        
        return {
            'session_id': self.state.session_id,
            'duration_seconds': time.time() - self.state.created_at,
            'conversation_count': len(self.state.conversation_history),
            'compaction_count': len(self.state.compacted_contexts),
            'planner_stats': self.planner.get_statistics(),
            'subagent_stats': self.subagent_router.get_stats(),
            'compaction_stats': self.compact_manager.get_compaction_stats()
        }
    
    def get_capabilities(self) -> Dict[str, Any]:
        """获取 Agent 能力说明"""
        return {
            'task_types': [t.value for t in TaskType],
            'subagents': self.subagent_router.get_capabilities(),
            'hooks_enabled': self.config.enable_hooks,
            'auto_compact_enabled': self.config.enable_auto_compact,
            'max_tokens': self.config.max_tokens
        }
    
    async def stop(self) -> None:
        """停止 Agent"""
        if self.state:
            await self.hook_manager.trigger(
                HookType.ON_STOP,
                data={
                    'session_id': self.state.session_id,
                    'duration': time.time() - self.state.created_at
                }
            )
            
            logger.info(f"Session stopped: {self.state.session_id}")
            self.state = None


# 便捷函数：快速创建 Agent
async def create_agent(
    project_root: Optional[Union[str, Path]] = None,
    llm_client: Optional[Any] = None,
    tool_registry: Optional[Any] = None,
    config: Optional[AgentConfig] = None
) -> ClaudeCodeAgent:
    """快速创建 ClaudeCodeAgent"""
    # 如果没有提供 LLM 客户端，创建一个模拟的
    if llm_client is None:
        from unittest.mock import Mock
        llm_client = Mock()
        llm_client.generate = lambda prompt: "{}"
    
    # 如果没有提供工具注册表，创建一个模拟的
    if tool_registry is None:
        from unittest.mock import Mock
        tool_registry = Mock()
    
    agent = ClaudeCodeAgent(
        llm_client=llm_client,
        tool_registry=tool_registry,
        config=config,
        project_root=project_root
    )
    
    await agent.start_session()
    return agent
