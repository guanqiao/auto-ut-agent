# 参考 Claude Code 核心能力的改进计划

## 一、Claude Code 核心设计哲学深度解析

### 1.1 核心架构理念

基于对 Claude Code 的深入研究，其核心设计哲学可以总结为以下几个关键原则：

#### 1.1.1 "计划-执行闭环" (Plan-Execute Loop)

```
用户输入需求 → Agent 分析并生成执行计划 → 逐步执行（编辑文件/运行命令/调用 API）
    ↑                                                              ↓
    ←———————————— 反馈观察结果（测试输出/报错信息/LSP 诊断） ←——————————
```

Claude Code 不再只是"建议"你该写什么代码，而是**接管整个开发流程**：理解需求、制定计划、编写代码、运行测试、修复 bug、提交 PR。开发者的角色从"代码的编写者"变成了"任务的指挥者"。

#### 1.1.2 工具生态的"连接-使用"分离

| 组件 | 职责 | 类比 |
|------|------|------|
| **MCP** | 工具连接的"桥梁" | 解决"工具如何接入" |
| **Skills** | 工具使用的"说明书" | 解决"如何正确使用工具" |

Skills 不是"直接调用工具然后祈祷它别乱用"，而是"教 Claude 如何正确使用 MCP 工具的使用说明书"。

#### 1.1.3 分层上下文管理

Claude Code 采用三层上下文管理策略：

1. **实时环境感知 (LSP 集成)**：AI 改完代码后立即获得诊断反馈
2. **持久化会话存储 (SQLite)**：跨会话记忆，断点续传
3. **结构化上下文管理 (编排者-子代理模式)**：复杂任务由子代理在独立会话中执行

### 1.2 关键创新机制

#### 1.2.1 Subagents 子代理系统

内置三大核心代理类型：
- **Bash 代理**：专注执行命令行相关任务
- **Plan 代理**：负责设计清晰的项目实现方案
- **Explore 代理**：快速遍历和分析代码库结构

#### 1.2.2 Hooks 钩子系统

在特定生命周期事件中注入自定义逻辑：
- `UserPromptSubmit`：用户提交提示后触发
- `PreToolUse`：工具执行前触发
- `PostToolUse`：工具执行后触发
- `Stop`：Claude 停止响应时触发

#### 1.2.3 CLAUDE.md 项目配置

通过 `/init` 命令自动生成，包含：
- 项目类型与技术栈
- 编码规范与偏好
- 常用命令与工作流
- 项目特定上下文信息

---

## 二、PyUT Agent 与 Claude Code 的能力对比

### 2.1 已具备的优势

| 能力 | PyUT Agent | Claude Code |
|------|------------|-------------|
| **UT 生成专业性** | ✅ 专业深度 | ⚠️ 通用能力 |
| **事件驱动架构** | ✅ 完整实现 | ❌ 未明确 |
| **多级缓存** | ✅ L1+L2 | ❌ 未明确 |
| **错误预测** | ✅ 12种类型 | ❌ 未明确 |
| **自适应策略** | ✅ ε-贪婪 | ❌ 未明确 |
| **覆盖率分析** | ✅ JaCoCo集成 | ⚠️ 基础支持 |

### 2.2 需要补齐的差距

| 能力维度 | Claude Code | PyUT Agent | 差距级别 | 优先级 |
|----------|-------------|------------|----------|--------|
| **通用任务规划** | 任意编程任务 | 仅 UT 生成 | 🔴 重大 | P0 |
| **Skills 生态** | 丰富的技能市场 | 基础 Skills | 🟡 中等 | P1 |
| **Hooks 系统** | 完整生命周期钩子 | 无 | 🟡 中等 | P1 |
| **项目配置 (CLAUDE.md)** | 自动初始化 | 无 | 🟡 中等 | P1 |
| **Subagents 分工** | Bash/Plan/Explore | 基础子代理 | 🟡 中等 | P1 |
| **Cowork 模式** | 非技术人员协作 | 无 | 🟢 轻微 | P2 |
| **浏览器控制** | Web 自动化 | 无 | 🟢 轻微 | P2 |

---

## 三、改进计划

### Phase 1: 通用任务规划能力（P0 - 1-2个月）

#### 3.1.1 目标

让 PyUT Agent 从"只能生成单元测试"进化为"能处理任意编程任务"的通用 Coding Agent。

#### 3.1.2 核心实现

```python
# pyutagent/agent/universal_planner.py

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Callable
from enum import Enum
import json

class TaskType(Enum):
    """任务类型枚举"""
    TEST_GENERATION = "test_generation"      # 生成单元测试
    CODE_REFACTORING = "code_refactoring"    # 代码重构
    BUG_FIX = "bug_fix"                      # Bug 修复
    FEATURE_ADD = "feature_add"              # 添加新功能
    CODE_REVIEW = "code_review"              # 代码审查
    DOCUMENTATION = "documentation"          # 文档生成
    DEPENDENCY_UPDATE = "dependency_update"  # 依赖更新
    QUERY = "query"                          # 问题查询

@dataclass
class TaskUnderstanding:
    """任务理解结果"""
    task_type: TaskType
    description: str
    target_files: List[str]
    constraints: List[str]
    success_criteria: List[str]
    estimated_complexity: int  # 1-5

@dataclass
class Subtask:
    """子任务"""
    id: str
    description: str
    task_type: TaskType
    dependencies: List[str]
    tools_needed: List[str]
    estimated_complexity: int
    success_criteria: str

@dataclass
class ExecutionPlan:
    """执行计划"""
    task_id: str
    original_request: str
    understanding: TaskUnderstanding
    subtasks: List[Subtask]
    execution_order: List[str]
    rollback_strategy: Optional[str] = None


class UniversalTaskPlanner:
    """通用任务规划器
    
    参考 Claude Code 的任务理解能力，实现：
    1. 理解任意编程需求
    2. 自动分解任务
    3. 动态调整计划
    """
    
    def __init__(
        self,
        llm_client: Any,
        project_analyzer: Any,
        tool_registry: Any
    ):
        self.llm = llm_client
        self.analyzer = project_analyzer
        self.tools = tool_registry
        self._task_handlers: Dict[TaskType, Callable] = {}
    
    async def understand_task(
        self,
        user_request: str,
        project_context: Dict[str, Any]
    ) -> TaskUnderstanding:
        """理解用户任务
        
        使用 LLM 分析用户请求，确定任务类型和关键信息。
        """
        prompt = f"""
        分析以下用户编程请求，提取关键信息：
        
        用户请求：{user_request}
        
        项目上下文：
        - 语言：{project_context.get('language', 'Unknown')}
        - 构建工具：{project_context.get('build_tool', 'Unknown')}
        - 项目结构：{json.dumps(project_context.get('structure', {}), indent=2)}
        
        请分析并返回 JSON 格式：
        {{
            "task_type": "test_generation|code_refactoring|bug_fix|feature_add|code_review|documentation|dependency_update|query",
            "description": "任务描述",
            "target_files": ["目标文件列表"],
            "constraints": ["约束条件"],
            "success_criteria": ["成功标准"],
            "estimated_complexity": 1-5
        }}
        """
        
        response = await self.llm.generate(prompt)
        data = json.loads(response)
        
        return TaskUnderstanding(
            task_type=TaskType(data['task_type']),
            description=data['description'],
            target_files=data['target_files'],
            constraints=data['constraints'],
            success_criteria=data['success_criteria'],
            estimated_complexity=data['estimated_complexity']
        )
    
    async def decompose_task(
        self,
        understanding: TaskUnderstanding,
        project_context: Dict[str, Any]
    ) -> ExecutionPlan:
        """分解任务为可执行子任务"""
        
        # 根据任务类型选择分解策略
        if understanding.task_type == TaskType.TEST_GENERATION:
            return await self._decompose_test_generation(understanding, project_context)
        elif understanding.task_type == TaskType.CODE_REFACTORING:
            return await self._decompose_refactoring(understanding, project_context)
        elif understanding.task_type == TaskType.BUG_FIX:
            return await self._decompose_bug_fix(understanding, project_context)
        elif understanding.task_type == TaskType.FEATURE_ADD:
            return await self._decompose_feature_add(understanding, project_context)
        else:
            return await self._decompose_generic(understanding, project_context)
    
    async def _decompose_test_generation(
        self,
        understanding: TaskUnderstanding,
        project_context: Dict[str, Any]
    ) -> ExecutionPlan:
        """分解测试生成任务"""
        subtasks = [
            Subtask(
                id="analyze_target",
                description=f"分析目标类：{understanding.target_files[0] if understanding.target_files else 'Unknown'}",
                task_type=TaskType.QUERY,
                dependencies=[],
                tools_needed=["file_read", "java_parser"],
                estimated_complexity=2,
                success_criteria="获取目标类的完整结构和方法列表"
            ),
            Subtask(
                id="analyze_dependencies",
                description="分析依赖关系和 mocking 需求",
                task_type=TaskType.QUERY,
                dependencies=["analyze_target"],
                tools_needed=["dependency_analyzer", "java_parser"],
                estimated_complexity=2,
                success_criteria="识别所有需要 mock 的依赖"
            ),
            Subtask(
                id="generate_test",
                description="生成单元测试代码",
                task_type=TaskType.TEST_GENERATION,
                dependencies=["analyze_target", "analyze_dependencies"],
                tools_needed=["test_generator", "file_write"],
                estimated_complexity=3,
                success_criteria="生成可编译的测试代码"
            ),
            Subtask(
                id="compile_test",
                description="编译测试代码",
                task_type=TaskType.QUERY,
                dependencies=["generate_test"],
                tools_needed=["maven", "compilation_handler"],
                estimated_complexity=2,
                success_criteria="测试代码编译通过"
            ),
            Subtask(
                id="run_test",
                description="运行测试并分析结果",
                task_type=TaskType.QUERY,
                dependencies=["compile_test"],
                tools_needed=["test_runner", "coverage_analyzer"],
                estimated_complexity=2,
                success_criteria="测试运行通过或获得详细错误信息"
            ),
            Subtask(
                id="fix_issues",
                description="修复测试中的问题",
                task_type=TaskType.BUG_FIX,
                dependencies=["run_test"],
                tools_needed=["incremental_fixer", "file_edit"],
                estimated_complexity=3,
                success_criteria="所有测试通过"
            )
        ]
        
        return ExecutionPlan(
            task_id=f"task_{hash(understanding.description) % 10000}",
            original_request=understanding.description,
            understanding=understanding,
            subtasks=subtasks,
            execution_order=["analyze_target", "analyze_dependencies", "generate_test", 
                           "compile_test", "run_test", "fix_issues"]
        )
    
    async def execute_with_feedback(
        self,
        plan: ExecutionPlan,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """执行计划并动态调整
        
        参考 Claude Code 的闭环执行模式：
        1. 执行子任务
        2. 观察结果
        3. 根据反馈调整计划
        4. 继续执行或终止
        """
        results = []
        completed_subtasks = set()
        
        for subtask_id in plan.execution_order:
            subtask = self._find_subtask(plan, subtask_id)
            
            # 检查依赖是否满足
            if not all(dep in completed_subtasks for dep in subtask.dependencies):
                # 重新排序执行顺序
                await self._reorder_execution(plan, completed_subtasks)
                continue
            
            # 执行子任务
            result = await self._execute_subtask(subtask)
            results.append({
                'subtask_id': subtask_id,
                'result': result,
                'success': result.get('success', False)
            })
            
            # 回调进度
            if progress_callback:
                await progress_callback(subtask, result)
            
            # 根据结果调整计划
            if not result.get('success', False):
                adjustment = await self._adjust_plan(plan, subtask, result)
                if adjustment:
                    # 插入新的子任务或修改现有任务
                    plan = self._apply_adjustment(plan, adjustment)
            
            completed_subtasks.add(subtask_id)
        
        return {
            'success': all(r['success'] for r in results),
            'results': results,
            'plan': plan
        }
    
    def register_task_handler(
        self,
        task_type: TaskType,
        handler: Callable
    ) -> None:
        """注册任务类型处理器"""
        self._task_handlers[task_type] = handler
    
    async def _execute_subtask(self, subtask: Subtask) -> Dict[str, Any]:
        """执行单个子任务"""
        handler = self._task_handlers.get(subtask.task_type)
        if handler:
            return await handler(subtask)
        return {'success': False, 'error': f'No handler for task type: {subtask.task_type}'}
    
    def _find_subtask(self, plan: ExecutionPlan, subtask_id: str) -> Subtask:
        """查找子任务"""
        for subtask in plan.subtasks:
            if subtask.id == subtask_id:
                return subtask
        raise ValueError(f"Subtask not found: {subtask_id}")
    
    async def _adjust_plan(
        self,
        plan: ExecutionPlan,
        failed_subtask: Subtask,
        result: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """根据失败结果调整计划"""
        # 使用 LLM 分析失败原因并建议调整
        prompt = f"""
        子任务执行失败，请分析原因并建议调整：
        
        失败子任务：{failed_subtask.description}
        错误信息：{result.get('error', 'Unknown')}
        
        当前计划：
        {json.dumps([s.description for s in plan.subtasks], indent=2)}
        
        建议如何调整计划以解决问题？
        """
        
        response = await self.llm.generate(prompt)
        # 解析调整建议
        return json.loads(response) if response else None
    
    def _apply_adjustment(
        self,
        plan: ExecutionPlan,
        adjustment: Dict[str, Any]
    ) -> ExecutionPlan:
        """应用计划调整"""
        # 实现计划调整逻辑
        return plan
    
    async def _reorder_execution(
        self,
        plan: ExecutionPlan,
        completed: set
    ) -> None:
        """重新排序执行顺序"""
        # 拓扑排序确保依赖顺序
        pass
```

#### 3.1.3 任务分解策略矩阵

| 任务类型 | 典型子任务 | 关键工具 |
|----------|------------|----------|
| **测试生成** | 分析目标→分析依赖→生成→编译→运行→修复 | java_parser, test_generator, maven |
| **代码重构** | 分析影响→制定计划→执行重构→验证→清理 | refactoring_engine, semantic_analyzer |
| **Bug 修复** | 复现→定位→修复→验证→回归测试 | debug_tools, error_analyzer |
| **功能添加** | 需求分析→设计→实现→测试→文档 | architect_editor, test_generator |
| **代码审查** | 读取代码→分析→生成报告→建议 | code_reviewer, quality_analyzer |

---

### Phase 2: Hooks 生命周期系统（P1 - 1个月）

#### 3.2.1 目标

实现类似 Claude Code Hooks 的生命周期钩子系统，允许在特定事件点注入自定义逻辑。

#### 3.2.2 核心实现

```python
# pyutagent/core/hooks.py

from dataclasses import dataclass
from typing import Dict, List, Callable, Any, Optional
from enum import Enum, auto
import asyncio
import logging

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


@dataclass
class HookContext:
    """钩子上下文"""
    hook_type: HookType
    data: Dict[str, Any]
    metadata: Dict[str, Any]
    
    def get(self, key: str, default=None):
        return self.data.get(key, default)


@dataclass
class HookResult:
    """钩子执行结果"""
    success: bool
    data: Dict[str, Any]
    should_abort: bool = False
    modified_context: Optional[Dict[str, Any]] = None


class Hook:
    """钩子定义"""
    
    def __init__(
        self,
        name: str,
        hook_type: HookType,
        handler: Callable[[HookContext], HookResult],
        priority: int = 0,
        condition: Optional[Callable[[HookContext], bool]] = None
    ):
        self.name = name
        self.hook_type = hook_type
        self.handler = handler
        self.priority = priority
        self.condition = condition
    
    async def execute(self, context: HookContext) -> HookResult:
        """执行钩子"""
        # 检查条件
        if self.condition and not self.condition(context):
            return HookResult(success=True, data={})
        
        try:
            result = await self.handler(context) if asyncio.iscoroutinefunction(self.handler) else self.handler(context)
            return result
        except Exception as e:
            logger.error(f"Hook {self.name} failed: {e}")
            return HookResult(success=False, data={'error': str(e)})


class HookRegistry:
    """钩子注册表"""
    
    def __init__(self):
        self._hooks: Dict[HookType, List[Hook]] = {hook_type: [] for hook_type in HookType}
    
    def register(self, hook: Hook) -> None:
        """注册钩子"""
        self._hooks[hook.hook_type].append(hook)
        # 按优先级排序
        self._hooks[hook.hook_type].sort(key=lambda h: h.priority, reverse=True)
        logger.info(f"Registered hook: {hook.name} for {hook.hook_type.name}")
    
    def unregister(self, hook_name: str) -> bool:
        """注销钩子"""
        for hooks in self._hooks.values():
            for i, hook in enumerate(hooks):
                if hook.name == hook_name:
                    hooks.pop(i)
                    logger.info(f"Unregistered hook: {hook_name}")
                    return True
        return False
    
    async def execute_hooks(
        self,
        hook_type: HookType,
        context: HookContext
    ) -> HookResult:
        """执行指定类型的所有钩子"""
        hooks = self._hooks.get(hook_type, [])
        combined_data = {}
        
        for hook in hooks:
            result = await hook.execute(context)
            combined_data.update(result.data)
            
            if result.should_abort:
                logger.info(f"Hook {hook.name} requested abort")
                return HookResult(
                    success=result.success,
                    data=combined_data,
                    should_abort=True
                )
            
            if result.modified_context:
                context.data.update(result.modified_context)
        
        return HookResult(success=True, data=combined_data)


class HookManager:
    """钩子管理器"""
    
    def __init__(self):
        self.registry = HookRegistry()
        self._builtin_hooks_registered = False
    
    def register_builtin_hooks(self) -> None:
        """注册内置钩子"""
        if self._builtin_hooks_registered:
            return
        
        # 1. 代码格式化钩子
        self.registry.register(Hook(
            name="auto_format",
            hook_type=HookType.POST_TOOL_USE,
            handler=self._auto_format_handler,
            priority=10,
            condition=lambda ctx: ctx.get('tool_name') == 'file_write' and 
                                 ctx.get('file_path', '').endswith('.java')
        ))
        
        # 2. 操作日志钩子
        self.registry.register(Hook(
            name="operation_logger",
            hook_type=HookType.POST_TOOL_USE,
            handler=self._operation_log_handler,
            priority=5
        ))
        
        # 3. 敏感操作确认钩子
        self.registry.register(Hook(
            name="sensitive_operation_confirm",
            hook_type=HookType.PRE_TOOL_USE,
            handler=self._sensitive_operation_handler,
            priority=100,
            condition=lambda ctx: ctx.get('tool_name') in ['file_delete', 'git_push', 'mvn_deploy']
        ))
        
        # 4. 错误恢复钩子
        self.registry.register(Hook(
            name="error_recovery",
            hook_type=HookType.ON_ERROR,
            handler=self._error_recovery_handler,
            priority=50
        ))
        
        self._builtin_hooks_registered = True
    
    def _auto_format_handler(self, context: HookContext) -> HookResult:
        """自动格式化代码"""
        file_path = context.get('file_path')
        if file_path:
            # 调用代码格式化工具
            logger.info(f"Auto-formatting: {file_path}")
            # 实际实现...
        return HookResult(success=True, data={'formatted': True})
    
    def _operation_log_handler(self, context: HookContext) -> HookResult:
        """记录操作日志"""
        tool_name = context.get('tool_name')
        logger.info(f"Tool executed: {tool_name}")
        return HookResult(success=True, data={'logged': True})
    
    def _sensitive_operation_handler(self, context: HookContext) -> HookResult:
        """敏感操作确认"""
        tool_name = context.get('tool_name')
        # 这里可以实现用户确认逻辑
        logger.warning(f"Sensitive operation detected: {tool_name}")
        return HookResult(success=True, data={'confirmed': True})
    
    def _error_recovery_handler(self, context: HookContext) -> HookResult:
        """错误恢复处理"""
        error = context.get('error')
        logger.error(f"Error occurred: {error}")
        return HookResult(success=True, data={'error_logged': True})
    
    async def trigger(
        self,
        hook_type: HookType,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> HookResult:
        """触发钩子"""
        context = HookContext(
            hook_type=hook_type,
            data=data,
            metadata=metadata or {}
        )
        return await self.registry.execute_hooks(hook_type, context)


# 装饰器方式注册钩子
def hook(
    hook_type: HookType,
    priority: int = 0,
    condition: Optional[Callable] = None
):
    """钩子装饰器"""
    def decorator(func: Callable) -> Callable:
        func._hook_type = hook_type
        func._hook_priority = priority
        func._hook_condition = condition
        return func
    return decorator


# 使用示例
class CustomHooks:
    """自定义钩子示例"""
    
    @hook(HookType.USER_PROMPT_SUBMIT, priority=10)
    def preprocess_prompt(self, context: HookContext) -> HookResult:
        """预处理用户输入"""
        prompt = context.get('prompt', '')
        # 添加项目特定的前缀
        modified = f"[Project Context] {prompt}"
        return HookResult(
            success=True,
            data={'prompt': modified},
            modified_context={'prompt': modified}
        )
    
    @hook(HookType.POST_SUBTASK, priority=5)
    def log_subtask(self, context: HookContext) -> HookResult:
        """记录子任务执行"""
        subtask_id = context.get('subtask_id')
        success = context.get('success')
        logger.info(f"Subtask {subtask_id} completed: {success}")
        return HookResult(success=True, data={})
```

---

### Phase 3: 项目配置系统（P1 - 1个月）

#### 3.3.1 目标

实现类似 CLAUDE.md 的项目配置系统，让 Agent 能快速理解项目上下文。

#### 3.3.2 核心实现

```python
# pyutagent/core/project_config.py

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import yaml


@dataclass
class BuildCommands:
    """构建命令配置"""
    build: str = "mvn compile"
    test: str = "mvn test"
    test_single: str = "mvn test -Dtest={test_class}"
    coverage: str = "mvn jacoco:report"
    clean: str = "mvn clean"
    package: str = "mvn package"


@dataclass
class ProjectContext:
    """项目上下文配置"""
    name: str = ""
    description: str = ""
    language: str = "java"
    build_tool: str = "maven"
    java_version: str = "17"
    test_framework: str = "junit5"
    mock_framework: str = "mockito"
    
    # 编码规范
    coding_standards: List[str] = field(default_factory=list)
    naming_conventions: Dict[str, str] = field(default_factory=dict)
    
    # 构建命令
    build_commands: BuildCommands = field(default_factory=BuildCommands)
    
    # 常用工作流
    common_workflows: Dict[str, str] = field(default_factory=dict)
    
    # 项目特定信息
    architecture: str = ""
    key_modules: List[str] = field(default_factory=list)
    external_dependencies: List[str] = field(default_factory=list)
    
    # 测试配置
    test_patterns: List[str] = field(default_factory=lambda: ["**/*Test.java"])
    exclude_patterns: List[str] = field(default_factory=list)
    coverage_threshold: float = 0.8


class ProjectConfigManager:
    """项目配置管理器
    
    管理 PYUT.md 配置文件，类似于 Claude Code 的 CLAUDE.md
    """
    
    CONFIG_FILENAME = "PYUT.md"
    CONFIG_JSON = ".pyutagent/config.json"
    
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.config_path = self.project_root / self.CONFIG_FILENAME
        self.json_config_path = self.project_root / self.CONFIG_JSON
        self._context: Optional[ProjectContext] = None
    
    def init_config(self, force: bool = False) -> bool:
        """初始化项目配置文件
        
        类似于 Claude Code 的 /init 命令
        """
        if self.config_path.exists() and not force:
            print(f"Config already exists: {self.config_path}")
            return False
        
        # 分析项目结构
        context = self._analyze_project()
        
        # 生成配置文件
        self._write_config(context)
        
        print(f"Created config: {self.config_path}")
        return True
    
    def _analyze_project(self) -> ProjectContext:
        """分析项目结构"""
        context = ProjectContext(
            name=self.project_root.name
        )
        
        # 检测构建工具
        if (self.project_root / "pom.xml").exists():
            context.build_tool = "maven"
        elif (self.project_root / "build.gradle").exists():
            context.build_tool = "gradle"
        elif (self.project_root / "BUILD").exists():
            context.build_tool = "bazel"
        
        # 检测 Java 版本
        pom_path = self.project_root / "pom.xml"
        if pom_path.exists():
            context.java_version = self._extract_java_version(pom_path)
        
        # 检测测试框架
        context.test_framework = self._detect_test_framework()
        context.mock_framework = self._detect_mock_framework()
        
        # 分析项目结构
        context.key_modules = self._detect_modules()
        
        return context
    
    def _extract_java_version(self, pom_path: Path) -> str:
        """从 pom.xml 提取 Java 版本"""
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(pom_path)
            root = tree.getroot()
            
            # 查找 java.version 或 maven.compiler.source
            ns = {'m': 'http://maven.apache.org/POM/4.0.0'}
            for prop in root.findall('.//m:properties/*', ns):
                if prop.tag.endswith('java.version') or prop.tag.endswith('maven.compiler.source'):
                    return prop.text or "17"
        except Exception:
            pass
        return "17"
    
    def _detect_test_framework(self) -> str:
        """检测测试框架"""
        pom_path = self.project_root / "pom.xml"
        if pom_path.exists():
            content = pom_path.read_text()
            if 'junit-jupiter' in content or 'junit5' in content:
                return "junit5"
            elif 'junit' in content:
                return "junit4"
            elif 'testng' in content:
                return "testng"
        return "junit5"
    
    def _detect_mock_framework(self) -> str:
        """检测 Mock 框架"""
        pom_path = self.project_root / "pom.xml"
        if pom_path.exists():
            content = pom_path.read_text()
            if 'mockito' in content:
                return "mockito"
            elif 'easymock' in content:
                return "easymock"
            elif 'jmock' in content:
                return "jmock"
        return "mockito"
    
    def _detect_modules(self) -> List[str]:
        """检测项目模块"""
        modules = []
        src_main = self.project_root / "src" / "main" / "java"
        if src_main.exists():
            for item in src_main.iterdir():
                if item.is_dir():
                    modules.append(item.name)
        return modules[:5]  # 最多返回5个
    
    def _write_config(self, context: ProjectContext) -> None:
        """写入配置文件"""
        config_content = self._generate_md_config(context)
        self.config_path.write_text(config_content, encoding='utf-8')
        
        # 同时写入 JSON 格式便于程序读取
        self.json_config_path.parent.mkdir(parents=True, exist_ok=True)
        self.json_config_path.write_text(
            json.dumps(self._context_to_dict(context), indent=2),
            encoding='utf-8'
        )
    
    def _generate_md_config(self, context: ProjectContext) -> str:
        """生成 Markdown 格式配置"""
        return f"""# PyUT Agent Configuration

## Project Context

When working with this codebase, prioritize readability over cleverness.
Ask clarifying questions when requirements are ambiguous.

### Project Information
- **Name**: {context.name}
- **Language**: {context.language}
- **Build Tool**: {context.build_tool}
- **Java Version**: {context.java_version}
- **Test Framework**: {context.test_framework}
- **Mock Framework**: {context.mock_framework}

### Architecture
{context.architecture or "Standard Maven/Gradle project structure"}

### Key Modules
{chr(10).join(f"- {m}" for m in context.key_modules) or "- Main source module"}

## Build Commands

```bash
# Build the project
{context.build_commands.build}

# Run all tests
{context.build_commands.test}

# Run single test class
{context.build_commands.test_single}

# Generate coverage report
{context.build_commands.coverage}

# Clean build artifacts
{context.build_commands.clean}
```

## Coding Standards

- Follow existing code style in the project
- Use meaningful variable and method names
- Add Javadoc for public APIs
- Keep methods focused and small
- Write unit tests for new functionality

## Test Generation Preferences

- Use {context.test_framework} for all new tests
- Mock external dependencies with {context.mock_framework}
- Target {context.coverage_threshold*100:.0f}% code coverage
- Include positive and negative test cases
- Use descriptive test method names

## Common Workflows

### Generate tests for a class
```
Generate unit tests for UserService
```

### Fix compilation errors
```
Fix compilation errors in OrderServiceTest
```

### Improve coverage
```
Improve test coverage for payment module
```
"""
    
    def _context_to_dict(self, context: ProjectContext) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'name': context.name,
            'description': context.description,
            'language': context.language,
            'build_tool': context.build_tool,
            'java_version': context.java_version,
            'test_framework': context.test_framework,
            'mock_framework': context.mock_framework,
            'coding_standards': context.coding_standards,
            'build_commands': {
                'build': context.build_commands.build,
                'test': context.build_commands.test,
                'test_single': context.build_commands.test_single,
                'coverage': context.build_commands.coverage,
            },
            'key_modules': context.key_modules,
            'test_patterns': context.test_patterns,
            'coverage_threshold': context.coverage_threshold,
        }
    
    def load_context(self) -> Optional[ProjectContext]:
        """加载项目上下文"""
        if self._context:
            return self._context
        
        # 优先读取 JSON 配置
        if self.json_config_path.exists():
            try:
                data = json.loads(self.json_config_path.read_text())
                self._context = self._dict_to_context(data)
                return self._context
            except Exception:
                pass
        
        # 回退到分析项目
        self._context = self._analyze_project()
        return self._context
    
    def _dict_to_context(self, data: Dict[str, Any]) -> ProjectContext:
        """从字典创建上下文"""
        build_cmds = data.get('build_commands', {})
        return ProjectContext(
            name=data.get('name', ''),
            description=data.get('description', ''),
            language=data.get('language', 'java'),
            build_tool=data.get('build_tool', 'maven'),
            java_version=data.get('java_version', '17'),
            test_framework=data.get('test_framework', 'junit5'),
            mock_framework=data.get('mock_framework', 'mockito'),
            coding_standards=data.get('coding_standards', []),
            build_commands=BuildCommands(
                build=build_cmds.get('build', 'mvn compile'),
                test=build_cmds.get('test', 'mvn test'),
                test_single=build_cmds.get('test_single', 'mvn test -Dtest={test_class}'),
                coverage=build_cmds.get('coverage', 'mvn jacoco:report'),
            ),
            key_modules=data.get('key_modules', []),
            test_patterns=data.get('test_patterns', ['**/*Test.java']),
            coverage_threshold=data.get('coverage_threshold', 0.8),
        )
    
    def get_prompt_context(self) -> str:
        """获取用于 LLM prompt 的上下文"""
        context = self.load_context()
        if not context:
            return ""
        
        return f"""
Project Context:
- Name: {context.name}
- Language: {context.language}
- Build Tool: {context.build_tool}
- Java Version: {context.java_version}
- Test Framework: {context.test_framework}
- Mock Framework: {context.mock_framework}

Build Commands:
- Build: {context.build_commands.build}
- Test: {context.build_commands.test}
- Coverage: {context.build_commands.coverage}

Key Modules: {', '.join(context.key_modules)}
"""
```

---

### Phase 4: 专业化 Subagents（P1 - 1个月）

#### 3.4.1 目标

实现类似 Claude Code 的专业化子代理系统，将复杂任务分配给专门的代理处理。

#### 3.4.2 核心实现

```python
# pyutagent/agent/subagents/specialized.py

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
import asyncio


@dataclass
class SubagentResult:
    """子代理执行结果"""
    success: bool
    data: Dict[str, Any]
    summary: str
    artifacts: List[str] = None


class SpecializedSubagent(ABC):
    """专业化子代理基类
    
    参考 Claude Code 的 Subagents 设计：
    - BashAgent: 专注命令行任务
    - PlanAgent: 专注方案设计
    - ExploreAgent: 专注代码库探索
    """
    
    def __init__(
        self,
        name: str,
        llm_client: Any,
        tool_registry: Any
    ):
        self.name = name
        self.llm = llm_client
        self.tools = tool_registry
    
    @abstractmethod
    async def execute(self, task: str, context: Dict[str, Any]) -> SubagentResult:
        """执行任务"""
        pass
    
    @abstractmethod
    def can_handle(self, task: str) -> float:
        """判断是否能处理该任务（返回置信度 0-1）"""
        pass


class BashSubagent(SpecializedSubagent):
    """Bash 子代理
    
    专注执行命令行相关任务
    """
    
    def __init__(self, llm_client: Any, tool_registry: Any):
        super().__init__("BashAgent", llm_client, tool_registry)
    
    def can_handle(self, task: str) -> float:
        """判断是否为命令行任务"""
        bash_keywords = [
            'run', 'execute', 'command', 'shell', 'bash',
            'mvn', 'gradle', 'git', 'npm', 'docker',
            'build', 'test', 'compile', 'package'
        ]
        task_lower = task.lower()
        score = sum(1 for kw in bash_keywords if kw in task_lower) / len(bash_keywords)
        return min(score * 3, 1.0)  # 放大匹配度
    
    async def execute(self, task: str, context: Dict[str, Any]) -> SubagentResult:
        """执行命令行任务"""
        # 解析任务，提取命令
        prompt = f"""
        将以下任务转换为可执行的 shell 命令：
        
        任务：{task}
        项目目录：{context.get('project_root', '.')}
        构建工具：{context.get('build_tool', 'maven')}
        
        请返回 JSON 格式：
        {{
            "commands": ["命令1", "命令2"],
            "description": "命令描述",
            "expected_output": "预期输出"
        }}
        """
        
        response = await self.llm.generate(prompt)
        # 解析并执行命令
        # ...
        
        return SubagentResult(
            success=True,
            data={'commands': []},
            summary=f"Executed bash commands for: {task}"
        )


class PlanSubagent(SpecializedSubagent):
    """Plan 子代理
    
    专注设计清晰的项目实现方案
    """
    
    def __init__(self, llm_client: Any, tool_registry: Any):
        super().__init__("PlanAgent", llm_client, tool_registry)
    
    def can_handle(self, task: str) -> float:
        """判断是否为规划任务"""
        plan_keywords = [
            'plan', 'design', 'architecture', 'strategy',
            'how to', 'approach', 'solution', 'implement',
            'refactor', 'restructure', 'organize'
        ]
        task_lower = task.lower()
        score = sum(1 for kw in plan_keywords if kw in task_lower) / len(plan_keywords)
        return min(score * 3, 1.0)
    
    async def execute(self, task: str, context: Dict[str, Any]) -> SubagentResult:
        """设计方案"""
        prompt = f"""
        为以下任务设计详细的实现方案：
        
        任务：{task}
        项目信息：{context.get('project_info', {})}
        
        请提供：
        1. 实现步骤（按优先级排序）
        2. 涉及的文件和模块
        3. 潜在风险和注意事项
        4. 验证方法
        """
        
        plan = await self.llm.generate(prompt)
        
        return SubagentResult(
            success=True,
            data={'plan': plan},
            summary=f"Created implementation plan for: {task}",
            artifacts=['plan.md']
        )


class ExploreSubagent(SpecializedSubagent):
    """Explore 子代理
    
    专注快速遍历和分析代码库结构
    """
    
    def __init__(self, llm_client: Any, tool_registry: Any):
        super().__init__("ExploreAgent", llm_client, tool_registry)
    
    def can_handle(self, task: str) -> float:
        """判断是否为探索任务"""
        explore_keywords = [
            'find', 'search', 'locate', 'explore',
            'where is', 'how is', 'what is', 'understand',
            'structure', 'organization', 'dependencies'
        ]
        task_lower = task.lower()
        score = sum(1 for kw in explore_keywords if kw in task_lower) / len(explore_keywords)
        return min(score * 3, 1.0)
    
    async def execute(self, task: str, context: Dict[str, Any]) -> SubagentResult:
        """探索代码库"""
        # 使用工具搜索代码库
        project_root = context.get('project_root', '.')
        
        # 执行探索
        findings = await self._explore_project(project_root, task)
        
        return SubagentResult(
            success=True,
            data={'findings': findings},
            summary=f"Explored codebase for: {task}",
            artifacts=['exploration_report.md']
        )
    
    async def _explore_project(self, root: str, query: str) -> Dict[str, Any]:
        """探索项目结构"""
        # 实现探索逻辑
        return {}


class TestGenSubagent(SpecializedSubagent):
    """测试生成子代理
    
    PyUT Agent 的核心专业能力
    """
    
    def __init__(self, llm_client: Any, tool_registry: Any):
        super().__init__("TestGenAgent", llm_client, tool_registry)
    
    def can_handle(self, task: str) -> float:
        """判断是否为测试生成任务"""
        test_keywords = [
            'test', 'unit test', 'generate test', 'create test',
            'test coverage', 'junit', 'mock', 'assert'
        ]
        task_lower = task.lower()
        score = sum(1 for kw in test_keywords if kw in task_lower) / len(test_keywords)
        return min(score * 3, 1.0)
    
    async def execute(self, task: str, context: Dict[str, Any]) -> SubagentResult:
        """生成测试"""
        # 使用现有的测试生成能力
        from ..test_generator import TestGenerator
        
        generator = TestGenerator(self.llm, self.tools)
        result = await generator.generate(task, context)
        
        return SubagentResult(
            success=result.success,
            data={'tests': result.tests},
            summary=f"Generated tests for: {task}",
            artifacts=result.test_files
        )


class SubagentRouter:
    """子代理路由器
    
    根据任务类型自动选择最合适的子代理
    """
    
    def __init__(self):
        self.subagents: List[SpecializedSubagent] = []
    
    def register(self, subagent: SpecializedSubagent) -> None:
        """注册子代理"""
        self.subagents.append(subagent)
    
    async def route(self, task: str, context: Dict[str, Any]) -> SubagentResult:
        """路由任务到合适的子代理"""
        # 评估每个子代理的匹配度
        scores = [
            (subagent, subagent.can_handle(task))
            for subagent in self.subagents
        ]
        
        # 按匹配度排序
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # 选择最佳匹配
        if scores and scores[0][1] > 0.3:
            best_agent = scores[0][0]
            return await best_agent.execute(task, context)
        
        # 没有合适的子代理，使用默认处理
        return SubagentResult(
            success=False,
            data={},
            summary="No suitable subagent found"
        )
    
    def get_capabilities(self) -> Dict[str, str]:
        """获取所有子代理的能力描述"""
        return {
            agent.name: agent.__doc__ or "No description"
            for agent in self.subagents
        }
```

---

### Phase 5: 智能上下文压缩（P2 - 1个月）

#### 3.5.1 目标

实现类似 OpenCode 的 Auto Compact 机制，在上下文窗口即将耗尽时自动压缩历史。

#### 3.5.2 核心实现

```python
# pyutagent/core/context_compactor.py

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import json


@dataclass
class CompactedContext:
    """压缩后的上下文"""
    summary: str
    completed_tasks: List[str]
    current_focus: str
    pending_tasks: List[str]
    key_decisions: List[str]
    active_files: List[str]
    token_count: int


class ContextCompactor:
    """上下文压缩器
    
    参考 OpenCode 的 Auto Compact 机制：
    当 Token 使用达到阈值时，自动压缩历史对话
    """
    
    def __init__(
        self,
        llm_client: Any,
        threshold: float = 0.95,
        target_ratio: float = 0.3
    ):
        self.llm = llm_client
        self.threshold = threshold  # 触发压缩的阈值
        self.target_ratio = target_ratio  # 压缩后的目标比例
    
    def should_compact(
        self,
        current_tokens: int,
        max_tokens: int
    ) -> bool:
        """判断是否需要压缩"""
        return current_tokens / max_tokens >= self.threshold
    
    async def compact(
        self,
        conversation_history: List[Dict[str, Any]],
        current_task: Optional[str] = None
    ) -> CompactedContext:
        """压缩对话历史"""
        
        # 使用 LLM 生成摘要
        prompt = f"""
        请将以下对话历史压缩为结构化摘要：
        
        当前任务：{current_task or 'Unknown'}
        
        对话历史：
        {json.dumps(conversation_history, indent=2, ensure_ascii=False)[:8000]}
        
        请提取并返回 JSON 格式：
        {{
            "summary": "整体任务摘要（100字以内）",
            "completed_tasks": ["已完成的任务1", "任务2"],
            "current_focus": "当前正在进行的工作",
            "pending_tasks": ["待办事项1", "事项2"],
            "key_decisions": ["关键决策1", "决策2"],
            "active_files": ["活跃文件路径1", "路径2"]
        }}
        """
        
        response = await self.llm.generate(prompt)
        data = json.loads(response)
        
        return CompactedContext(
            summary=data['summary'],
            completed_tasks=data['completed_tasks'],
            current_focus=data['current_focus'],
            pending_tasks=data['pending_tasks'],
            key_decisions=data['key_decisions'],
            active_files=data['active_files'],
            token_count=self._estimate_tokens(data)
        )
    
    def _estimate_tokens(self, data: Dict[str, Any]) -> int:
        """估算 Token 数量"""
        text = json.dumps(data, ensure_ascii=False)
        # 粗略估算：1 token ≈ 4 字符
        return len(text) // 4
    
    def format_for_prompt(self, compacted: CompactedContext) -> str:
        """格式化为 Prompt 可用的上下文"""
        return f"""
[任务上下文摘要]
{compacted.summary}

[已完成]
{chr(10).join(f"- {t}" for t in compacted.completed_tasks)}

[当前焦点]
{compacted.current_focus}

[待办事项]
{chr(10).join(f"- {t}" for t in compacted.pending_tasks)}

[关键决策]
{chr(10).join(f"- {d}" for d in compacted.key_decisions)}

[活跃文件]
{chr(10).join(f"- {f}" for f in compacted.active_files)}
"""


class AutoCompactManager:
    """自动压缩管理器"""
    
    def __init__(
        self,
        llm_client: Any,
        max_tokens: int = 200000,
        threshold: float = 0.95
    ):
        self.compactor = ContextCompactor(llm_client, threshold)
        self.max_tokens = max_tokens
        self.compaction_history: List[CompactedContext] = []
    
    async def check_and_compact(
        self,
        conversation_history: List[Dict[str, Any]],
        current_tokens: int,
        current_task: Optional[str] = None
    ) -> Optional[CompactedContext]:
        """检查并执行压缩"""
        if not self.compactor.should_compact(current_tokens, self.max_tokens):
            return None
        
        compacted = await self.compactor.compact(conversation_history, current_task)
        self.compaction_history.append(compacted)
        
        return compacted
    
    def get_compaction_stats(self) -> Dict[str, Any]:
        """获取压缩统计"""
        if not self.compaction_history:
            return {'total_compactions': 0}
        
        total_saved = sum(
            self.max_tokens - c.token_count
            for c in self.compaction_history
        )
        
        return {
            'total_compactions': len(self.compaction_history),
            'total_tokens_saved': total_saved,
            'average_compression_ratio': total_saved / (len(self.compaction_history) * self.max_tokens)
        }
```

---

## 四、实施路线图

### 4.1 优先级矩阵

| 阶段 | 功能 | 优先级 | 预计时间 | 依赖 |
|------|------|--------|----------|------|
| **Phase 1** | 通用任务规划 | P0 | 1-2个月 | 无 |
| **Phase 2** | Hooks 生命周期系统 | P1 | 1个月 | Phase 1 |
| **Phase 3** | 项目配置系统 | P1 | 1个月 | 无 |
| **Phase 4** | 专业化 Subagents | P1 | 1个月 | Phase 1 |
| **Phase 5** | 智能上下文压缩 | P2 | 1个月 | Phase 1 |

### 4.2 关键里程碑

```
Month 1: Phase 1 完成
├── Week 1-2: 任务理解模块
├── Week 3-4: 任务分解器
└── Week 5-6: 闭环执行引擎

Month 2: Phase 2 & 3 完成
├── Week 7-8: Hooks 系统
└── Week 9-10: 项目配置系统

Month 3: Phase 4 & 5 完成
├── Week 11-12: 专业化 Subagents
└── Week 13-14: 智能上下文压缩
```

### 4.3 预期效果

| 改进项 | 改进前 | 改进后 |
|--------|--------|--------|
| 任务范围 | 仅 UT 生成 | 任意编程任务 |
| 扩展性 | 固定流程 | Hooks 自定义 |
| 项目理解 | 每次重新分析 | PYUT.md 持久化 |
| 任务分工 | 单一 Agent | 专业化 Subagents |
| 长任务支持 | 上下文丢失 | 自动压缩续传 |

---

## 五、总结

本改进计划参考 Claude Code 的核心设计哲学，提出了五个关键改进方向：

1. **通用任务规划**：从专用工具进化为通用 Coding Agent
2. **Hooks 系统**：提供生命周期扩展点，增强可定制性
3. **项目配置系统**：类似 CLAUDE.md，让 Agent 快速理解项目
4. **专业化 Subagents**：任务分工，提升效率和准确性
5. **智能上下文压缩**：支持长周期复杂任务

这些改进将使 PyUT Agent 在保持 UT 生成领域优势的同时，具备处理更广泛编程任务的能力，向真正的"自主编程 Agent"迈进。

---

**参考资源**：
- [Claude Code 高级功能全解析](https://www.cnblogs.com/dqtx33/p/19488109)
- [Coding Agent 的进化之路](https://juejin.cn/post/7607358297457475584)
- [Claude Skills 与 MCP 的关系](http://m.toutiao.com/group/7596997791998591488/)
- [OpenCode 超级详细入门指南](http://m.toutiao.com/group/7592244035357393446/)

**计划制定日期**：2026-03-05
