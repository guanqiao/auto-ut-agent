"""Universal Task Planner - 通用任务规划器

参考 Claude Code 的任务理解能力，实现：
1. 理解任意编程需求
2. 自动分解任务
3. 动态调整计划
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Set
from enum import Enum, auto
from pathlib import Path
import json
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


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
    EXPLORATION = "exploration"              # 代码库探索
    PLANNING = "planning"                    # 方案设计


@dataclass
class TaskUnderstanding:
    """任务理解结果"""
    task_type: TaskType
    description: str
    target_files: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    estimated_complexity: int = 3  # 1-5
    context_requirements: List[str] = field(default_factory=list)


@dataclass
class Subtask:
    """子任务"""
    id: str
    description: str
    task_type: TaskType
    dependencies: List[str] = field(default_factory=list)
    tools_needed: List[str] = field(default_factory=list)
    estimated_complexity: int = 3
    success_criteria: str = ""
    max_retries: int = 3
    timeout_seconds: int = 300


@dataclass
class ExecutionPlan:
    """执行计划"""
    task_id: str
    original_request: str
    understanding: TaskUnderstanding
    subtasks: List[Subtask]
    execution_order: List[str]
    rollback_strategy: Optional[str] = None
    created_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SubtaskResult:
    """子任务执行结果"""
    subtask_id: str
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    execution_time: float = 0.0
    retry_count: int = 0


@dataclass
class ExecutionResult:
    """执行结果"""
    success: bool
    plan: ExecutionPlan
    subtask_results: List[SubtaskResult]
    completed_subtasks: Set[str] = field(default_factory=set)
    failed_subtasks: Set[str] = field(default_factory=set)
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class TaskHandler(ABC):
    """任务处理器基类"""
    
    @abstractmethod
    async def handle(self, subtask: Subtask, context: Dict[str, Any]) -> SubtaskResult:
        """处理子任务"""
        pass
    
    @abstractmethod
    def can_handle(self, task_type: TaskType) -> bool:
        """是否能处理该任务类型"""
        pass


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
        self._task_handlers: Dict[TaskType, TaskHandler] = {}
        self._execution_history: List[ExecutionResult] = []
    
    def register_task_handler(self, task_type: TaskType, handler: TaskHandler) -> None:
        """注册任务类型处理器"""
        self._task_handlers[task_type] = handler
        logger.info(f"Registered task handler for {task_type.value}")
    
    async def understand_task(
        self,
        user_request: str,
        project_context: Dict[str, Any]
    ) -> TaskUnderstanding:
        """理解用户任务
        
        使用 LLM 分析用户请求，确定任务类型和关键信息。
        """
        prompt = f"""分析以下用户编程请求，提取关键信息：

用户请求：{user_request}

项目上下文：
- 语言：{project_context.get('language', 'Unknown')}
- 构建工具：{project_context.get('build_tool', 'Unknown')}
- 项目结构：{json.dumps(project_context.get('structure', {}), indent=2, ensure_ascii=False)[:500]}

请分析并返回 JSON 格式：
{{
    "task_type": "test_generation|code_refactoring|bug_fix|feature_add|code_review|documentation|dependency_update|query|exploration|planning",
    "description": "任务描述",
    "target_files": ["目标文件列表"],
    "constraints": ["约束条件"],
    "success_criteria": ["成功标准"],
    "estimated_complexity": 1-5,
    "context_requirements": ["需要的上下文信息"]
}}
"""
        
        try:
            response = await self.llm.generate(prompt)
            data = json.loads(response)
            
            return TaskUnderstanding(
                task_type=TaskType(data['task_type']),
                description=data['description'],
                target_files=data.get('target_files', []),
                constraints=data.get('constraints', []),
                success_criteria=data.get('success_criteria', []),
                estimated_complexity=data.get('estimated_complexity', 3),
                context_requirements=data.get('context_requirements', [])
            )
        except Exception as e:
            logger.error(f"Failed to understand task: {e}")
            # 回退到默认理解
            return TaskUnderstanding(
                task_type=TaskType.QUERY,
                description=user_request,
                target_files=[],
                constraints=[],
                success_criteria=["完成任务"],
                estimated_complexity=3
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
        elif understanding.task_type == TaskType.CODE_REVIEW:
            return await self._decompose_code_review(understanding, project_context)
        elif understanding.task_type == TaskType.EXPLORATION:
            return await self._decompose_exploration(understanding, project_context)
        elif understanding.task_type == TaskType.PLANNING:
            return await self._decompose_planning(understanding, project_context)
        else:
            return await self._decompose_generic(understanding, project_context)
    
    async def _decompose_test_generation(
        self,
        understanding: TaskUnderstanding,
        project_context: Dict[str, Any]
    ) -> ExecutionPlan:
        """分解测试生成任务"""
        import time
        
        target = understanding.target_files[0] if understanding.target_files else 'Unknown'
        
        subtasks = [
            Subtask(
                id="analyze_target",
                description=f"分析目标类：{target}",
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
                success_criteria="所有测试通过",
                max_retries=5
            )
        ]
        
        return ExecutionPlan(
            task_id=f"task_{int(time.time() * 1000)}",
            original_request=understanding.description,
            understanding=understanding,
            subtasks=subtasks,
            execution_order=["analyze_target", "analyze_dependencies", "generate_test", 
                           "compile_test", "run_test", "fix_issues"],
            rollback_strategy="git checkout -- tests/",
            created_at=time.time()
        )
    
    async def _decompose_refactoring(
        self,
        understanding: TaskUnderstanding,
        project_context: Dict[str, Any]
    ) -> ExecutionPlan:
        """分解代码重构任务"""
        import time
        
        subtasks = [
            Subtask(
                id="analyze_impact",
                description="分析重构影响范围",
                task_type=TaskType.EXPLORATION,
                dependencies=[],
                tools_needed=["semantic_analyzer", "dependency_analyzer"],
                estimated_complexity=3,
                success_criteria="识别所有受影响的文件和依赖"
            ),
            Subtask(
                id="create_plan",
                description="制定重构计划",
                task_type=TaskType.PLANNING,
                dependencies=["analyze_impact"],
                tools_needed=[],
                estimated_complexity=2,
                success_criteria="获得详细的分步重构计划"
            ),
            Subtask(
                id="backup_code",
                description="备份原始代码",
                task_type=TaskType.QUERY,
                dependencies=[],
                tools_needed=["git", "file_copy"],
                estimated_complexity=1,
                success_criteria="创建可恢复的代码备份"
            ),
            Subtask(
                id="execute_refactoring",
                description="执行重构",
                task_type=TaskType.CODE_REFACTORING,
                dependencies=["create_plan", "backup_code"],
                tools_needed=["refactoring_engine", "file_edit"],
                estimated_complexity=4,
                success_criteria="完成所有计划的重构步骤"
            ),
            Subtask(
                id="verify_refactoring",
                description="验证重构结果",
                task_type=TaskType.QUERY,
                dependencies=["execute_refactoring"],
                tools_needed=["maven", "test_runner"],
                estimated_complexity=2,
                success_criteria="所有测试通过，功能正常"
            )
        ]
        
        return ExecutionPlan(
            task_id=f"task_{int(time.time() * 1000)}",
            original_request=understanding.description,
            understanding=understanding,
            subtasks=subtasks,
            execution_order=["backup_code", "analyze_impact", "create_plan", 
                           "execute_refactoring", "verify_refactoring"],
            rollback_strategy="git reset --hard HEAD",
            created_at=time.time()
        )
    
    async def _decompose_bug_fix(
        self,
        understanding: TaskUnderstanding,
        project_context: Dict[str, Any]
    ) -> ExecutionPlan:
        """分解 Bug 修复任务"""
        import time
        
        subtasks = [
            Subtask(
                id="reproduce_bug",
                description="复现 Bug",
                task_type=TaskType.QUERY,
                dependencies=[],
                tools_needed=["test_runner"],
                estimated_complexity=2,
                success_criteria="成功复现 Bug"
            ),
            Subtask(
                id="analyze_root_cause",
                description="分析根本原因",
                task_type=TaskType.QUERY,
                dependencies=["reproduce_bug"],
                tools_needed=["error_analyzer", "debug_tools"],
                estimated_complexity=3,
                success_criteria="确定 Bug 的根本原因"
            ),
            Subtask(
                id="implement_fix",
                description="实现修复",
                task_type=TaskType.BUG_FIX,
                dependencies=["analyze_root_cause"],
                tools_needed=["file_edit", "code_editor"],
                estimated_complexity=3,
                success_criteria="修复代码实现完成"
            ),
            Subtask(
                id="verify_fix",
                description="验证修复",
                task_type=TaskType.QUERY,
                dependencies=["implement_fix"],
                tools_needed=["test_runner"],
                estimated_complexity=2,
                success_criteria="Bug 不再复现，所有测试通过"
            ),
            Subtask(
                id="regression_test",
                description="回归测试",
                task_type=TaskType.QUERY,
                dependencies=["verify_fix"],
                tools_needed=["test_runner", "coverage_analyzer"],
                estimated_complexity=2,
                success_criteria="没有引入新的问题"
            )
        ]
        
        return ExecutionPlan(
            task_id=f"task_{int(time.time() * 1000)}",
            original_request=understanding.description,
            understanding=understanding,
            subtasks=subtasks,
            execution_order=["reproduce_bug", "analyze_root_cause", "implement_fix",
                           "verify_fix", "regression_test"],
            created_at=time.time()
        )
    
    async def _decompose_feature_add(
        self,
        understanding: TaskUnderstanding,
        project_context: Dict[str, Any]
    ) -> ExecutionPlan:
        """分解功能添加任务"""
        import time
        
        subtasks = [
            Subtask(
                id="analyze_requirements",
                description="分析需求",
                task_type=TaskType.PLANNING,
                dependencies=[],
                tools_needed=[],
                estimated_complexity=2,
                success_criteria="明确功能需求和验收标准"
            ),
            Subtask(
                id="design_solution",
                description="设计方案",
                task_type=TaskType.PLANNING,
                dependencies=["analyze_requirements"],
                tools_needed=[],
                estimated_complexity=3,
                success_criteria="获得技术设计方案"
            ),
            Subtask(
                id="implement_feature",
                description="实现功能",
                task_type=TaskType.FEATURE_ADD,
                dependencies=["design_solution"],
                tools_needed=["code_editor", "file_write"],
                estimated_complexity=4,
                success_criteria="功能代码实现完成"
            ),
            Subtask(
                id="write_tests",
                description="编写测试",
                task_type=TaskType.TEST_GENERATION,
                dependencies=["implement_feature"],
                tools_needed=["test_generator"],
                estimated_complexity=3,
                success_criteria="测试覆盖新功能"
            ),
            Subtask(
                id="verify_feature",
                description="验证功能",
                task_type=TaskType.QUERY,
                dependencies=["write_tests"],
                tools_needed=["test_runner"],
                estimated_complexity=2,
                success_criteria="功能按预期工作"
            )
        ]
        
        return ExecutionPlan(
            task_id=f"task_{int(time.time() * 1000)}",
            original_request=understanding.description,
            understanding=understanding,
            subtasks=subtasks,
            execution_order=["analyze_requirements", "design_solution", "implement_feature",
                           "write_tests", "verify_feature"],
            created_at=time.time()
        )
    
    async def _decompose_code_review(
        self,
        understanding: TaskUnderstanding,
        project_context: Dict[str, Any]
    ) -> ExecutionPlan:
        """分解代码审查任务"""
        import time
        
        subtasks = [
            Subtask(
                id="read_code",
                description="读取代码",
                task_type=TaskType.QUERY,
                dependencies=[],
                tools_needed=["file_read"],
                estimated_complexity=2,
                success_criteria="获取完整代码内容"
            ),
            Subtask(
                id="analyze_quality",
                description="分析代码质量",
                task_type=TaskType.CODE_REVIEW,
                dependencies=["read_code"],
                tools_needed=["quality_analyzer", "static_analysis"],
                estimated_complexity=3,
                success_criteria="识别代码质量问题"
            ),
            Subtask(
                id="generate_report",
                description="生成审查报告",
                task_type=TaskType.DOCUMENTATION,
                dependencies=["analyze_quality"],
                tools_needed=[],
                estimated_complexity=2,
                success_criteria="获得详细的审查报告"
            )
        ]
        
        return ExecutionPlan(
            task_id=f"task_{int(time.time() * 1000)}",
            original_request=understanding.description,
            understanding=understanding,
            subtasks=subtasks,
            execution_order=["read_code", "analyze_quality", "generate_report"],
            created_at=time.time()
        )
    
    async def _decompose_exploration(
        self,
        understanding: TaskUnderstanding,
        project_context: Dict[str, Any]
    ) -> ExecutionPlan:
        """分解代码库探索任务"""
        import time
        
        subtasks = [
            Subtask(
                id="explore_structure",
                description="探索项目结构",
                task_type=TaskType.EXPLORATION,
                dependencies=[],
                tools_needed=["file_list", "glob"],
                estimated_complexity=2,
                success_criteria="获得项目整体结构"
            ),
            Subtask(
                id="analyze_dependencies",
                description="分析依赖关系",
                task_type=TaskType.EXPLORATION,
                dependencies=["explore_structure"],
                tools_needed=["dependency_analyzer"],
                estimated_complexity=3,
                success_criteria="理解模块间依赖关系"
            ),
            Subtask(
                id="summarize_findings",
                description="总结发现",
                task_type=TaskType.DOCUMENTATION,
                dependencies=["analyze_dependencies"],
                tools_needed=[],
                estimated_complexity=2,
                success_criteria="获得清晰的探索报告"
            )
        ]
        
        return ExecutionPlan(
            task_id=f"task_{int(time.time() * 1000)}",
            original_request=understanding.description,
            understanding=understanding,
            subtasks=subtasks,
            execution_order=["explore_structure", "analyze_dependencies", "summarize_findings"],
            created_at=time.time()
        )
    
    async def _decompose_planning(
        self,
        understanding: TaskUnderstanding,
        project_context: Dict[str, Any]
    ) -> ExecutionPlan:
        """分解方案设计任务"""
        import time
        
        subtasks = [
            Subtask(
                id="gather_requirements",
                description="收集需求",
                task_type=TaskType.QUERY,
                dependencies=[],
                tools_needed=[],
                estimated_complexity=2,
                success_criteria="明确所有需求"
            ),
            Subtask(
                id="research_solutions",
                description="研究解决方案",
                task_type=TaskType.EXPLORATION,
                dependencies=["gather_requirements"],
                tools_needed=["web_search", "file_read"],
                estimated_complexity=3,
                success_criteria="了解可行的解决方案"
            ),
            Subtask(
                id="create_design_doc",
                description="创建设计文档",
                task_type=TaskType.PLANNING,
                dependencies=["research_solutions"],
                tools_needed=[],
                estimated_complexity=3,
                success_criteria="获得完整的设计文档"
            )
        ]
        
        return ExecutionPlan(
            task_id=f"task_{int(time.time() * 1000)}",
            original_request=understanding.description,
            understanding=understanding,
            subtasks=subtasks,
            execution_order=["gather_requirements", "research_solutions", "create_design_doc"],
            created_at=time.time()
        )
    
    async def _decompose_generic(
        self,
        understanding: TaskUnderstanding,
        project_context: Dict[str, Any]
    ) -> ExecutionPlan:
        """通用任务分解"""
        import time
        
        # 创建一个简单的单任务计划
        subtask = Subtask(
            id="execute_task",
            description=understanding.description,
            task_type=understanding.task_type,
            dependencies=[],
            tools_needed=[],
            estimated_complexity=understanding.estimated_complexity,
            success_criteria="任务完成"
        )
        
        return ExecutionPlan(
            task_id=f"task_{int(time.time() * 1000)}",
            original_request=understanding.description,
            understanding=understanding,
            subtasks=[subtask],
            execution_order=["execute_task"],
            created_at=time.time()
        )
    
    async def execute_with_feedback(
        self,
        plan: ExecutionPlan,
        context: Dict[str, Any],
        progress_callback: Optional[Callable[[Subtask, SubtaskResult], Any]] = None
    ) -> ExecutionResult:
        """执行计划并动态调整
        
        参考 Claude Code 的闭环执行模式：
        1. 执行子任务
        2. 观察结果
        3. 根据反馈调整计划
        4. 继续执行或终止
        """
        import time
        
        start_time = time.time()
        results: List[SubtaskResult] = []
        completed_subtasks: Set[str] = set()
        failed_subtasks: Set[str] = set()
        
        for subtask_id in plan.execution_order[:]:
            # 检查是否已失败
            if subtask_id in failed_subtasks:
                continue
            
            subtask = self._find_subtask(plan, subtask_id)
            if not subtask:
                logger.warning(f"Subtask {subtask_id} not found in plan")
                continue
            
            # 检查依赖是否满足
            missing_deps = [dep for dep in subtask.dependencies if dep not in completed_subtasks]
            if missing_deps:
                logger.warning(f"Dependencies not met for {subtask_id}: {missing_deps}")
                # 尝试重新排序
                if not await self._reorder_execution(plan, completed_subtasks):
                    failed_subtasks.add(subtask_id)
                    results.append(SubtaskResult(
                        subtask_id=subtask_id,
                        success=False,
                        error=f"Dependencies not met: {missing_deps}"
                    ))
                    continue
            
            # 执行子任务
            result = await self._execute_subtask(subtask, context)
            results.append(result)
            
            # 回调进度
            if progress_callback:
                try:
                    await progress_callback(subtask, result)
                except Exception as e:
                    logger.error(f"Progress callback failed: {e}")
            
            # 处理结果
            if result.success:
                completed_subtasks.add(subtask_id)
            else:
                failed_subtasks.add(subtask_id)
                
                # 尝试调整计划
                if result.retry_count < subtask.max_retries:
                    adjustment = await self._adjust_plan(plan, subtask, result, context)
                    if adjustment:
                        plan = self._apply_adjustment(plan, adjustment)
                        # 重试当前任务
                        result = await self._execute_subtask(subtask, context, retry_count=result.retry_count + 1)
                        results[-1] = result
                        if result.success:
                            completed_subtasks.add(subtask_id)
                            failed_subtasks.discard(subtask_id)
        
        execution_time = time.time() - start_time
        
        execution_result = ExecutionResult(
            success=len(failed_subtasks) == 0,
            plan=plan,
            subtask_results=results,
            completed_subtasks=completed_subtasks,
            failed_subtasks=failed_subtasks,
            execution_time=execution_time
        )
        
        self._execution_history.append(execution_result)
        return execution_result
    
    async def _execute_subtask(
        self,
        subtask: Subtask,
        context: Dict[str, Any],
        retry_count: int = 0
    ) -> SubtaskResult:
        """执行单个子任务"""
        import time
        
        start_time = time.time()
        
        handler = self._task_handlers.get(subtask.task_type)
        if not handler:
            return SubtaskResult(
                subtask_id=subtask.id,
                success=False,
                error=f"No handler for task type: {subtask.task_type}",
                execution_time=time.time() - start_time,
                retry_count=retry_count
            )
        
        try:
            result = await handler.handle(subtask, context)
            result.execution_time = time.time() - start_time
            result.retry_count = retry_count
            return result
        except Exception as e:
            logger.error(f"Failed to execute subtask {subtask.id}: {e}")
            return SubtaskResult(
                subtask_id=subtask.id,
                success=False,
                error=str(e),
                execution_time=time.time() - start_time,
                retry_count=retry_count
            )
    
    def _find_subtask(self, plan: ExecutionPlan, subtask_id: str) -> Optional[Subtask]:
        """查找子任务"""
        for subtask in plan.subtasks:
            if subtask.id == subtask_id:
                return subtask
        return None
    
    async def _adjust_plan(
        self,
        plan: ExecutionPlan,
        failed_subtask: Subtask,
        result: SubtaskResult,
        context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """根据失败结果调整计划"""
        # 使用 LLM 分析失败原因并建议调整
        prompt = f"""子任务执行失败，请分析原因并建议调整：

失败子任务：{failed_subtask.description}
错误信息：{result.error}

当前计划：
{json.dumps([{"id": s.id, "desc": s.description} for s in plan.subtasks], indent=2, ensure_ascii=False)}

建议如何调整计划以解决问题？返回 JSON 格式：
{{
    "action": "add_subtask|remove_subtask|modify_subtask|reorder",
    "details": {{}}
}}
"""
        
        try:
            response = await self.llm.generate(prompt)
            return json.loads(response)
        except Exception as e:
            logger.error(f"Failed to adjust plan: {e}")
            return None
    
    def _apply_adjustment(
        self,
        plan: ExecutionPlan,
        adjustment: Dict[str, Any]
    ) -> ExecutionPlan:
        """应用计划调整"""
        action = adjustment.get('action')
        
        if action == 'add_subtask':
            # 添加新子任务
            details = adjustment.get('details', {})
            new_subtask = Subtask(
                id=details.get('id', f"added_{len(plan.subtasks)}"),
                description=details.get('description', ''),
                task_type=TaskType(details.get('task_type', 'query')),
                dependencies=details.get('dependencies', []),
                tools_needed=details.get('tools_needed', []),
                estimated_complexity=details.get('estimated_complexity', 2),
                success_criteria=details.get('success_criteria', '')
            )
            plan.subtasks.append(new_subtask)
            # 更新执行顺序
            insert_after = details.get('insert_after')
            if insert_after and insert_after in plan.execution_order:
                idx = plan.execution_order.index(insert_after)
                plan.execution_order.insert(idx + 1, new_subtask.id)
            else:
                plan.execution_order.append(new_subtask.id)
        
        elif action == 'reorder':
            # 重新排序
            new_order = adjustment.get('details', {}).get('new_order', [])
            if new_order:
                plan.execution_order = new_order
        
        return plan
    
    async def _reorder_execution(
        self,
        plan: ExecutionPlan,
        completed: Set[str]
    ) -> bool:
        """重新排序执行顺序，确保依赖满足"""
        # 简单的拓扑排序
        pending = [s for s in plan.subtasks if s.id not in completed]
        new_order = list(completed)
        
        while pending:
            progress = False
            for subtask in pending[:]:
                if all(dep in new_order for dep in subtask.dependencies):
                    new_order.append(subtask.id)
                    pending.remove(subtask)
                    progress = True
            
            if not progress and pending:
                # 存在循环依赖，无法重排
                return False
        
        plan.execution_order = new_order
        return True
    
    def get_execution_history(self) -> List[ExecutionResult]:
        """获取执行历史"""
        return self._execution_history.copy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        if not self._execution_history:
            return {
                'total_executions': 0,
                'success_rate': 0.0,
                'average_execution_time': 0.0
            }
        
        total = len(self._execution_history)
        successful = sum(1 for r in self._execution_history if r.success)
        avg_time = sum(r.execution_time for r in self._execution_history) / total
        
        return {
            'total_executions': total,
            'successful_executions': successful,
            'failed_executions': total - successful,
            'success_rate': successful / total,
            'average_execution_time': avg_time
        }
