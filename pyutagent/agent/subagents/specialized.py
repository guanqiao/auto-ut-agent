"""Specialized Subagents - 专业化子代理

参考 Claude Code 的 Subagents 设计：
- BashAgent: 专注命令行任务
- PlanAgent: 专注方案设计
- ExploreAgent: 专注代码库探索
- TestGenAgent: 专注测试生成 (PyUT Agent 的核心能力)
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set
from abc import ABC, abstractmethod
from pathlib import Path
import logging
import json
import time

logger = logging.getLogger(__name__)


@dataclass
class SubagentResult:
    """子代理执行结果"""
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    summary: str = ""
    artifacts: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    token_usage: int = 0


class SpecializedSubagent(ABC):
    """专业化子代理基类
    
    参考 Claude Code 的 Subagents 设计：
    - BashAgent: 专注执行命令行相关任务
    - PlanAgent: 负责设计清晰的项目实现方案
    - ExploreAgent: 快速遍历和分析代码库结构
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        llm_client: Any,
        tool_registry: Any
    ):
        self.name = name
        self.description = description
        self.llm = llm_client
        self.tools = tool_registry
        self.execution_count = 0
        self.total_execution_time = 0.0
        self.success_count = 0
    
    @abstractmethod
    async def execute(self, task: str, context: Dict[str, Any]) -> SubagentResult:
        """执行任务"""
        pass
    
    @abstractmethod
    def can_handle(self, task: str) -> float:
        """判断是否能处理该任务（返回置信度 0-1）"""
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'name': self.name,
            'description': self.description,
            'execution_count': self.execution_count,
            'success_count': self.success_count,
            'success_rate': self.success_count / max(self.execution_count, 1),
            'average_execution_time': self.total_execution_time / max(self.execution_count, 1)
        }


class BashSubagent(SpecializedSubagent):
    """Bash 子代理
    
    专注执行命令行相关任务
    """
    
    BASH_KEYWORDS = [
        'run', 'execute', 'command', 'shell', 'bash', 'cmd', 'terminal',
        'mvn', 'maven', 'gradle', 'gradlew', 'git', 'npm', 'yarn', 'pnpm',
        'docker', 'kubectl', 'helm', 'aws', 'gcloud', 'azure',
        'build', 'test', 'compile', 'package', 'install', 'deploy',
        'clean', 'verify', 'validate', 'assemble', 'check'
    ]
    
    def __init__(self, llm_client: Any, tool_registry: Any):
        super().__init__(
            name="BashAgent",
            description="专注执行命令行相关任务",
            llm_client=llm_client,
            tool_registry=tool_registry
        )
    
    def can_handle(self, task: str) -> float:
        """判断是否为命令行任务"""
        task_lower = task.lower()
        matches = sum(1 for kw in self.BASH_KEYWORDS if kw in task_lower)
        score = matches / len(self.BASH_KEYWORDS)
        # 放大匹配度，但不超过 1.0
        return min(score * 4, 1.0)
    
    async def execute(self, task: str, context: Dict[str, Any]) -> SubagentResult:
        """执行命令行任务"""
        start_time = time.time()
        self.execution_count += 1
        
        try:
            # 解析任务，提取命令
            prompt = f"""将以下任务转换为可执行的 shell 命令：

任务：{task}
项目目录：{context.get('project_root', '.')}
构建工具：{context.get('build_tool', 'maven')}

请分析任务并返回 JSON 格式：
{{
    "commands": ["命令1", "命令2"],
    "description": "命令描述",
    "expected_output": "预期输出",
    "working_directory": "工作目录（可选）",
    "timeout_seconds": 300,
    "risk_level": "low|medium|high"
}}

注意：
1. 命令应该是安全且可执行的
2. 考虑项目上下文和构建工具
3. 高风险操作（如删除、部署）需要明确标注"""
            
            response = await self.llm.generate(prompt)
            data = json.loads(response)
            
            commands = data.get('commands', [])
            description = data.get('description', '')
            risk_level = data.get('risk_level', 'low')
            
            # 执行命令（通过工具注册表）
            execution_results = []
            for cmd in commands:
                # 这里调用实际的 bash 工具
                result = await self._execute_command(cmd, context)
                execution_results.append(result)
            
            success = all(r.get('success', False) for r in execution_results)
            
            if success:
                self.success_count += 1
            
            execution_time = time.time() - start_time
            self.total_execution_time += execution_time
            
            return SubagentResult(
                success=success,
                data={
                    'commands': commands,
                    'execution_results': execution_results,
                    'risk_level': risk_level,
                    'description': description
                },
                summary=f"Executed {len(commands)} bash commands for: {task}",
                execution_time=execution_time
            )
            
        except Exception as e:
            logger.error(f"BashSubagent failed: {e}")
            return SubagentResult(
                success=False,
                error=str(e),
                summary=f"Failed to execute bash commands: {e}",
                execution_time=time.time() - start_time
            )
    
    async def _execute_command(self, command: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """执行单个命令"""
        # 通过工具注册表执行
        bash_tool = self.tools.get('bash')
        if bash_tool:
            return await bash_tool.execute(command, context)
        
        # 模拟执行
        return {
            'success': True,
            'command': command,
            'stdout': '',
            'stderr': '',
            'return_code': 0
        }


class PlanSubagent(SpecializedSubagent):
    """Plan 子代理
    
    专注设计清晰的项目实现方案
    """
    
    PLAN_KEYWORDS = [
        'plan', 'design', 'architecture', 'strategy', 'blueprint',
        'how to', 'approach', 'solution', 'implement', 'implementation',
        'refactor', 'restructure', 'organize', 'structure',
        'best practice', 'pattern', 'methodology'
    ]
    
    def __init__(self, llm_client: Any, tool_registry: Any):
        super().__init__(
            name="PlanAgent",
            description="专注设计清晰的项目实现方案",
            llm_client=llm_client,
            tool_registry=tool_registry
        )
    
    def can_handle(self, task: str) -> float:
        """判断是否为规划任务"""
        task_lower = task.lower()
        matches = sum(1 for kw in self.PLAN_KEYWORDS if kw in task_lower)
        score = matches / len(self.PLAN_KEYWORDS)
        return min(score * 4, 1.0)
    
    async def execute(self, task: str, context: Dict[str, Any]) -> SubagentResult:
        """设计方案"""
        start_time = time.time()
        self.execution_count += 1
        
        try:
            # 获取项目上下文
            project_info = context.get('project_info', {})
            
            prompt = f"""为以下任务设计详细的实现方案：

任务：{task}

项目信息：
- 名称：{project_info.get('name', 'Unknown')}
- 语言：{project_info.get('language', 'java')}
- 构建工具：{project_info.get('build_tool', 'maven')}
- 架构：{project_info.get('architecture', 'Standard')}

请提供详细的实现方案，返回 JSON 格式：
{{
    "overview": "方案概述",
    "steps": [
        {{
            "order": 1,
            "title": "步骤标题",
            "description": "详细描述",
            "estimated_hours": 2,
            "dependencies": []
        }}
    ],
    "files_involved": ["文件路径1", "文件路径2"],
    "modules_involved": ["模块1", "模块2"],
    "risks": [
        {{
            "description": "风险描述",
            "mitigation": "缓解措施",
            "severity": "low|medium|high"
        }}
    ],
    "validation_methods": ["验证方法1", "验证方法2"],
    "alternatives": [
        {{
            "approach": "替代方案",
            "pros": ["优点1"],
            "cons": ["缺点1"],
            "recommendation": "推荐程度 (1-10)"
        }}
    ]
}}

要求：
1. 步骤要具体且可执行
2. 考虑项目的现有架构和约束
3. 评估潜在风险并提供缓解措施
4. 提供验证方法确保实施成功"""
            
            response = await self.llm.generate(prompt)
            plan_data = json.loads(response)
            
            self.success_count += 1
            execution_time = time.time() - start_time
            self.total_execution_time += execution_time
            
            # 生成方案文档
            plan_doc = self._format_plan_document(plan_data)
            
            return SubagentResult(
                success=True,
                data=plan_data,
                summary=f"Created implementation plan for: {task}",
                artifacts=['plan.md'],
                execution_time=execution_time
            )
            
        except Exception as e:
            logger.error(f"PlanSubagent failed: {e}")
            return SubagentResult(
                success=False,
                error=str(e),
                summary=f"Failed to create plan: {e}",
                execution_time=time.time() - start_time
            )
    
    def _format_plan_document(self, plan_data: Dict[str, Any]) -> str:
        """格式化方案文档"""
        lines = [
            "# Implementation Plan",
            "",
            f"## Overview\n{plan_data.get('overview', '')}",
            "",
            "## Steps",
        ]
        
        for step in plan_data.get('steps', []):
            lines.extend([
                f"\n### {step.get('order', 0)}. {step.get('title', '')}",
                f"{step.get('description', '')}",
                f"- Estimated: {step.get('estimated_hours', 0)} hours",
            ])
        
        lines.extend([
            "",
            "## Files Involved",
        ])
        for f in plan_data.get('files_involved', []):
            lines.append(f"- {f}")
        
        return '\n'.join(lines)


class ExploreSubagent(SpecializedSubagent):
    """Explore 子代理
    
    专注快速遍历和分析代码库结构
    """
    
    EXPLORE_KEYWORDS = [
        'find', 'search', 'locate', 'explore', 'discover', 'lookup',
        'where is', 'how is', 'what is', 'understand', 'analyze',
        'structure', 'organization', 'dependencies', 'relationship',
        'usage', 'reference', 'call', 'invoke'
    ]
    
    def __init__(self, llm_client: Any, tool_registry: Any):
        super().__init__(
            name="ExploreAgent",
            description="专注快速遍历和分析代码库结构",
            llm_client=llm_client,
            tool_registry=tool_registry
        )
    
    def can_handle(self, task: str) -> float:
        """判断是否为探索任务"""
        task_lower = task.lower()
        matches = sum(1 for kw in self.EXPLORE_KEYWORDS if kw in task_lower)
        score = matches / len(self.EXPLORE_KEYWORDS)
        return min(score * 4, 1.0)
    
    async def execute(self, task: str, context: Dict[str, Any]) -> SubagentResult:
        """探索代码库"""
        start_time = time.time()
        self.execution_count += 1
        
        try:
            project_root = context.get('project_root', '.')
            
            # 执行探索
            findings = await self._explore_project(project_root, task)
            
            # 使用 LLM 分析发现
            prompt = f"""分析以下代码库探索结果：

探索任务：{task}

发现结果：
{json.dumps(findings, indent=2, ensure_ascii=False)[:4000]}

请提供分析，返回 JSON 格式：
{{
    "summary": "探索摘要",
    "key_findings": ["关键发现1", "关键发现2"],
    "file_structure": {{
        "total_files": 100,
        "main_directories": ["dir1", "dir2"]
    }},
    "important_files": [
        {{
            "path": "文件路径",
            "relevance": "相关程度 (high/medium/low)",
            "description": "文件描述"
        }}
    ],
    "dependencies": ["依赖1", "依赖2"],
    "patterns_identified": ["模式1", "模式2"],
    "recommendations": ["建议1", "建议2"]
}}
"""
            
            response = await self.llm.generate(prompt)
            analysis = json.loads(response)
            
            self.success_count += 1
            execution_time = time.time() - start_time
            self.total_execution_time += execution_time
            
            return SubagentResult(
                success=True,
                data={
                    'findings': findings,
                    'analysis': analysis
                },
                summary=f"Explored codebase for: {task}",
                artifacts=['exploration_report.md'],
                execution_time=execution_time
            )
            
        except Exception as e:
            logger.error(f"ExploreSubagent failed: {e}")
            return SubagentResult(
                success=False,
                error=str(e),
                summary=f"Failed to explore codebase: {e}",
                execution_time=time.time() - start_time
            )
    
    async def _explore_project(self, root: str, query: str) -> Dict[str, Any]:
        """探索项目结构"""
        findings = {
            'project_root': root,
            'directories': [],
            'files': [],
            'patterns': []
        }
        
        root_path = Path(root)
        if not root_path.exists():
            return findings
        
        # 收集目录结构
        for item in root_path.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                findings['directories'].append({
                    'name': item.name,
                    'path': str(item.relative_to(root_path))
                })
            elif item.is_file():
                findings['files'].append({
                    'name': item.name,
                    'path': str(item.relative_to(root_path)),
                    'size': item.stat().st_size
                })
        
        # 识别项目模式
        if (root_path / 'pom.xml').exists():
            findings['patterns'].append('maven_project')
        if (root_path / 'build.gradle').exists():
            findings['patterns'].append('gradle_project')
        if (root_path / '.git').exists():
            findings['patterns'].append('git_repository')
        if (root_path / 'src').exists():
            findings['patterns'].append('standard_source_structure')
        
        return findings


class TestGenSubagent(SpecializedSubagent):
    """测试生成子代理
    
    PyUT Agent 的核心专业能力
    """
    
    TEST_KEYWORDS = [
        'test', 'unit test', 'generate test', 'create test', 'write test',
        'test coverage', 'coverage', 'junit', 'mock', 'assert', 'verify',
        'test case', 'test scenario', 'test suite', 'integration test'
    ]
    
    def __init__(self, llm_client: Any, tool_registry: Any):
        super().__init__(
            name="TestGenAgent",
            description="专注生成高质量的单元测试 (PyUT Agent 核心能力)",
            llm_client=llm_client,
            tool_registry=tool_registry
        )
    
    def can_handle(self, task: str) -> float:
        """判断是否为测试生成任务"""
        task_lower = task.lower()
        matches = sum(1 for kw in self.TEST_KEYWORDS if kw in task_lower)
        score = matches / len(self.TEST_KEYWORDS)
        return min(score * 4, 1.0)
    
    async def execute(self, task: str, context: Dict[str, Any]) -> SubagentResult:
        """生成测试"""
        start_time = time.time()
        self.execution_count += 1
        
        try:
            # 提取目标类信息
            target_class = context.get('target_class', '')
            target_file = context.get('target_file', '')
            
            prompt = f"""为以下目标生成单元测试：

任务：{task}
目标类：{target_class}
目标文件：{target_file}

项目配置：
- 测试框架：{context.get('test_framework', 'junit5')}
- Mock 框架：{context.get('mock_framework', 'mockito')}
- 覆盖率目标：{context.get('coverage_threshold', 0.8) * 100}%

请生成完整的单元测试代码，返回 JSON 格式：
{{
    "test_class_name": "测试类名",
    "package": "包名",
    "imports": ["import1", "import2"],
    "test_methods": [
        {{
            "name": "testMethodName",
            "description": "测试描述",
            "body": "测试方法代码",
            "test_type": "positive|negative|edge|boundary"
        }}
    ],
    "mocks_needed": ["需要 mock 的类"],
    "test_data": ["测试数据说明"],
    "coverage_analysis": {{
        "expected_line_coverage": 0.85,
        "expected_branch_coverage": 0.75,
        "scenarios_covered": ["场景1", "场景2"]
    }}
}}

要求：
1. 使用指定的测试框架和 Mock 框架
2. 覆盖正常、异常、边界情况
3. 测试方法命名清晰，符合规范
4. 包含必要的注释说明"""
            
            response = await self.llm.generate(prompt)
            test_data = json.loads(response)
            
            # 生成测试代码文件
            test_code = self._generate_test_code(test_data)
            
            # 保存测试文件
            test_file_path = await self._save_test_file(test_data, test_code, context)
            
            self.success_count += 1
            execution_time = time.time() - start_time
            self.total_execution_time += execution_time
            
            return SubagentResult(
                success=True,
                data={
                    'test_class': test_data.get('test_class_name'),
                    'test_methods': len(test_data.get('test_methods', [])),
                    'coverage_analysis': test_data.get('coverage_analysis', {}),
                    'test_code': test_code
                },
                summary=f"Generated tests for: {target_class or task}",
                artifacts=[test_file_path] if test_file_path else [],
                execution_time=execution_time
            )
            
        except Exception as e:
            logger.error(f"TestGenSubagent failed: {e}")
            return SubagentResult(
                success=False,
                error=str(e),
                summary=f"Failed to generate tests: {e}",
                execution_time=time.time() - start_time
            )
    
    def _generate_test_code(self, test_data: Dict[str, Any]) -> str:
        """生成测试代码"""
        lines = []
        
        # Package
        if test_data.get('package'):
            lines.append(f"package {test_data['package']};")
            lines.append("")
        
        # Imports
        for imp in test_data.get('imports', []):
            lines.append(f"import {imp};")
        lines.append("")
        
        # Class
        lines.append(f"public class {test_data.get('test_class_name', 'Test')} {{")
        lines.append("")
        
        # Test methods
        for method in test_data.get('test_methods', []):
            lines.append(f"    // {method.get('description', '')}")
            lines.append(f"    @Test")
            lines.append(f"    public {method.get('body', '')}")
            lines.append("")
        
        lines.append("}")
        
        return '\n'.join(lines)
    
    async def _save_test_file(self, test_data: Dict[str, Any], test_code: str, context: Dict[str, Any]) -> Optional[str]:
        """保存测试文件"""
        # 这里可以调用文件写入工具
        return None


class SubagentRouter:
    """子代理路由器
    
    根据任务类型自动选择最合适的子代理
    """
    
    def __init__(self):
        self.subagents: List[SpecializedSubagent] = []
        self._routing_history: List[Dict[str, Any]] = []
    
    def register(self, subagent: SpecializedSubagent) -> None:
        """注册子代理"""
        self.subagents.append(subagent)
        logger.info(f"Registered subagent: {subagent.name}")
    
    def unregister(self, subagent_name: str) -> bool:
        """注销子代理"""
        for i, agent in enumerate(self.subagents):
            if agent.name == subagent_name:
                self.subagents.pop(i)
                logger.info(f"Unregistered subagent: {subagent_name}")
                return True
        return False
    
    async def route(self, task: str, context: Dict[str, Any]) -> SubagentResult:
        """路由任务到合适的子代理"""
        # 评估每个子代理的匹配度
        scores = [
            (subagent, subagent.can_handle(task))
            for subagent in self.subagents
        ]
        
        # 按匹配度排序
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # 记录路由决策
        routing_decision = {
            'task': task,
            'timestamp': time.time(),
            'scores': [(agent.name, score) for agent, score in scores],
            'selected': None
        }
        
        # 选择最佳匹配
        if scores and scores[0][1] > 0.3:
            best_agent = scores[0][0]
            routing_decision['selected'] = best_agent.name
            self._routing_history.append(routing_decision)
            
            logger.info(f"Routing task to {best_agent.name} (confidence: {scores[0][1]:.2f})")
            return await best_agent.execute(task, context)
        
        # 没有合适的子代理
        routing_decision['selected'] = None
        self._routing_history.append(routing_decision)
        
        logger.warning(f"No suitable subagent found for task: {task}")
        return SubagentResult(
            success=False,
            error="No suitable subagent found",
            summary="No suitable subagent found for the task"
        )
    
    def get_capabilities(self) -> Dict[str, str]:
        """获取所有子代理的能力描述"""
        return {
            agent.name: agent.description
            for agent in self.subagents
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """获取路由统计"""
        agent_stats = [agent.get_stats() for agent in self.subagents]
        
        return {
            'registered_agents': len(self.subagents),
            'agent_stats': agent_stats,
            'routing_history_count': len(self._routing_history)
        }
    
    def get_routing_history(self) -> List[Dict[str, Any]]:
        """获取路由历史"""
        return self._routing_history.copy()


# 便捷函数：创建默认的子代理路由器
def create_default_router(llm_client: Any, tool_registry: Any) -> SubagentRouter:
    """创建默认的子代理路由器"""
    router = SubagentRouter()
    
    # 注册所有子代理
    router.register(BashSubagent(llm_client, tool_registry))
    router.register(PlanSubagent(llm_client, tool_registry))
    router.register(ExploreSubagent(llm_client, tool_registry))
    router.register(TestGenSubagent(llm_client, tool_registry))
    
    return router
