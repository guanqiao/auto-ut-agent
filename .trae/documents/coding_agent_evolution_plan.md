# PyUT Agent 进化计划：从 UT 生成工具到通用 Coding Agent

## 一、现状分析

### 1.1 PyUT Agent 当前能力

PyUT Agent 已经建立了相对完整的架构体系：

| 能力领域 | 当前状态 | 成熟度 |
|----------|----------|--------|
| 测试生成闭环 | Parse → Generate → Compile → Test → Fix | ⭐⭐⭐⭐⭐ |
| 子代理系统 | SubAgent + Orchestrator + 任务路由 | ⭐⭐⭐⭐ |
| 任务规划 | HierarchicalTaskPlanner + TaskPlanner | ⭐⭐⭐⭐ |
| 工具系统 | ToolRegistry + StandardTools + SearchTools | ⭐⭐⭐⭐ |
| Skills 系统 | SkillRegistry + Built-in Skills | ⭐⭐⭐⭐ |
| 上下文管理 | ContextManager + 压缩策略 | ⭐⭐⭐⭐ |
| 错误恢复 | AI 驱动错误分析 + 自适应修复 | ⭐⭐⭐⭐ |
| 覆盖率分析 | JaCoCo 集成 | ⭐⭐⭐⭐⭐ |
| 多语言支持 | Java (Maven/Gradle), Python | ⭐⭐⭐ |

### 1.2 与顶级 Coding Agent 的核心差距

| 能力维度 | PyUT Agent | Claude Code | Cursor | Devin | 差距等级 |
|----------|------------|-------------|--------|-------|----------|
| **通用任务处理** | 仅 UT 生成 | 任意编程任务 | 任意编程任务 | 完整项目 | 🔴 重大 |
| **工具使用智能度** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 🔴 重大 |
| **代码编辑精度** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 🟡 中等 |
| **上下文理解** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 🔴 重大 |
| **自然语言交互** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 🟡 中等 |
| **多文件编辑** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 🔴 重大 |
| **MCP 生态** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 🔴 重大 |
| **项目配置** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 🟡 中等 |
| **子代理协作** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 🟡 中等 |
| **长期规划** | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 🔴 重大 |

---

## 二、核心差距详细分析

### 2.1 Gap 1: 通用任务处理能力（P0 - 最高优先级）

**现状问题：**
- PyUT Agent 专注于单元测试生成，无法处理其他编程任务
- 缺乏任务类型识别和动态分解能力
- 用户无法通过自然语言描述任意编程需求

**Claude Code 的优势：**
```
用户："帮我重构这个类，提取接口并更新所有依赖"
Claude Code：
1. 识别任务类型：CODE_REFACTORING
2. 分析影响范围：找出所有依赖文件
3. 制定计划：提取接口 → 更新实现 → 更新依赖 → 验证
4. 执行并验证
```

**改进目标：**
实现 UniversalTaskPlanner，支持 8+ 种任务类型的自动识别和处理。

### 2.2 Gap 2: MCP 生态集成（P0 - 最高优先级）

**现状问题：**
- 工具系统封闭，无法接入外部工具生态
- 缺乏标准化的工具调用协议
- 无法利用社区开发的工具

**Claude Code 的优势：**
- 完整的 MCP 客户端实现
- 支持本地和远程 MCP 服务器
- 丰富的 Skills 市场
- 工具使用"说明书"机制

**改进目标：**
实现完整的 MCP 客户端，支持动态工具发现和调用。

### 2.3 Gap 3: 智能上下文管理（P1 - 高优先级）

**现状问题：**
- 上下文窗口需要手动配置
- 缺乏智能代码选择能力
- 大型代码库理解能力有限

**Claude Code 的优势：**
- 三层上下文管理（LSP + SQLite + 结构化）
- 自动代码索引和语义搜索
- 智能上下文压缩

**改进目标：**
实现自动上下文选择 + 代码库索引 + 语义搜索。

### 2.4 Gap 4: 多文件编辑能力（P1 - 高优先级）

**现状问题：**
- 主要支持单文件编辑
- 缺乏复杂重构能力
- 多文件修改需要多次交互

**Cursor/Claude Code 的优势：**
- 自动识别相关文件
- 批量修改能力
- Diff 预览和确认机制

**改进目标：**
实现多文件编辑支持 + 智能影响分析 + Diff 预览。

### 2.5 Gap 5: 项目配置系统（P1 - 高优先级）

**现状问题：**
- 每次都需要重新分析项目
- 缺乏项目特定的配置持久化
- 编码规范无法定制

**Claude Code 的优势：**
- CLAUDE.md 自动初始化
- 项目上下文持久化
- 编码规范定制

**改进目标：**
实现 PYUT.md 项目配置系统。

### 2.6 Gap 6: 专业化 Subagents（P1 - 高优先级）

**现状问题：**
- 子代理分工不够明确
- 缺乏专门的 Bash/Plan/Explore 代理
- 任务路由不够智能

**Claude Code 的优势：**
- BashAgent：专注命令行
- PlanAgent：专注方案设计
- ExploreAgent：专注代码探索

**改进目标：**
实现三大专业化子代理。

### 2.7 Gap 7: 长期规划能力（P2 - 中优先级）

**现状问题：**
- 缺乏长期任务规划能力
- 上下文压缩机制不够完善
- 无法处理数小时的复杂任务

**Devin 的优势：**
- 独立工作数小时
- 完整的项目开发流程
- 自主学习和迭代

**改进目标：**
实现上下文压缩 + 检查点机制 + 长期任务支持。

---

## 三、改进路线图

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

### Phase 2: MCP 生态集成（P0 - 1-2个月）

#### 3.2.1 目标
实现完整的 MCP 客户端，支持动态工具发现和调用，接入丰富的工具生态。

#### 3.2.2 核心实现

```python
# pyutagent/mcp/client.py

from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Callable
import json
import asyncio
from enum import Enum

class MCPTransport(Enum):
    """MCP 传输协议"""
    STDIO = "stdio"
    SSE = "sse"
    WEBSOCKET = "websocket"

@dataclass
class MCPTool:
    """MCP 工具定义"""
    name: str
    description: str
    parameters: Dict[str, Any]
    server_name: str
    
@dataclass
class MCPServer:
    """MCP 服务器配置"""
    name: str
    command: str
    args: List[str]
    env: Dict[str, str]
    transport: MCPTransport
    tools: List[MCPTool] = None


class MCPClient:
    """MCP 客户端
    
    实现 Model Context Protocol 客户端，支持：
    1. 连接 MCP 服务器
    2. 发现可用工具
    3. 调用远程工具
    4. 管理工具生命周期
    """
    
    def __init__(self):
        self.servers: Dict[str, MCPServer] = {}
        self.tools: Dict[str, MCPTool] = {}
        self._connections: Dict[str, Any] = {}
    
    async def connect_server(self, server: MCPServer) -> bool:
        """连接 MCP 服务器"""
        try:
            if server.transport == MCPTransport.STDIO:
                connection = await self._connect_stdio(server)
            elif server.transport == MCPTransport.SSE:
                connection = await self._connect_sse(server)
            else:
                raise ValueError(f"Unsupported transport: {server.transport}")
            
            self._connections[server.name] = connection
            self.servers[server.name] = server
            
            # 发现工具
            tools = await self._discover_tools(server.name)
            server.tools = tools
            for tool in tools:
                self.tools[f"{server.name}:{tool.name}"] = tool
            
            return True
        except Exception as e:
            print(f"Failed to connect MCP server {server.name}: {e}")
            return False
    
    async def _connect_stdio(self, server: MCPServer) -> Any:
        """通过 STDIO 连接"""
        import subprocess
        
        process = subprocess.Popen(
            [server.command] + server.args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={**server.env}
        )
        
        # 发送初始化请求
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "pyutagent", "version": "1.0.0"}
            }
        }
        
        process.stdin.write(json.dumps(init_request).encode() + b'\n')
        process.stdin.flush()
        
        # 读取响应
        response = process.stdout.readline()
        return process
    
    async def _discover_tools(self, server_name: str) -> List[MCPTool]:
        """发现服务器上的工具"""
        request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list"
        }
        
        response = await self._send_request(server_name, request)
        
        tools = []
        for tool_data in response.get("tools", []):
            tools.append(MCPTool(
                name=tool_data["name"],
                description=tool_data["description"],
                parameters=tool_data.get("parameters", {}),
                server_name=server_name
            ))
        
        return tools
    
    async def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """调用 MCP 工具"""
        tool = self.tools.get(tool_name)
        if not tool:
            raise ValueError(f"Tool not found: {tool_name}")
        
        request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": tool.name,
                "arguments": arguments
            }
        }
        
        return await self._send_request(tool.server_name, request)
    
    async def _send_request(
        self,
        server_name: str,
        request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """发送请求到 MCP 服务器"""
        connection = self._connections.get(server_name)
        if not connection:
            raise ValueError(f"Server not connected: {server_name}")
        
        # 发送请求
        connection.stdin.write(json.dumps(request).encode() + b'\n')
        connection.stdin.flush()
        
        # 读取响应
        response = connection.stdout.readline()
        return json.loads(response.decode())
    
    def list_tools(self) -> List[MCPTool]:
        """列出所有可用工具"""
        return list(self.tools.values())
    
    async def disconnect_all(self):
        """断开所有连接"""
        for connection in self._connections.values():
            connection.terminate()
        self._connections.clear()


# MCP 工具适配器
class MCPToolAdapter:
    """将 MCP 工具适配为 PyUT Agent 工具"""
    
    def __init__(self, mcp_client: MCPClient):
        self.mcp = mcp_client
    
    def adapt_tool(self, mcp_tool: MCPTool) -> Dict[str, Any]:
        """适配 MCP 工具为内部工具格式"""
        return {
            "name": f"mcp_{mcp_tool.server_name}_{mcp_tool.name}",
            "description": mcp_tool.description,
            "parameters": mcp_tool.parameters,
            "handler": self._create_handler(mcp_tool)
        }
    
    def _create_handler(self, mcp_tool: MCPTool) -> Callable:
        """创建工具处理函数"""
        async def handler(**kwargs) -> Dict[str, Any]:
            tool_name = f"{mcp_tool.server_name}:{mcp_tool.name}"
            return await self.mcp.call_tool(tool_name, kwargs)
        
        return handler
```

#### 3.2.3 MCP Skills 集成

```python
# pyutagent/mcp/skills.py

from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class MCPSkill:
    """MCP Skill 定义
    
    Skill = 工具使用说明书 + 最佳实践 + 错误处理
    """
    name: str
    description: str
    triggers: List[str]              # 触发关键词
    tool_usage_guide: str            # 工具使用指南
    best_practices: List[str]        # 最佳实践
    error_handling: List[str]        # 错误处理建议
    required_tools: List[str]        # 所需 MCP 工具
    examples: List[Dict[str, Any]]   # 使用示例


class MCPSkillRegistry:
    """MCP Skill 注册表"""
    
    def __init__(self):
        self._skills: Dict[str, MCPSkill] = {}
        self._register_builtin_skills()
    
    def _register_builtin_skills(self):
        """注册内置 Skills"""
        
        # 1. 文件搜索 Skill
        self.register(MCPSkill(
            name="file_search",
            description="在代码库中搜索文件",
            triggers=["find", "search", "locate", "where is"],
            tool_usage_guide="""
            使用文件搜索工具时：
            1. 优先使用 glob 模式匹配文件名
            2. 使用 grep 搜索文件内容
            3. 结合两种方法提高准确性
            """,
            best_practices=[
                "先使用宽泛的搜索，再逐步精确",
                "记录搜索结果供后续使用",
                "验证找到的文件是否符合预期"
            ],
            error_handling=[
                "如果搜索无结果，尝试不同的关键词",
                "检查搜索路径是否正确",
                "考虑文件可能被忽略或排除"
            ],
            required_tools=["mcp_filesystem_glob", "mcp_filesystem_grep"],
            examples=[
                {
                    "request": "找到所有 UserService 相关的文件",
                    "steps": [
                        "使用 glob 搜索 *UserService*",
                        "使用 grep 搜索 class UserService",
                        "验证结果"
                    ]
                }
            ]
        ))
        
        # 2. 代码编辑 Skill
        self.register(MCPSkill(
            name="code_edit",
            description="编辑代码文件",
            triggers=["edit", "modify", "update", "change", "fix"],
            tool_usage_guide="""
            编辑代码时：
            1. 先读取文件内容
            2. 使用 Search/Replace 格式进行精确编辑
            3. 验证编辑结果
            4. 必要时使用 diff 预览
            """,
            best_practices=[
                "编辑前备份原文件",
                "使用精确的搜索字符串",
                "一次只做一个修改",
                "编辑后立即验证"
            ],
            error_handling=[
                "如果搜索不匹配，检查是否有隐藏字符",
                "如果替换失败，尝试更小的修改范围",
                "保留错误信息以便分析"
            ],
            required_tools=["mcp_filesystem_read", "mcp_filesystem_edit"],
            examples=[]
        ))
        
        # 3. 命令执行 Skill
        self.register(MCPSkill(
            name="command_execution",
            description="执行命令行命令",
            triggers=["run", "execute", "build", "test", "compile"],
            tool_usage_guide="""
            执行命令时：
            1. 确认命令安全性
            2. 设置合理的超时时间
            3. 捕获 stdout 和 stderr
            4. 分析退出码
            """,
            best_practices=[
                "优先使用项目定义的命令",
                "检查命令是否已安装",
                "设置超时防止挂起",
                "记录命令输出"
            ],
            error_handling=[
                "退出码非 0 时分析错误输出",
                "命令未找到时检查 PATH",
                "超时时考虑分批执行"
            ],
            required_tools=["mcp_bash_execute"],
            examples=[]
        ))
    
    def register(self, skill: MCPSkill):
        """注册 Skill"""
        self._skills[skill.name] = skill
    
    def find_skill(self, request: str) -> Optional[MCPSkill]:
        """根据请求查找匹配的 Skill"""
        request_lower = request.lower()
        
        for skill in self._skills.values():
            for trigger in skill.triggers:
                if trigger in request_lower:
                    return skill
        
        return None
    
    def get_skill_prompt(self, skill_name: str) -> str:
        """获取 Skill 的 Prompt 增强"""
        skill = self._skills.get(skill_name)
        if not skill:
            return ""
        
        return f"""
## Skill: {skill.name}

{skill.description}

### 工具使用指南
{skill.tool_usage_guide}

### 最佳实践
{chr(10).join(f"- {p}" for p in skill.best_practices)}

### 错误处理
{chr(10).join(f"- {e}" for e in skill.error_handling)}

### 所需工具
{chr(10).join(f"- {t}" for t in skill.required_tools)}
"""
```

---

### Phase 3: 智能上下文管理（P1 - 1个月）

#### 3.3.1 目标
实现自动上下文选择 + 代码库索引 + 语义搜索，提升大型代码库理解能力。

#### 3.3.2 核心实现

```python
# pyutagent/indexing/code_indexer.py

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from pathlib import Path
import hashlib
import json

@dataclass
class CodeSymbol:
    """代码符号"""
    name: str
    type: str  # class, method, field, interface
    file_path: str
    line_start: int
    line_end: int
    signature: str
    docstring: Optional[str] = None
    dependencies: List[str] = None

@dataclass
class CodeFile:
    """代码文件"""
    path: str
    content_hash: str
    symbols: List[CodeSymbol]
    imports: List[str]
    last_modified: float


class CodeIndexer:
    """代码索引器
    
    为整个代码库建立索引，支持：
    1. 快速符号查找
    2. 依赖关系分析
    3. 变更检测
    """
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.index_path = project_root / ".pyutagent" / "code_index.json"
        self.files: Dict[str, CodeFile] = {}
        self.symbols: Dict[str, List[CodeSymbol]] = {}
    
    async def build_index(self) -> None:
        """构建代码索引"""
        # 扫描所有代码文件
        code_files = list(self.project_root.rglob("*.java"))
        
        for file_path in code_files:
            relative_path = str(file_path.relative_to(self.project_root))
            
            # 解析文件
            code_file = await self._parse_file(file_path)
            self.files[relative_path] = code_file
            
            # 索引符号
            for symbol in code_file.symbols:
                if symbol.name not in self.symbols:
                    self.symbols[symbol.name] = []
                self.symbols[symbol.name].append(symbol)
        
        # 保存索引
        await self._save_index()
    
    async def _parse_file(self, file_path: Path) -> CodeFile:
        """解析代码文件"""
        content = file_path.read_text()
        content_hash = hashlib.md5(content.encode()).hexdigest()
        
        # 使用 Java 解析器提取符号
        symbols = await self._extract_symbols(content, str(file_path))
        imports = self._extract_imports(content)
        
        return CodeFile(
            path=str(file_path.relative_to(self.project_root)),
            content_hash=content_hash,
            symbols=symbols,
            imports=imports,
            last_modified=file_path.stat().st_mtime
        )
    
    async def _extract_symbols(
        self,
        content: str,
        file_path: str
    ) -> List[CodeSymbol]:
        """提取代码符号"""
        # 使用 TreeSitter 或类似工具解析
        symbols = []
        
        # 简单正则提取（实际应使用 AST 解析）
        import re
        
        # 提取类
        class_pattern = r'(?:public\s+|private\s+|protected\s+)?(?:abstract\s+)?class\s+(\w+)'
        for match in re.finditer(class_pattern, content):
            symbols.append(CodeSymbol(
                name=match.group(1),
                type="class",
                file_path=file_path,
                line_start=content[:match.start()].count('\n') + 1,
                line_end=content[:match.start()].count('\n') + 1,
                signature=match.group(0)
            ))
        
        # 提取方法
        method_pattern = r'(?:public\s+|private\s+|protected\s+)?(?:static\s+)?[\w<>\[\]]+\s+(\w+)\s*\([^)]*\)\s*\{'
        for match in re.finditer(method_pattern, content):
            symbols.append(CodeSymbol(
                name=match.group(1),
                type="method",
                file_path=file_path,
                line_start=content[:match.start()].count('\n') + 1,
                line_end=content[:match.start()].count('\n') + 1,
                signature=match.group(0)
            ))
        
        return symbols
    
    def _extract_imports(self, content: str) -> List[str]:
        """提取导入语句"""
        import re
        imports = []
        
        import_pattern = r'import\s+([\w.]+);'
        for match in re.finditer(import_pattern, content):
            imports.append(match.group(1))
        
        return imports
    
    async def _save_index(self) -> None:
        """保存索引到文件"""
        data = {
            "files": {
                path: {
                    "content_hash": file.content_hash,
                    "symbols": [
                        {
                            "name": s.name,
                            "type": s.type,
                            "line_start": s.line_start,
                            "line_end": s.line_end,
                            "signature": s.signature
                        }
                        for s in file.symbols
                    ],
                    "imports": file.imports,
                    "last_modified": file.last_modified
                }
                for path, file in self.files.items()
            },
            "symbols": {
                name: [
                    {
                        "name": s.name,
                        "type": s.type,
                        "file_path": s.file_path,
                        "line_start": s.line_start
                    }
                    for s in symbols
                ]
                for name, symbols in self.symbols.items()
            }
        }
        
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text(json.dumps(data, indent=2))
    
    async def load_index(self) -> bool:
        """加载索引"""
        if not self.index_path.exists():
            return False
        
        try:
            data = json.loads(self.index_path.read_text())
            # 解析索引数据...
            return True
        except Exception:
            return False
    
    def find_symbol(self, name: str) -> List[CodeSymbol]:
        """查找符号"""
        return self.symbols.get(name, [])
    
    def find_references(self, symbol_name: str) -> List[CodeSymbol]:
        """查找符号引用"""
        references = []
        
        for file in self.files.values():
            for symbol in file.symbols:
                if symbol.dependencies and symbol_name in symbol.dependencies:
                    references.append(symbol)
        
        return references


# 语义搜索
class SemanticSearcher:
    """语义搜索器"""
    
    def __init__(self, indexer: CodeIndexer, embedding_client: Any):
        self.indexer = indexer
        self.embeddings = embedding_client
        self._embedding_cache: Dict[str, List[float]] = {}
    
    async def search(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """语义搜索"""
        # 获取查询的 embedding
        query_embedding = await self.embeddings.embed(query)
        
        # 计算相似度
        results = []
        for file in self.indexer.files.values():
            for symbol in file.symbols:
                # 获取符号的 embedding（从缓存或计算）
                symbol_key = f"{symbol.file_path}:{symbol.name}"
                
                if symbol_key not in self._embedding_cache:
                    self._embedding_cache[symbol_key] = await self.embeddings.embed(
                        f"{symbol.type} {symbol.name} {symbol.signature}"
                    )
                
                symbol_embedding = self._embedding_cache[symbol_key]
                
                # 计算余弦相似度
                similarity = self._cosine_similarity(query_embedding, symbol_embedding)
                
                results.append({
                    "symbol": symbol,
                    "similarity": similarity,
                    "file_path": symbol.file_path
                })
        
        # 排序并返回 top_k
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """计算余弦相似度"""
        import math
        
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        
        if norm_a == 0 or norm_b == 0:
            return 0
        
        return dot_product / (norm_a * norm_b)
```

#### 3.3.3 智能上下文选择

```python
# pyutagent/context/smart_selector.py

from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass
class ContextSelection:
    """上下文选择结果"""
    relevant_files: List[str]
    relevant_symbols: List[str]
    summary: str
    estimated_tokens: int


class SmartContextSelector:
    """智能上下文选择器
    
    根据用户请求自动选择最相关的代码上下文。
    """
    
    def __init__(
        self,
        indexer: CodeIndexer,
        semantic_searcher: SemanticSearcher,
        llm_client: Any
    ):
        self.indexer = indexer
        self.searcher = semantic_searcher
        self.llm = llm_client
    
    async def select_context(
        self,
        request: str,
        max_tokens: int = 8000
    ) -> ContextSelection:
        """选择相关上下文"""
        
        # 1. 语义搜索相关符号
        semantic_results = await self.searcher.search(request, top_k=20)
        
        # 2. 提取关键词搜索
        keywords = await self._extract_keywords(request)
        keyword_results = []
        for keyword in keywords:
            keyword_results.extend(self.indexer.find_symbol(keyword))
        
        # 3. 合并结果并去重
        all_symbols = {}
        for result in semantic_results:
            symbol = result["symbol"]
            all_symbols[f"{symbol.file_path}:{symbol.name}"] = symbol
        
        for symbol in keyword_results:
            all_symbols[f"{symbol.file_path}:{symbol.name}"] = symbol
        
        # 4. 使用 LLM 进一步筛选
        selected = await self._llm_filter(request, list(all_symbols.values()), max_tokens)
        
        # 5. 收集相关文件
        relevant_files = list(set(s.file_path for s in selected))
        
        # 6. 生成摘要
        summary = await self._generate_summary(request, selected)
        
        # 7. 估算 Token 数
        estimated_tokens = self._estimate_tokens(selected)
        
        return ContextSelection(
            relevant_files=relevant_files,
            relevant_symbols=[s.name for s in selected],
            summary=summary,
            estimated_tokens=estimated_tokens
        )
    
    async def _extract_keywords(self, request: str) -> List[str]:
        """提取关键词"""
        prompt = f"""
        从以下请求中提取关键代码符号名称（类名、方法名等）：
        
        请求：{request}
        
        返回 JSON 格式：
        {{"keywords": ["keyword1", "keyword2"]}}
        """
        
        response = await self.llm.generate(prompt)
        data = json.loads(response)
        return data.get("keywords", [])
    
    async def _llm_filter(
        self,
        request: str,
        symbols: List[CodeSymbol],
        max_tokens: int
    ) -> List[CodeSymbol]:
        """使用 LLM 筛选最相关的符号"""
        
        # 构建符号列表
        symbol_list = "\n".join([
            f"{i+1}. {s.type} {s.name} in {s.file_path}"
            for i, s in enumerate(symbols[:50])  # 限制数量
        ])
        
        prompt = f"""
        用户请求：{request}
        
        可用代码符号：
        {symbol_list}
        
        请选择与请求最相关的符号编号（最多选择能容纳在 {max_tokens} tokens 内的数量）：
        返回 JSON 格式：{{"selected_indices": [1, 5, 10]}}
        """
        
        response = await self.llm.generate(prompt)
        data = json.loads(response)
        indices = data.get("selected_indices", [])
        
        return [symbols[i-1] for i in indices if 1 <= i <= len(symbols)]
    
    async def _generate_summary(
        self,
        request: str,
        symbols: List[CodeSymbol]
    ) -> str:
        """生成上下文摘要"""
        prompt = f"""
        用户请求：{request}
        
        相关代码符号：
        {chr(10).join(f"- {s.type} {s.name}" for s in symbols[:10])}
        
        请生成一个简短的上下文摘要（50字以内）：
        """
        
        return await self.llm.generate(prompt)
    
    def _estimate_tokens(self, symbols: List[CodeSymbol]) -> int:
        """估算 Token 数量"""
        # 粗略估算：每个符号约 100 tokens
        return len(symbols) * 100
```

---

### Phase 4: 多文件编辑能力（P1 - 1个月）

#### 3.4.1 目标
实现多文件编辑支持 + 智能影响分析 + Diff 预览。

#### 3.4.2 核心实现

```python
# pyutagent/editor/multi_file_editor.py

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from pathlib import Path
import difflib

@dataclass
class FileEdit:
    """文件编辑操作"""
    file_path: str
    old_content: str
    new_content: str
    description: str

@dataclass
class EditGroup:
    """编辑组（原子操作）"""
    id: str
    description: str
    edits: List[FileEdit]
    dependencies: List[str]  # 依赖的其他编辑组

@dataclass
class EditPlan:
    """编辑计划"""
    groups: List[EditGroup]
    rollback_plan: List[FileEdit]


class MultiFileEditor:
    """多文件编辑器
    
    支持复杂的多文件重构操作。
    """
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self._backup: Dict[str, str] = {}
    
    async def create_edit_plan(
        self,
        request: str,
        target_files: List[str],
        llm_client: Any
    ) -> EditPlan:
        """创建编辑计划"""
        
        # 1. 读取所有目标文件
        file_contents = {}
        for file_path in target_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                file_contents[file_path] = full_path.read_text()
        
        # 2. 使用 LLM 生成编辑计划
        prompt = f"""
        用户请求：{request}
        
        目标文件：
        {chr(10).join(target_files)}
        
        请生成详细的编辑计划，返回 JSON 格式：
        {{
            "groups": [
                {{
                    "id": "group_1",
                    "description": "编辑组描述",
                    "edits": [
                        {{
                            "file_path": "文件路径",
                            "search": "要搜索的内容",
                            "replace": "替换为的内容",
                            "description": "编辑描述"
                        }}
                    ],
                    "dependencies": []
                }}
            ]
        }}
        """
        
        response = await llm_client.generate(prompt)
        plan_data = json.loads(response)
        
        # 3. 构建 EditPlan
        groups = []
        for group_data in plan_data.get("groups", []):
            edits = []
            for edit_data in group_data.get("edits", []):
                file_path = edit_data["file_path"]
                old_content = file_contents.get(file_path, "")
                
                # 应用 search/replace
                search = edit_data["search"]
                replace = edit_data["replace"]
                new_content = old_content.replace(search, replace, 1)
                
                edits.append(FileEdit(
                    file_path=file_path,
                    old_content=old_content,
                    new_content=new_content,
                    description=edit_data["description"]
                ))
            
            groups.append(EditGroup(
                id=group_data["id"],
                description=group_data["description"],
                edits=edits,
                dependencies=group_data.get("dependencies", [])
            ))
        
        # 4. 生成回滚计划
        rollback_plan = [
            FileEdit(
                file_path=path,
                old_content="",
                new_content=content,
                description=f"Restore {path}"
            )
            for path, content in file_contents.items()
        ]
        
        return EditPlan(groups=groups, rollback_plan=rollback_plan)
    
    async def preview_edits(self, plan: EditPlan) -> str:
        """生成编辑预览（Diff 格式）"""
        preview = []
        
        for group in plan.groups:
            preview.append(f"\n## {group.description}\n")
            
            for edit in group.edits:
                preview.append(f"\n### {edit.file_path}\n")
                preview.append("```diff\n")
                
                # 生成 unified diff
                diff = difflib.unified_diff(
                    edit.old_content.splitlines(keepends=True),
                    edit.new_content.splitlines(keepends=True),
                    fromfile=f"a/{edit.file_path}",
                    tofile=f"b/{edit.file_path}"
                )
                
                preview.extend(diff)
                preview.append("```\n")
        
        return "".join(preview)
    
    async def apply_edits(
        self,
        plan: EditPlan,
        confirm: bool = True
    ) -> Dict[str, Any]:
        """应用编辑"""
        
        if confirm:
            # 显示预览并等待确认
            preview = await self.preview_edits(plan)
            print(preview)
            # 实际实现中这里应该等待用户确认
        
        # 备份原文件
        for group in plan.groups:
            for edit in group.edits:
                if edit.file_path not in self._backup:
                    file_path = self.project_root / edit.file_path
                    if file_path.exists():
                        self._backup[edit.file_path] = file_path.read_text()
        
        # 按依赖顺序应用编辑
        applied = []
        failed = []
        
        for group in self._sort_groups(plan.groups):
            for edit in group.edits:
                try:
                    file_path = self.project_root / edit.file_path
                    file_path.write_text(edit.new_content)
                    applied.append(edit.file_path)
                except Exception as e:
                    failed.append({"file": edit.file_path, "error": str(e)})
        
        return {
            "success": len(failed) == 0,
            "applied": applied,
            "failed": failed
        }
    
    async def rollback(self) -> Dict[str, Any]:
        """回滚所有编辑"""
        restored = []
        failed = []
        
        for file_path, content in self._backup.items():
            try:
                full_path = self.project_root / file_path
                full_path.write_text(content)
                restored.append(file_path)
            except Exception as e:
                failed.append({"file": file_path, "error": str(e)})
        
        self._backup.clear()
        
        return {
            "success": len(failed) == 0,
            "restored": restored,
            "failed": failed
        }
    
    def _sort_groups(self, groups: List[EditGroup]) -> List[EditGroup]:
        """按依赖排序编辑组"""
        # 拓扑排序
        group_map = {g.id: g for g in groups}
        visited = set()
        result = []
        
        def visit(group_id: str):
            if group_id in visited:
                return
            visited.add(group_id)
            
            group = group_map.get(group_id)
            if group:
                for dep in group.dependencies:
                    visit(dep)
                result.append(group)
        
        for group in groups:
            visit(group.id)
        
        return result


# 影响分析器
class ImpactAnalyzer:
    """代码变更影响分析器"""
    
    def __init__(self, indexer: CodeIndexer):
        self.indexer = indexer
    
    async def analyze_impact(
        self,
        file_path: str,
        change_description: str
    ) -> Dict[str, Any]:
        """分析变更影响范围"""
        
        # 1. 找到文件中的符号
        file = self.indexer.files.get(file_path)
        if not file:
            return {"error": "File not found in index"}
        
        # 2. 分析每个符号的影响
        impacts = []
        for symbol in file.symbols:
            # 查找引用
            references = self.indexer.find_references(symbol.name)
            
            impacts.append({
                "symbol": symbol.name,
                "type": symbol.type,
                "references": [
                    {"file": r.file_path, "name": r.name}
                    for r in references
                ],
                "impact_level": "high" if len(references) > 5 else "medium" if len(references) > 0 else "low"
            })
        
        # 3. 生成影响报告
        high_impact = [i for i in impacts if i["impact_level"] == "high"]
        medium_impact = [i for i in impacts if i["impact_level"] == "medium"]
        
        return {
            "file": file_path,
            "change": change_description,
            "total_symbols": len(file.symbols),
            "high_impact_changes": high_impact,
            "medium_impact_changes": medium_impact,
            "affected_files": list(set(
                ref["file"]
                for impact in impacts
                for ref in impact["references"]
            )),
            "recommendation": "需要仔细测试" if high_impact else "标准测试流程"
        }
```

---

### Phase 5: 项目配置系统（P1 - 1个月）

#### 3.5.1 目标
实现 PYUT.md 项目配置系统，类似于 Claude Code 的 CLAUDE.md。

#### 3.5.2 核心实现

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

### Phase 6: 专业化 Subagents（P1 - 1个月）

#### 3.6.1 目标
实现三大专业化子代理：BashAgent、PlanAgent、ExploreAgent。

#### 3.6.2 核心实现

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

### Phase 7: 长期规划能力（P2 - 1个月）

#### 3.7.1 目标
实现上下文压缩 + 检查点机制 + 长期任务支持。

#### 3.7.2 核心实现

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
| **Phase 2** | MCP 生态集成 | P0 | 1-2个月 | 无 |
| **Phase 3** | 智能上下文管理 | P1 | 1个月 | Phase 2 |
| **Phase 4** | 多文件编辑能力 | P1 | 1个月 | Phase 3 |
| **Phase 5** | 项目配置系统 | P1 | 1个月 | 无 |
| **Phase 6** | 专业化 Subagents | P1 | 1个月 | Phase 1 |
| **Phase 7** | 长期规划能力 | P2 | 1个月 | Phase 1 |

### 4.2 关键里程碑

```
Month 1: Phase 1 & 2 完成
├── Week 1-2: 通用任务规划器
├── Week 3-4: MCP 客户端实现
└── Week 5-6: Skills 系统集成

Month 2: Phase 3 & 4 & 5 完成
├── Week 7-8: 代码索引 + 语义搜索
├── Week 9-10: 多文件编辑器
└── Week 11-12: PYUT.md 配置系统

Month 3: Phase 6 & 7 完成
├── Week 13-14: 专业化 Subagents
└── Week 15-16: 上下文压缩 + 检查点
```

### 4.3 预期效果

| 改进项 | 改进前 | 改进后 |
|--------|--------|--------|
| 任务范围 | 仅 UT 生成 | 任意编程任务 |
| 工具生态 | 封闭系统 | MCP 生态 |
| 上下文理解 | 手动配置 | 自动选择 |
| 代码编辑 | 单文件 | 多文件重构 |
| 项目理解 | 每次重新分析 | PYUT.md 持久化 |
| 任务分工 | 单一 Agent | 专业化 Subagents |
| 长任务支持 | 上下文丢失 | 自动压缩续传 |

---

## 五、总结

本改进计划提出了将 PyUT Agent 从专用 UT 生成工具进化为通用 Coding Agent 的完整路线图。通过实施这 7 个阶段，PyUT Agent 将具备：

1. **通用任务规划**：支持 8+ 种任务类型的自动识别和处理
2. **MCP 生态集成**：接入丰富的工具生态
3. **智能上下文管理**：自动选择相关代码上下文
4. **多文件编辑能力**：支持复杂重构操作
5. **项目配置系统**：快速理解项目上下文
6. **专业化 Subagents**：任务分工，提升效率
7. **长期规划能力**：支持数小时的复杂任务

这些改进将使 PyUT Agent 在保持 UT 生成领域优势的同时，具备处理更广泛编程任务的能力，向真正的"自主编程 Agent"迈进。

---

**参考资源**：
- [Claude Code 高级功能全解析](https://www.cnblogs.com/dqtx33/p/19488109)
- [Coding Agent 的进化之路](https://juejin.cn/post/7607358297457475584)
- [Claude Skills 与 MCP 的关系](http://m.toutiao.com/group/7596997791998591488/)
- [OpenCode 超级详细入门指南](http://m.toutiao.com/group/7592244035357393446/)
- [MCP 协议规范](https://modelcontextprotocol.io/)

**计划制定日期**：2026-03-05
