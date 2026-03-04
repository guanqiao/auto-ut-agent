# 增强工具使用能力 - 对标顶级Coding Agent

## 背景分析

### 当前状态

**已有基础设施**：
- ✅ 工具框架：`pyutagent/tools/tool.py` - Tool基类、ToolDefinition、ToolResult
- ✅ 标准工具：`pyutagent/tools/standard_tools.py` - ReadTool, WriteTool, EditTool, GlobTool, GrepTool, BashTool
- ✅ 工具注册：`pyutagent/tools/tool_registry.py` - ToolRegistry、schema生成
- ✅ 工具编排：`pyutagent/agent/tool_orchestrator.py` - DependencyGraph、ExecutionPlan、自适应执行
- ✅ MCP集成：`pyutagent/tools/mcp_integration.py` - MCPClient、MCPToolAdapter、MCPManager

**关键Gap**：
- ❌ ReAct Agent未使用标准工具
- ❌ 缺乏Git工具（git_status, git_diff, git_commit等）
- ❌ 工具调用是预设流程，非自主决策
- ❌ 缺乏完整的自主纠错循环（Observe-Think-Act-Verify-Learn）

---

## 计划概览

### 阶段1：基础工具能力集成（1周）
1. [P1] 创建Agent工具服务层 - 统一管理工具注册和执行
2. [P1] 为ReActAgent集成标准工具 - 让Agent能调用read/write/edit/bash/grep
3. [P1] 添加Git工具支持 - git_status, git_diff, git_commit, git_branch
4. [P1] 修改Agent提示词 - 引导LLM使用工具而非直接生成

### 阶段2：工具使用智能化（1周）
5. [P1] 增强工具编排器 - 支持自主工具选择
6. [P1] 实现观察-思考-行动-验证循环 - Autonomous Loop基础
7. [P2] 添加工具使用记忆 - 学习成功工具调用模式

### 阶段3：高级工具能力（1周）
8. [P2] 增强MCP集成 - 自动发现和连接MCP服务器
9. [P2] 添加工具安全控制 - 权限分级和确认机制
10. [P2] 实现工具组合能力 - 自动组合工具完成复杂任务

---

## 详细任务

### 阶段1：基础工具能力集成

#### Task 1: 创建Agent工具服务层
**文件**: `pyutagent/agent/tool_service.py` (新文件)

创建统一的工具服务层，整合现有工具资源：

```python
class AgentToolService:
    """Agent工具服务 - 统一管理工具"""
    
    def __init__(self, project_path: str):
        self.registry = ToolRegistry()
        self.executor = ToolExecutor()
        self._register_standard_tools()
        self._register_specialized_tools()
    
    def _register_standard_tools(self):
        # 注册 read_file, write_file, edit_file, glob, grep, bash
        pass
    
    def _register_specialized_tools(self):
        # 注册 Java解析、Maven构建等专用工具
        pass
    
    def get_schemas_for_llm(self) -> List[Dict]:
        """获取LLM可用的工具schema"""
        pass
    
    async def execute_tool(self, tool_name: str, params: Dict) -> ToolResult:
        """执行工具并返回结果"""
        pass
```

**验收标准**：
- 工具服务可以正常初始化
- 可以获取工具schema
- 可以执行工具并返回结果

---

#### Task 2: 为ReActAgent集成标准工具
**修改文件**: `pyutagent/agent/react_agent.py`

在ReActAgent中集成工具服务：

1. 初始化时创建AgentToolService
2. 在反馈循环中添加工具调用步骤
3. 修改提示词引导LLM使用工具

**提示词修改示例**：
```
你是一个UT生成专家。在生成代码时：
1. 首先使用 read_file 工具读取源文件
2. 使用 grep 工具搜索相关代码
3. 使用 write_file 工具创建测试文件
4. 使用 bash 工具运行编译和测试
```

**验收标准**：
- ReActAgent可以调用工具
- 工具执行结果能反馈到LLM
- 能完成基本的"读取-修改-执行"循环

---

#### Task 3: 添加Git工具支持
**文件**: `pyutagent/tools/git_tools.py` (新文件)

实现Git工具类：

```python
class GitStatusTool(BaseFileTool):
    """Git status - 查看仓库状态"""
    
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="git_status",
            description="查看Git仓库的当前状态",
            category=ToolCategory.COMMAND,
            parameters=[...]
        )
    
    async def execute(self, **kwargs) -> ToolResult:
        # 实现 git status
        pass


class GitDiffTool(BaseFileTool):
    """Git diff - 查看文件变化"""
    
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="git_diff",
            description="查看文件的更改内容",
            category=ToolCategory.COMMAND,
            parameters=[
                create_tool_parameter("file_path", "string", "文件路径", required=False),
                create_tool_parameter("staged", "boolean", "仅显示暂存区", required=False, default=False),
            ]
        )
    
    async def execute(self, **kwargs) -> ToolResult:
        # 实现 git diff
        pass


class GitCommitTool(BaseFileTool):
    """Git commit - 提交更改"""
    
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="git_commit",
            description="提交更改到Git仓库",
            category=ToolCategory.COMMAND,
            parameters=[
                create_tool_parameter("message", "string", "提交信息", required=True),
                create_tool_parameter("add_all", "boolean", "是否添加所有文件", required=False, default=False),
            ]
        )


class GitBranchTool(BaseFileTool):
    """Git branch - 分支管理"""
    
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="git_branch",
            description="列出、创建或删除Git分支",
            category=ToolCategory.COMMAND,
            parameters=[
                create_tool_parameter("action", "string", "操作: list/create/delete", required=False, default="list"),
                create_tool_parameter("branch_name", "string", "分支名", required=False),
            ]
        )
```

**验收标准**：
- 可以执行 git status
- 可以执行 git diff
- 可以执行 git commit
- 可以执行 git branch

---

#### Task 4: 修改Agent提示词引导工具使用
**修改文件**: `pyutagent/agent/prompts.py`

更新系统提示词，明确引导LLM使用工具：

```python
TOOL_USAGE_SYSTEM_PROMPT = """你是一个智能编程助手，具有强大的工具使用能力。

## 可用工具
你可以通过以下工具与系统交互：

### 文件操作
- read_file: 读取文件内容
- write_file: 创建或写入文件
- edit_file: 修改文件（Search/Replace）
- glob: 查找匹配模式的文件

### 代码搜索
- grep: 在代码中搜索文本或正则表达式

### 命令执行
- bash: 执行shell命令

### Git操作
- git_status: 查看仓库状态
- git_diff: 查看文件更改
- git_commit: 提交更改
- git_branch: 分支管理

## 工具使用原则
1. 优先使用工具完成操作，不要直接生成代码
2. 读取文件使用 read_file，不要假设文件内容
3. 执行构建和测试使用 bash
4. 提交前使用 git_diff 检查更改
"""
```

**验收标准**：
- LLM能理解工具用途
- 生成的响应包含工具调用

---

### 阶段2：工具使用智能化

#### Task 5: 增强工具编排器
**修改文件**: `pyutagent/agent/tool_orchestrator.py`

增强工具编排器，支持：
- 基于目标的智能工具选择
- 动态工具链规划
- 工具执行结果推理

```python
class EnhancedToolOrchestrator(ToolOrchestrator):
    """增强版工具编排器"""
    
    async def plan_from_goal(self, goal: str, context: Dict) -> ExecutionPlan:
        """根据目标自动规划工具调用"""
        # 使用LLM分析目标，选择合适的工具
        analysis_prompt = f"""
        分析以下目标，确定需要哪些工具：
        目标: {goal}
        可用工具: {self.list_available_tools()}
        
        返回JSON格式的工具调用计划：
        """
        # 解析LLM响应，生成执行计划
        pass
```

**验收标准**：
- 可以根据自然语言目标自动生成工具调用计划
- 计划包含合理的工具依赖顺序

---

#### Task 6: 实现自主纠错循环基础
**文件**: `pyutagent/agent/autonomous_loop.py` (新文件)

实现基础的自主循环（Observe-Think-Act-Verify）：

```python
class AutonomousLoop:
    """自主纠错循环"""
    
    def __init__(
        self,
        tool_service: AgentToolService,
        llm_client: LLMClient,
        max_iterations: int = 10
    ):
        self.tool_service = tool_service
        self.llm_client = llm_client
        self.max_iterations = max_iterations
    
    async def run(self, task: str, context: Dict) -> LoopResult:
        """运行自主循环"""
        
        state = LoopState(task=task)
        
        for iteration in range(self.max_iterations):
            # 1. 观察 (Observe)
            observation = await self._observe(state)
            
            # 2. 思考 (Think)
            thought = await self._think(state, observation)
            
            # 3. 行动 (Act) - 可能调用工具
            result = await self._act(state, thought)
            
            # 4. 验证 (Verify)
            verified = await self._verify(state, result)
            
            if verified:
                return LoopResult(success=True, iterations=iteration + 1)
            
            # 5. 学习 - 记录失败模式
            await self._learn(state, result)
        
        return LoopResult(success=False, iterations=self.max_iterations)
```

**验收标准**：
- 循环能正常运行
- 能根据结果判断是否需要继续
- 失败时能调整策略

---

#### Task 7: 添加工具使用记忆
**文件**: `pyutagent/memory/tool_memory.py` (新文件)

记录工具调用成功/失败模式：

```python
class ToolMemory:
    """工具使用记忆 - 学习成功模式"""
    
    async def record_success(
        self,
        tool_name: str,
        params: Dict,
        context: Dict,
        result: Any
    ):
        """记录成功的工具调用"""
        
    async def record_failure(
        self,
        tool_name: str,
        params: Dict,
        context: Dict,
        error: str
    ):
        """记录失败的调用"""
    
    async def get_recommended_tools(
        self,
        task_type: str
    ) -> List[Dict]:
        """获取推荐的工具调用模式"""
```

**验收标准**：
- 能记录工具调用历史
- 能根据历史推荐工具

---

### 阶段3：高级工具能力

#### Task 8: 增强MCP集成
**修改文件**: `pyutagent/tools/mcp_integration.py`

增强功能：
- 自动发现MCP服务器
- 动态加载MCP工具
- MCP工具适配

```python
class EnhancedMCPManager(MCPManager):
    """增强版MCP管理器"""
    
    async def auto_discover(self):
        """自动发现可用的MCP服务器"""
        # 扫描常见MCP服务器位置
        # 检查环境变量中的MCP配置
        pass
    
    async def load_mcp_tools(self, config_path: str):
        """从配置文件加载MCP工具"""
        pass
```

**验收标准**：
- 能自动发现MCP服务器
- 能动态加载MCP工具到工具注册表

---

#### Task 9: 添加工具安全控制
**文件**: `pyutagent/tools/safe_executor.py` (新文件)

实现工具安全控制：

```python
class ToolSecurityLevel(Enum):
    """工具安全级别"""
    SAFE = 1      # 只读操作
    NORMAL = 2    # 常规操作
    CAUTION = 3   # 需要谨慎
    DANGEROUS = 4 # 需要确认


class SafeToolExecutor:
    """安全工具执行器"""
    
    def __init__(self):
        self.security_config = {
            "read_file": ToolSecurityLevel.SAFE,
            "write_file": ToolSecurityLevel.CAUTION,
            "bash": ToolSecurityLevel.DANGEROUS,
            "git_commit": ToolSecurityLevel.DANGEROUS,
        }
    
    async def execute_with_check(
        self,
        tool_name: str,
        params: Dict,
        require_confirmation: bool = False
    ) -> ToolResult:
        """带安全检查的工具执行"""
        pass
```

**验收标准**：
- 危险操作需要确认
- 可配置安全级别

---

#### Task 10: 实现工具组合能力
**修改文件**: `pyutagent/agent/tool_composer.py` (新文件)

实现自动工具组合：

```python
class ToolComposer:
    """工具组合器 - 自动组合工具完成复杂任务"""
    
    async def compose(
        self,
        goal: str,
        available_tools: List[Tool]
    ) -> List[ToolCall]:
        """自动组合工具完成目标"""
        
        # 分析目标
        # 选择工具序列
        # 确定参数
        # 返回工具调用计划
        pass
```

**验收标准**：
- 能分析复杂任务
- 能生成工具调用序列

---

## 实现顺序

1. **Task 1**: 创建AgentToolService - 基础设施
2. **Task 3**: 添加Git工具 - 补充缺失工具
3. **Task 4**: 修改提示词 - 引导LLM使用工具
4. **Task 2**: 集成到ReActAgent - 激活工具使用
5. **Task 5**: 增强工具编排器 - 智能化
6. **Task 6**: 自主纠错循环 - 自主能力
7. **Task 7**: 工具使用记忆 - 学习能力
8. **Task 8-10**: 高级功能 - MCP、安全、组合

---

## 测试策略

每个任务需要：
1. 单元测试 - 测试工具功能
2. 集成测试 - 测试Agent工具协作
3. E2E测试 - 测试完整流程

---

## 风险与应对

| 风险 | 影响 | 应对 |
|-----|-----|-----|
| LLM不按预期使用工具 | 高 | 优化提示词，增加示例 |
| 工具执行失败 | 中 | 完善错误处理和重试 |
| 性能下降 | 中 | 优化工具缓存 |
| 安全风险 | 高 | 严格安全控制 |

---

## 成功指标

- ✅ Agent能使用标准工具完成文件操作
- ✅ Agent能使用Git工具进行版本管理
- ✅ Agent能根据任务自主选择工具
- ✅ 工具使用有记忆学习能力
- ✅ MCP工具可动态加载
- ✅ 安全控制可配置
