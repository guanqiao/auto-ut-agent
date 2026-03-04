# Coding Agent 核心能力 Gap 分析报告

## 一、执行摘要

基于对 Claude Code、OpenCode、Trae 等顶级 Coding Agent 的深入研究，结合 PyUT Agent 现有架构分析，本报告识别了 **9 个核心能力维度** 的差距，并制定了 **分阶段改进路线图**。

### 核心发现

| 差距级别 | 数量 | 说明 |
|---------|------|------|
| 🔴 重大差距 | 3 | Skills 机制、语音交互、通用任务规划 |
| 🟡 中等差距 | 4 | 远程 MCP、IDE 集成、用户协作、知识推理 |
| 🟢 小差距 | 2 | CLI/TUI 体验、流式输出 |

---

## 二、Claude Code 核心能力深度分析

### 2.1 Claude Code 是什么

Claude Code 是 Anthropic 于 2025 年 2 月推出的官方终端 AI 编程工具，是 **大模型厂商直接下场做工具** 的代表作。它不仅是一个代码编写工具，更是一个可以通过自然语言指令完成各种电脑任务的智能助手 [1]。

### 2.2 Claude Code 核心能力矩阵

| 能力 | 描述 | 创新点 |
|------|------|--------|
| **终端交互** | 原生 CLI/TUI 界面，支持 /voice 语音模式 | 长按空格键说话，松开完成输入，转录 Token 免费 [1] |
| **Skills 机制** | 自定义技能系统，"使用外部工具的使用说明书" | 教 AI 如何正确使用 MCP 工具，而非直接调用 [4] |
| **MCP 远程服务器** | 2025年6月支持远程 MCP，安全的资源访问 | 支持远程 MCP 服务器，安全地访问各类数据源 [2] |
| **ACP 插件** | 2025年支持接入 IDEA，实现 IDE 深度集成 | IntelliJ IDEA 2024.2 到 2025.x 版本支持 [3] |
| **文件系统访问** | 读取、修改、执行任意文件 | 与 Claude Code 对话式完成编程任务 |
| **Shell 命令执行** | 运行任意终端命令 | 完整的系统操作能力 |
| **Git 操作** | git status、diff、commit 等 | 集成版本控制 |
| **浏览器访问** | 访问文档、搜索解决方案 | 实时信息获取 |
| **自主循环** | Observe → Think → Act → Verify → Learn | 完全自主的问题解决循环 |

### 2.3 Claude Code 独特创新

#### 2.3.1 Skills 机制

Skills 是 Claude Code 的核心创新，它不是"直接调用工具然后祈祷它别乱用"，而是"教 Claude 如何正确使用 MCP 工具的使用说明书" [4]。

```
Skill = 工具使用说明书 + 最佳实践 + 错误处理
```

#### 2.3.2 语音交互模式

通过 `/voice` 命令开启语音模式，长按空格键说话，松开完成输入。这标志着"编程进入对讲机时代" [1]。

#### 2.3.3 远程 MCP 支持

2025年6月发布远程 MCP 支持，可以安全地从远程服务器访问各类数据源，不再局限于本地资源 [2]。

---

## 三、OpenCode 核心能力分析

### 3.1 OpenCode 定位

OpenCode 是专为终端设计的 AI 编程代理(AI Coding Agent)，它将先进的大语言模型(LLM)能力无缝集成到命令行工作流中。通过原生 TUI(终端用户界面)提供沉浸式开发体验 [5]。

### 3.2 OpenCode 核心特性

| 能力 | 描述 |
|------|------|
| **原生 TUI** | 终端用户界面，与命令行工作流深度集成 |
| **开源生态** | 开源项目，吸引社区贡献 |
| **多 LLM 支持** | 支持多种大语言模型后端 |
| **轻量级** | 专注于终端体验 |

### 3.3 与 Claude Code 对比

| 维度 | Claude Code | OpenCode |
|------|------------|----------|
| **定位** | 大厂官方工具 | 开源社区项目 |
| **界面** | CLI + 语音 | 原生 TUI |
| **生态** | Anthropic 生态 | 开放生态 |
| **创新** | Skills + 语音 | 轻量终端 |

---

## 四、Trae 核心能力分析

### 4.1 Trae 定位

Trae 是字节跳动推出的 AI 编程工具，主打**中国免费方案**，对标 Cursor。2025年 AI 编程工具市场形成四大主力阵营：Cursor(AI原生IDE)、GitHub Copilot(行业标准)、Trae(中国免费方案)和 Claude Code(终端交互) [6]。

### 4.2 Trae 核心特性

| 能力 | 描述 |
|------|------|
| **免费策略** | 完全免费，中国市场专属优势 |
| **中文本地化** | 完整的中文界面和文档 |
| **VS Code 深度集成** | 基于 VS Code 的 AI 增强 |
| **Agent 模式** | 类似于 Cursor 的 Agent 能力 |
| **智能补全** | Tab 代码补全 + 对话式生成 |

---

## 五、PyUT Agent 当前能力评估

### 5.1 已具备的核心能力（从 README 梳理）

#### P0 核心能力 ✅
- Agent 架构（ReAct Agent）
- 对话式 UI（PyQt6）
- 记忆系统（多层记忆 + 向量）
- 暂停/恢复
- 覆盖率分析（JaCoCo）
- 多 LLM 支持

#### P1 增强能力 ✅
- 流式代码生成
- 智能增量编辑
- 错误模式学习
- 提示词优化
- 多构建工具（Maven/Gradle/Bazel）
- 静态分析集成
- MCP 集成

#### P2 增强能力 ✅
- 多智能体协作
- 上下文智能压缩
- 多文件协调
- 并行恢复
- 性能监控

#### P3 高级能力 ✅
- 错误预测（12种类型）
- 自适应策略（ε-贪婪）
- 工具沙箱
- 检查点恢复
- 智能代码分析
- 用户交互

#### 架构重构（2026-03-04）✅
- 事件驱动架构
- 组件化系统
- 状态管理
- 多级缓存
- 智能聚类
- 性能监控

### 5.2 架构优势

| 优势 | 说明 |
|------|------|
| **事件驱动** | EventBus 组件完全解耦 |
| **状态管理** | Redux 风格 StateStore |
| **多级缓存** | L1 内存 + L2 磁盘，5-10倍性能提升 |
| **组件化** | ComponentRegistry 装饰器注册 |
| **测试覆盖** | 290+ 测试，100% 通过率 |

---

## 六、核心能力 Gap 矩阵

### 6.1 完整 Gap 分析

| 能力维度 | Claude Code | OpenCode | Trae | PyUT Agent | 差距 | 优先级 |
|---------|------------|----------|------|------------|------|--------|
| **1. Skills 机制** | ✅ 完整 | ❌ | ❌ | ❌ | 🔴 重大 | P0 |
| **2. 语音交互** | ✅ /voice | ❌ | ❌ | ❌ | 🔴 重大 | P1 |
| **3. 通用任务规划** | ✅ 任意任务 | ⚠️ 有限 | ✅ | ❌ 仅 UT | 🔴 重大 | P0 |
| **4. 远程 MCP** | ✅ 远程服务器 | ⚠️ 基础 | ⚠️ 基础 | ⚠️ 基础 | 🟡 中等 | P1 |
| **5. IDE 集成** | ✅ ACP/VS Code | ❌ | ✅ VS Code | ❌ 独立 | 🟡 中等 | P1 |
| **6. 用户协作** | ✅ 多模式 | ⚠️ 基础 | ✅ | ⚠️ 基础 | 🟡 中等 | P2 |
| **7. CLI/TUI 体验** | ✅ 优秀 TUI | ✅ 优秀 | ⚠️ | ⚠️ 基础 | 🟢 轻微 | P2 |
| **8. 知识推理** | ✅ 强 | ⚠️ 中 | ✅ | ⚠️ 规则+LLM | 🟡 中等 | P2 |
| **9. 自主循环** | ✅ 完整自主 | ⚠️ 有限 | ✅ | ⚠️ 预设流程 | 🟡 中等 | P1 |

### 6.2 Gap 优先级解释

#### P0 - 必须立即改进（3项）

1. **Skills 机制**：Claude Code 的核心创新，是"使用工具的使用说明书"
2. **通用任务规划**：当前仅限于 UT 生成，无法处理其他编程任务
3. **语音交互**：行业趋势，"编程进入对讲机时代"

#### P1 - 重要改进（3项）

4. **远程 MCP**：扩展工具生态，支持远程服务器
5. **IDE 集成**：深度嵌入主流 IDE（VS Code/IDEA）
6. **自主循环**：从预设流程到完全自主

#### P2 - 体验优化（3项）

7. **CLI/TUI 体验**：提升终端用户体验
8. **知识推理**：基于文档和知识的推理能力
9. **用户协作**：灵活的确认/建议/拒绝模式

---

## 七、详细 Gap 分析

### Gap 1: Skills 机制（重大差距）

#### Claude Code 的 Skills

Skills =「使用外部工具的使用说明书 + 最佳实践 + 错误处理」

不是"直接调用工具然后祈祷它别乱用"，而是"教 Claude 如何正确使用 MCP 工具"。

#### PyUT Agent 现状

- 仅有基础 MCP 集成
- 缺乏 Skills 机制
- 无法自定义技能

#### 改进方案

```python
class Skill:
    """技能定义"""
    name: str
    description: str
    tool_usage_guide: str  # 如何正确使用工具
    best_practices: List[str]
    error_handling: List[str]
    examples: List[Example]
```

---

### Gap 2: 语音交互（重大差距）

#### Claude Code 的语音交互

- `/voice` 命令开启语音模式
- 长按空格键说话，松开完成输入
- 转录 Token 免费
- "编程进入对讲机时代"

#### PyUT Agent 现状

- 仅支持文本输入
- 无语音交互能力

#### 改进方案

1. 集成语音识别（Whisper/云服务）
2. 设计语音交互协议
3. 支持语音指令执行

---

### Gap 3: 通用任务规划（重大差距）

#### Claude Code 的任务理解

- 理解任意编程需求："为这个项目添加登录功能"
- 自动分解任务并执行
- 动态调整计划

#### PyUT Agent 现状

- 仅限于："为 UserService 生成测试"
- 预设流程：解析→生成→编译→测试→分析

#### 改进方案

```python
class UniversalPlanner:
    """通用任务规划器"""
    
    async def understand_task(
        self,
        user_request: str,
        project_context: ProjectContext
    ) -> TaskUnderstanding:
        """理解任意编程任务"""
        pass
    
    async def decompose_task(
        self,
        understanding: TaskUnderstanding
    ) -> List[Subtask]:
        """分解为可执行子任务"""
        pass
    
    async def execute_with_feedback(
        self,
        plan: List[Subtask],
        progress_callback: Callable
    ) -> ExecutionResult:
        """执行计划并动态调整"""
        pass
```

---

### Gap 4-9: 其他差距分析

#### Gap 4: 远程 MCP

- Claude Code：支持远程 MCP 服务器，安全访问远程资源
- PyUT：仅有基础 MCP 实现

#### Gap 5: IDE 集成

- Claude Code：ACP 插件接入 IDEA/VS Code
- Trae：深度集成 VS Code
- PyUT：独立 GUI 应用

#### Gap 6: 用户协作

- Claude Code：全自动、建议确认、每步确认、人工审查多种模式
- PyUT：基础对话交互

#### Gap 7: CLI/TUI

- Claude Code：优秀 TUI + 语音
- OpenCode：原生终端体验
- PyUT：基础 CLI

#### Gap 8: 知识推理

- Claude Code：基于文档和知识推理
- PyUT：规则 + LLM

#### Gap 9: 自主循环

- Claude Code：Observe→Think→Act→Verify→Learn
- PyUT：预设流程执行

---

## 八、改进路线图

### Phase 1: 核心能力补齐（1-3个月）

#### 1.1 Skills 机制（P0）

| 任务 | 描述 | 优先级 |
|------|------|--------|
| Skill 数据模型设计 | 定义 Skill 结构 | P0 |
| Skill 注册系统 | 装饰器/配置注册 | P0 |
| Skill 执行引擎 | Skill 加载和执行 | P0 |
| 内置 Skills | 常用技能模板 | P1 |

#### 1.2 通用任务规划（P0）

| 任务 | 描述 | 优先级 |
|------|------|--------|
| 任务理解层 | 理解任意编程需求 | P0 |
| 任务分解器 | 分解为子任务 | P0 |
| 计划执行器 | 执行并动态调整 | P0 |
| 任务类型扩展 | 支持重构/Bug修复/功能添加 | P1 |

#### 1.3 语音交互（P1）

| 任务 | 描述 | 优先级 |
|------|------|--------|
| 语音识别集成 | Whisper/云服务 | P1 |
| 语音指令解析 | 解析语音为指令 | P1 |
| 语音反馈 | 语音合成输出 | P2 |

### Phase 2: 生态增强（3-6个月）

#### 2.1 远程 MCP

| 任务 | 描述 | 优先级 |
|------|------|--------|
| MCP 远程服务器 | 支持远程连接 | P1 |
| MCP 安全机制 | 认证和授权 | P1 |
| MCP 发现服务 | 自动发现可用服务 | P2 |

#### 2.2 IDE 集成

| 任务 | 描述 | 优先级 |
|------|------|--------|
| VS Code 插件 | 深度集成 VS Code | P1 |
| IDEA 插件 | 集成 IDEA（ACP 协议） | P1 |
| Vim/Neovim 支持 | 终端编辑器支持 | P2 |

#### 2.3 自主循环

| 任务 | 描述 | 优先级 |
|------|------|--------|
| 观察-思考-行动-验证 | 完整自主循环 | P1 |
| 安全边界 | 最大迭代/风险评估 | P1 |
| 用户介入 | 随时中断和调整 | P1 |

### Phase 3: 体验优化（6-12个月）

#### 3.1 CLI/TUI 体验

| 任务 | 描述 | 优先级 |
|------|------|--------|
| 增强 TUI | 更丰富的终端界面 | P2 |
| 快捷键支持 | 键盘快捷操作 | P2 |
| 主题定制 | 主题和配色 | P2 |

#### 3.2 用户协作

| 任务 | 描述 | 优先级 |
|------|------|--------|
| 多种协作模式 | 全自动/建议确认/每步确认 | P2 |
| 实时预览 | 代码变更预览 | P2 |
| 细粒度控制 | 权限和范围控制 | P2 |

---

## 九、技术方案建议

### 9.1 Skills 机制实现

```python
# skill.py
from dataclasses import dataclass
from typing import List, Dict, Any, Callable
import json
import os

@dataclass
class Skill:
    """技能定义"""
    name: str
    description: str
    triggers: List[str]  # 触发关键词
    tool_usage_guide: str
    best_practices: List[str]
    error_handling: List[str]
    
    @classmethod
    def from_file(cls, path: str) -> "Skill":
        """从文件加载技能"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(**data)
    
    @classmethod
    def from_decorator(cls, name: str, description: str, triggers: List[str]):
        """装饰器方式注册"""
        def decorator(func: Callable):
            func._skill = cls(
                name=name,
                description=description,
                triggers=triggers,
                tool_usage_guide=func.__doc__ or "",
                best_practices=[],
                error_handling=[]
            )
            return func
        return decorator


class SkillRegistry:
    """技能注册表"""
    
    def __init__(self):
        self._skills: Dict[str, Skill] = {}
    
    def register(self, skill: Skill) -> None:
        """注册技能"""
        self._skills[skill.name] = skill
    
    def find_skill(self, query: str) -> List[Skill]:
        """根据查询找到相关技能"""
        results = []
        for skill in self._skills.values():
            if any(trigger in query for trigger in skill.triggers):
                results.append(skill)
        return results
    
    def load_from_directory(self, dir_path: str) -> None:
        """从目录加载所有技能"""
        for filename in os.listdir(dir_path):
            if filename.endswith('.json'):
                path = os.path.join(dir_path, filename)
                skill = Skill.from_file(path)
                self.register(skill)


class SkillExecutor:
    """技能执行器"""
    
    def __init__(
        self,
        skill_registry: SkillRegistry,
        llm_client: LLMClient,
        tool_orchestrator: ToolOrchestrator
    ):
        self.registry = skill_registry
        self.llm = llm_client
        self.tools = tool_orchestrator
    
    async def execute_skill(
        self,
        skill_name: str,
        context: Dict[str, Any]
    ) -> ExecutionResult:
        """执行指定技能"""
        skill = self.registry._skills.get(skill_name)
        if not skill:
            raise ValueError(f"Skill not found: {skill_name}")
        
        # 使用工具使用指南构建执行计划
        plan = await self._build_execution_plan(skill, context)
        
        # 执行并处理错误
        result = await self._execute_plan(plan, skill.error_handling)
        
        return result
```

### 9.2 通用任务规划器实现

```python
# universal_planner.py
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum

class TaskType(Enum):
    """任务类型"""
    TEST_GENERATION = "test_generation"
    CODE_REFACTORING = "code_refactoring"
    BUG_FIX = "bug_fix"
    FEATURE_ADD = "feature_add"
    CODE_REVIEW = "code_review"
    QUERY = "query"

@dataclass
class Subtask:
    """子任务"""
    id: str
    description: str
    task_type: TaskType
    dependencies: List[str]
    tools_needed: List[str]
    estimated_complexity: int  # 1-5

@dataclass
class TaskPlan:
    """任务计划"""
    task_id: str
    original_request: str
    task_type: TaskType
    subtasks: List[Subtask]
    execution_order: List[str]  # subtask IDs


class UniversalPlanner:
    """通用任务规划器"""
    
    def __init__(
        self,
        llm_client: LLMClient,
        project_analyzer: ProjectAnalyzer,
        tool_registry: ToolRegistry
    ):
        self.llm = llm_client
        self.analyzer = project_analyzer
        self.tools = tool_registry
    
    async def understand_task(
        self,
        user_request: str,
        project_context: ProjectContext
    ) -> TaskType:
        """理解任务类型"""
        prompt = f"""
        分析以下用户请求，确定任务类型：
        - test_generation: 生成单元测试
        - code_refactoring: 代码重构
        - bug_fix: Bug 修复
        - feature_add: 添加新功能
        - code_review: 代码审查
        - query: 问题查询
        
        用户请求：{user_request}
        
        返回任务类型（仅返回类型名称）：
        """
        
        response = await self.llm.generate(prompt)
        return TaskType(response.strip())
    
    async def decompose_task(
        self,
        task_type: TaskType,
        user_request: str,
        project_context: ProjectContext
    ) -> TaskPlan:
        """分解任务为子任务"""
        # 根据任务类型选择分解策略
        if task_type == TaskType.TEST_GENERATION:
            return await self._decompose_test_generation(user_request, project_context)
        elif task_type == TaskType.CODE_REFACTORING:
            return await self._decompose_refactoring(user_request, project_context)
        # ... 其他任务类型
    
    async def execute_with_feedback(
        self,
        plan: TaskPlan,
        progress_callback: Optional[Callable] = None
    ) -> ExecutionResult:
        """执行计划，支持动态调整"""
        results = []
        failed_subtasks = []
        
        for subtask_id in plan.execution_order:
            subtask = self._find_subtask(plan, subtask_id)
            
            # 执行子任务
            result = await self._execute_subtask(subtask)
            results.append(result)
            
            # 回调进度
            if progress_callback:
                await progress_callback(subtask, result)
            
            # 检查是否需要调整计划
            if not result.success:
                # 分析失败原因，调整后续计划
                await self._adjust_plan(plan, subtask, result)
            
            # 检查是否可继续
            if subtask_id in self._get_blocked_subtasks(plan, results):
                break
        
        return ExecutionResult(
            success=all(r.success for r in results),
            results=results
        )
```

---

## 十、总结

### 10.1 核心差距

| 优先级 | 差距 | 影响 |
|--------|------|------|
| P0 | Skills 机制 | 无法自定义技能，工具使用不够智能 |
| P0 | 通用任务规划 | 只能处理 UT 生成，无法处理其他任务 |
| P0 | 语音交互 | 缺少行业趋势的交互方式 |
| P1 | 远程 MCP | 工具生态受限 |
| P1 | IDE 集成 | 用户习惯差异 |
| P1 | 自主循环 | 灵活性不足 |

### 10.2 已有优势

- 完善的架构设计（事件驱动、组件化）
- 强大的 UT 生成能力
- 290+ 测试的质量保障
- 多 LLM 支持
- 错误预测和自适应策略

### 10.3 改进方向

1. **保持优势**：继续强化 UT 生成领域专业能力
2. **补齐差距**：按优先级逐步改进核心能力
3. **差异化**：在测试领域保持领先，同时扩展通用能力

---

**报告完成日期**：2026-03-04

---

## 十一、已实现改进（2026-03-04）

### 11.1 Phase 1: 真正自主决策能力

#### ✅ LLM驱动的增强自主循环
- **文件**: `pyutagent/agent/llm_driven_autonomous_loop.py`
- **实现**:
  - `LLMActionDecider`: LLM驱动的动作决策器，使用prompt让LLM自主决定下一步
  - `DynamicToolSelector`: 动态工具选择器，根据任务类型选择合适工具
  - `LLMDrivenAutonomousLoop`: 真正的自主循环，摆脱预设流程
- **创新点**:
  - 决策策略支持（目标导向/探索/保守）
  - 置信度阈值判断任务完成
  - 动态工具组合选择

#### ✅ 测试覆盖
- **文件**: `tests/unit/agent/test_llm_driven_autonomous_loop.py`
- **覆盖**: DecisionContext, LLMDecision, LLMActionDecider, DynamicToolSelector, LLMDrivenAutonomousLoop

### 11.2 Phase 2: 长期记忆增强

#### ✅ 情景记忆 (Episodic Memory)
- **文件**: `pyutagent/memory/episodic_memory.py`
- **实现**:
  - `Episode`: 任务执行记录数据结构
  - `ProjectSummary`: 项目执行摘要
  - `EpisodicMemory`: 情景记忆存储和检索
- **功能**:
  - 跨项目经验记录和搜索
  - 项目执行摘要统计
  - 经验教训提取

#### ✅ 程序记忆 (Procedural Memory)
- **文件**: `pyutagent/memory/procedural_memory.py`
- **实现**:
  - `Skill`: 技能/策略数据结构
  - `ProceduralMemory`: 程序记忆存储和学习
- **功能**:
  - 成功策略学习和复用
  - 技能成功率跟踪
  - 最优技能推荐

### 11.3 改进效果

| 改进项 | 改进前 | 改进后 |
|--------|--------|--------|
| 自主决策 | 预设流程执行 | LLM驱动，真正自主选择 |
| 工具选择 | 固定工具集 | 动态工具组合 |
| 经验积累 | 单项目记忆 | 跨项目情景记忆 |
| 策略学习 | 无 | 成功策略学习和复用 |

### 11.4 Phase 3: MCP深度集成

#### ✅ MCP Server自动发现
- **文件**: `pyutagent/tools/mcp_dynamic_manager.py`
- **实现**:
  - `MCPServerDiscovery`: MCP服务器自动发现
  - 支持npm全局包发现
  - 支持配置文件发现 (~/.config/mcp/*.json)
  - 支持项目配置发现
- **功能**:
  - 自动识别常见MCP服务器
  - 多源发现（npm_global/config_file/project）

#### ✅ 动态工具注册
- **文件**: `pyutagent/tools/mcp_dynamic_manager.py`
- **实现**:
  - `DynamicToolRegistry`: 动态工具注册表
  - `MCPDynamicManager`: MCP动态管理器
  - SQLite持久化存储
- **功能**:
  - 工具使用统计
  - 热插拔支持
  - 按服务器分组管理

### 11.5 Phase 4: 灵活用户协作

#### ✅ 灵活协作模式
- **文件**: `pyutagent/agent/collaboration.py`
- **实现**:
  - `CollaborationMode`: 4种协作模式枚举
  - `CollaborationHandler`: 协作处理器
  - `CollaborationConfig`: 协作配置
- **协作模式**:
  - `FULL_AUTONOMOUS`: 完全自主执行
  - `SUGGEST_AND_CONFIRM`: 建议后确认（默认）
  - `STEP_BY_STEP`: 每步确认
  - `MANUAL_REVIEW`: 人工审查
- **功能**:
  - 置信度阈值自动审批
  - 风险级别评估
  - 执行历史追踪

### 11.6 额外已确认的能力

#### ✅ Skills机制（已存在）
- **文件**: `pyutagent/agent/skills.py`
- **实现**:
  - `Skill`: 技能基类
  - `SkillRegistry`: 技能注册表
  - `SkillLoader`: 技能加载器
  - `SkillCategory`: 技能分类（代码生成/代码审查/调试/重构/测试等）
- **特性**:
  - 技能元数据管理
  - 技能步骤定义
  - 示例和最佳实践

#### ✅ 语音交互（已存在）
- **文件**: `pyutagent/agent/voice.py`
- **实现**:
  - `VoiceInputHandler`: 语音识别输入
  - `VoiceOutputHandler`: TTS语音输出
  - `VoiceCommandParser`: 语音命令解析
  - 支持多提供商（Whisper/Google/Azure/OpenAI）
- **功能**:
  - 中文语音命令支持
  - 多TTS引擎（GTTS/Edge TTS/Azure/OpenAI）

#### ✅ 远程MCP支持
- **文件**: `pyutagent/tools/remote_mcp.py`
- **实现**:
  - `RemoteMCPClient`: 远程MCP客户端
  - `RemoteMCPManager`: 远程MCP管理器
  - 支持WebSocket/HTTP连接
  - 认证支持（API Key/Bearer Token）
- **功能**:
  - 安全远程连接
  - 连接池
  - 自动重连

#### ✅ ACP协议/IDE集成（已存在）
- **文件**: `pyutagent/agent/acp_client.py`
- **实现**:
  - `ACPClient`: ACP协议客户端
  - 支持IDEA集成
- **VSCode扩展**: `pyutagent-vscode/` 目录

---

**参考资料**：
- [1] 编程进入「对讲机」时代!Claude抢发语音写代码
- [2] Remote MCP support in Claude Code
- [3] IDEA 里终于能爽用 Claude Code了
- [4] 如何编写你的第一个 MCP Skill
- [5] OpenCode:终端中的AI编程助手
- [6] AI编程工具深度对比:Cursor、Copilot、Trae与Claude Code
