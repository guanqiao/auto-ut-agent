# Coding Agent 能力增强开发计划

## 一、计划背景

基于 Gap 分析报告，项目已有部分基础实现：

### 已有的能力（需要增强）
| 模块 | 文件 | 现状 |
|------|------|------|
| **Skills 机制** | `agent/skills.py`, `builtin_skills.py` | 基础框架已有，需增强工具使用指南、最佳实践、错误处理 |
| **通用任务规划** | `agent/universal_agent.py`, `task_understanding.py`, `task_planner.py` | 任务分类和基础规划已有，需完善执行和动态调整 |
| **LLM 驱动自主循环** | `agent/llm_driven_autonomous_loop.py` | 基础已有，需增强安全边界、用户介入 |

### 需要新增的能力
| 模块 | 优先级 | 说明 |
|------|--------|------|
| **语音交互** | P1 | Whisper 语音识别 + TTS 语音反馈 |
| **IDE 集成** | P1 | VS Code / IDEA 插件 |

---

## 二、开发阶段

### Phase 1: Skills 机制增强（P0，2周）

#### 1.1 增强 Skill 数据模型

```python
# 目标：增加 tool_usage_guide, best_practices, error_handling
@dataclass
class SkillStep:
    """技能执行步骤"""
    step_id: str
    description: str
    tool_name: str
    parameters: Dict[str, Any]
    validation: Optional[str] = None
    rollback: Optional[str] = None

@dataclass
class SkillExample:
    """技能使用示例"""
    input_params: Dict[str, Any]
    expected_output: Any
    description: str

@dataclass
class EnhancedSkillMetadata:
    """增强的技能元数据"""
    name: str
    description: str
    category: SkillCategory
    version: str = "1.0.0"
    author: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    # 新增字段
    triggers: List[str] = field(default_factory=list)  # 触发关键词
    tool_usage_guide: str = ""  # 如何正确使用工具
    best_practices: List[str] = field(default_factory=list)  # 最佳实践
    error_handling: List[str] = field(default_factory=list)  # 错误处理
    examples: List[SkillExample] = field(default_factory=list)  # 示例
    requires_tools: List[str] = field(default_factory=list)  # 需要的工具
    estimated_duration: Optional[str] = None  # 预估时长
```

#### 1.2 Skill 执行引擎增强

```python
class EnhancedSkillExecutor:
    """增强的技能执行器"""
    
    async def execute_with_retry(
        self,
        skill: Skill,
        context: Dict[str, Any],
        max_retries: int = 3
    ) -> SkillOutput:
        """带重试的执行"""
        pass
    
    async def validate_execution(
        self,
        skill: Skill,
        context: Dict[str, Any]
    ) -> bool:
        """验证执行条件"""
        pass
    
    async def rollback_on_failure(
        self,
        skill: Skill,
        context: Dict[str, Any],
        error: Exception
    ) -> bool:
        """失败时回滚"""
        pass
```

#### 1.3 内置 Skills 扩展

| Skill | 描述 | 触发词 |
|-------|------|--------|
| `generate_unit_test` | 单元测试生成 | "测试", "生成测试" |
| `fix_compilation_error` | 编译错误修复 | "编译错误", "修复" |
| `analyze_code` | 代码分析 | "分析", "审查" |
| `refactor_code` | 代码重构 | "重构", "优化" |
| `explain_code` | 代码解释 | "解释", "是什么" |
| `debug_test` | 测试调试 | "调试", "测试失败" |

#### 1.4 任务清单

- [ ] 1.1.1 增强 SkillMetadata 数据类，添加 triggers、tool_usage_guide 等字段
- [ ] 1.1.2 创建 SkillStep、SkillExample 数据类
- [ ] 1.1.3 更新 SkillRegistry 支持新字段
- [ ] 1.2.1 实现 EnhancedSkillExecutor 执行引擎
- [ ] 1.2.2 添加执行验证和回滚机制
- [ ] 1.3.1 扩展 builtin_skills.py 添加更多内置技能
- [ ] 1.3.2 添加技能使用示例
- [ ] 1.3.3 创建技能配置文件格式（YAML/JSON）
- [ ] 1.4.1 编写单元测试
- [ ] 1.4.2 集成测试验证

---

### Phase 2: 通用任务规划增强（P0，3周）

#### 2.1 任务理解层增强

```python
class EnhancedTaskClassifier:
    """增强的任务分类器"""
    
    async def classify_with_context(
        self,
        request: str,
        project_context: ProjectContext,
        history: List[TaskHistory]
    ) -> TaskUnderstanding:
        """带上下文的任务分类"""
        pass
    
    async def extract_entities(
        self,
        request: str
    ) -> Dict[str, Any]:
        """提取实体（文件、类、方法、参数）"""
        pass
    
    async def detect_intent(
        self,
        request: str
    ) -> Intent:
        """检测意图"""
        pass
```

#### 2.2 任务分解器增强

```python
class EnhancedTaskPlanner:
    """增强的任务规划器"""
    
    async def decompose_with_dependencies(
        self,
        understanding: TaskUnderstanding,
        project_context: ProjectContext
    ) -> ExecutionPlan:
        """带依赖的任务分解"""
        pass
    
    async def estimate_complexity(
        self,
        plan: ExecutionPlan
    ) -> TaskComplexity:
        """估算复杂度"""
        pass
    
    async def refine_plan_dynamic(
        self,
        plan: ExecutionPlan,
        feedback: ExecutionFeedback
    ) -> ExecutionPlan:
        """动态调整计划"""
        pass
```

#### 2.3 计划执行器增强

```python
class EnhancedPlanExecutor:
    """增强的计划执行器"""
    
    async def execute_with_checkpoints(
        self,
        plan: ExecutionPlan,
        checkpoint_interval: int = 5
    ) -> ExecutionResult:
        """带检查点的执行"""
        pass
    
    async def parallel_execute_independent(
        self,
        subtasks: List[SubTask],
        max_parallel: int = 3
    ) -> List[SubTaskResult]:
        """并行执行独立任务"""
        pass
    
    async def handle_failure(
        self,
        subtask: SubTask,
        error: Exception
    ) -> RecoveryAction:
        """处理失败"""
        pass
```

#### 2.4 任务清单

- [ ] 2.1.1 增强 TaskClassifier，添加实体提取和意图检测
- [ ] 2.1.2 添加项目上下文理解
- [ ] 2.1.3 支持历史任务学习
- [ ] 2.2.1 增强 TaskPlanner，支持依赖分析
- [ ] 2.2.2 添加复杂度估算
- [ ] 2.2.3 实现动态计划调整
- [ ] 2.3.1 增强 PlanExecutor
- [ ] 2.3.2 实现检查点机制
- [ ] 2.3.3 支持并行执行
- [ ] 2.4.1 添加更多任务类型支持
- [ ] 2.4.2 集成测试

---

### Phase 3: 自主循环增强（P1，2周）

#### 3.1 LLM 决策增强

```python
class EnhancedLLMActionDecider:
    """增强的 LLM 决策器"""
    
    async def decide_with_safety(
        self,
        context: DecisionContext,
        safety_policy: SafetyPolicy
    ) -> LLMDecision:
        """带安全策略的决策"""
        pass
    
    async def explain_decision(
        self,
        decision: LLMDecision
    ) -> str:
        """解释决策"""
        pass
    
    async def learn_from_feedback(
        self,
        decision: LLMDecision,
        result: ExecutionResult
    ) -> None:
        """从反馈学习"""
        pass
```

#### 3.2 安全边界增强

```python
@dataclass
class SafetyPolicy:
    """安全策略"""
    max_iterations: int = 50
    max_file_edits: int = 100
    max_command_executions: int = 50
    allow_destructive: bool = False
    allowed_commands: List[str] = field(default_factory=list)
    blocked_paths: List[str] = field(default_factory=list)
    
class SafetyValidator:
    """安全验证器"""
    
    def validate_action(
        self,
        action: LLMDecision,
        policy: SafetyPolicy
    ) -> ValidationResult:
        """验证动作安全性"""
        pass
    
    def validate_file_edit(
        self,
        file_path: str,
        old_content: str,
        new_content: str,
        policy: SafetyPolicy
    ) -> ValidationResult:
        """验证文件编辑"""
        pass
```

#### 3.3 用户介入机制

```python
class UserInterventionHandler:
    """用户介入处理器"""
    
    async def request_confirmation(
        self,
        action: ProposedAction,
        context: Dict[str, Any]
    ) -> UserResponse:
        """请求确认"""
        pass
    
    async def handle_interrupt(
        self,
        interrupt: UserInterrupt
    ) -> InterruptAction:
        """处理中断"""
        pass
    
    async def suggest_alternatives(
        self,
        action: LLMDecision
    ) -> List[AlternativeAction]:
        """建议替代方案"""
        pass
```

#### 3.4 任务清单

- [ ] 3.1.1 增强 LLMActionDecider
- [ ] 3.1.2 添加决策解释功能
- [ ] 3.1.3 实现反馈学习
- [ ] 3.2.1 实现 SafetyPolicy 和 SafetyValidator
- [ ] 3.2.2 添加文件编辑验证
- [ ] 3.2.3 添加命令执行验证
- [ ] 3.3.1 实现用户确认机制
- [ ] 3.3.2 实现中断处理
- [ ] 3.3.3 添加替代建议
- [ ] 3.4.1 集成测试

---

### Phase 4: 语音交互（P1，3周）

#### 4.1 语音识别集成

```python
class VoiceInputHandler:
    """语音输入处理器"""
    
    def __init__(self, config: VoiceConfig):
        self.config = config
        self.recognizer = None
        self.is_listening = False
    
    async def start_listening(self) -> None:
        """开始监听"""
        pass
    
    async def stop_listening(self) -> str:
        """停止监听并返回识别结果"""
        pass
    
    async def transcribe_audio(
        self,
        audio_data: bytes
    ) -> str:
        """转录音频"""
        pass
```

#### 4.2 语音反馈集成

```python
class VoiceOutputHandler:
    """语音输出处理器"""
    
    def __init__(self, config: VoiceConfig):
        self.config = config
        self.tts_engine = None
    
    async def speak(
        self,
        text: str,
        interrupt_current: bool = True
    ) -> None:
        """语音合成输出"""
        pass
    
    async def speak_with_voice(
        self,
        text: str,
        voice: str,
        interrupt_current: bool = True
    ) -> None:
        """指定音色输出"""
        pass
```

#### 4.3 语音命令协议

```python
class VoiceCommandParser:
    """语音命令解析器"""
    
    # 命令模式
    COMMAND_PATTERNS = [
        (r"生成.*测试", "generate_test"),
        (r"修复.*错误", "fix_error"),
        (r"分析.*代码", "analyze_code"),
        (r"暂停", "pause"),
        (r"继续", "resume"),
        (r"停止", "stop"),
    ]
    
    async def parse_command(
        self,
        text: str
    ) -> VoiceCommand:
        """解析语音命令"""
        pass
```

#### 4.4 任务清单

- [ ] 4.1.1 实现 VoiceInputHandler（支持 Whisper/云服务）
- [ ] 4.1.2 添加音频预处理
- [ ] 4.1.3 实现持续监听模式
- [ ] 4.2.1 实现 VoiceOutputHandler（TTS）
- [ ] 4.2.2 添加语音配置
- [ ] 4.2.3 实现语音打断
- [ ] 4.3.1 实现 VoiceCommandParser
- [ ] 4.3.2 添加命令模式匹配
- [ ] 4.3.3 实现语音反馈过滤
- [ ] 4.4.1 集成 GUI
- [ ] 4.4.2 测试验证

---

### Phase 5: IDE 集成（P1，4周）

#### 5.1 VS Code 插件

```
vscode-extension/
├── package.json
├── src/
│   ├── extension.ts      # 入口
│   ├── commands.ts       # 命令定义
│   ├── panel.ts          # Webview 面板
│   └── api.ts            # 与主程序通信
└── README.md
```

#### 5.2 IDEA 集成（ACP 协议）

```python
class ACPClient:
    """Anthropic Claude Protocol 客户端"""
    
    async def connect(
        self,
        endpoint: str,
        api_key: str
    ) -> bool:
        """连接到 ACP 服务器"""
        pass
    
    async def send_message(
        self,
        message: Dict[str, Any]
    ) -> Dict[str, Any]:
        """发送消息"""
        pass
    
    async def receive_notification(
        self
    ) -> Notification:
        """接收通知"""
        pass
```

#### 5.3 任务清单

- [ ] 5.1.1 创建 VS Code 插件项目结构
- [ ] 5.1.2 实现插件入口和命令
- [ ] 5.1.3 实现 Webview 面板
- [ ] 5.1.4 添加与主程序通信
- [ ] 5.2.1 实现 ACP 客户端
- [ ] 5.2.2 添加 IDE 事件监听
- [ ] 5.2.3 实现代码操作
- [ ] 5.3.1 测试验证

---

## 三、技术架构

### 3.1 模块依赖关系

```
┌─────────────────────────────────────────────────────┐
│                    UI Layer                         │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌────────┐ │
│  │ GUI     │  │ CLI     │  │ TUI     │  │ Voice  │ │
│  └────┬────┘  └────┬────┘  └────┬────┘  └───┬────┘ │
└───────┼────────────┼────────────┼──────────────┘
        │            │            │
┌───────▼────────────▼────────────▼──────────────┐
│              Agent Layer                            │
│  ┌──────────────────────────────────────────┐     │
│  │         UniversalCodingAgent             │     │
│  │  ┌─────────────┐  ┌─────────────────┐   │     │
│  │  │ TaskClassifier│  │ TaskPlanner     │   │     │
│  │  └─────────────┘  └─────────────────┘   │     │
│  │  ┌─────────────┐  ┌─────────────────┐   │     │
│  │  │ PlanExecutor │  │ LLMActionDecider│   │     │
│  │  └─────────────┘  └─────────────────┘   │     │
│  └──────────────────────────────────────────┘     │
└───────────────────────────────────────────────────┘
        │
┌───────▼───────────────────────────────────────────┐
│              Skill Layer                           │
│  ┌──────────────┐  ┌────────────────────────┐    │
│  │SkillRegistry │  │ SkillExecutor          │    │
│  │ - register   │  │ - execute              │    │
│  │ - search      │  │ - validate             │    │
│  │ - load        │  │ - rollback             │    │
│  └──────────────┘  └────────────────────────┘    │
└───────────────────────────────────────────────────┘
        │
┌───────▼───────────────────────────────────────────┐
│              Tool Layer                            │
│  ┌──────────────┐  ┌────────────────────────┐    │
│  │ToolRegistry  │  │ SafetyValidator        │    │
│  └──────────────┘  └────────────────────────┘    │
└───────────────────────────────────────────────────┘
```

### 3.2 数据流

```
User Request
    │
    ▼
┌─────────────────┐
│ TaskClassifier │ ──► TaskUnderstanding
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ TaskPlanner    │ ──► ExecutionPlan
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  SkillMatcher   │ ──► Skill (if matched)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ PlanExecutor   │ ──► SubTask Results
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌───────┐ ┌───────┐
│ Success│ │ Failure│
└───┬───┘ └───┬───┘
    │         │
    ▼         ▼
┌─────────┐ ┌──────────┐
│Feedback │ │Refine   │
│Learning │ │Plan     │
└─────────┘ └──────────┘
```

---

## 四、测试计划

### 4.1 单元测试

- Skill 框架：20+ 测试
- 任务理解：15+ 测试
- 计划执行：20+ 测试
- 语音交互：15+ 测试
- 安全验证：10+ 测试

### 4.2 集成测试

- 端到端任务执行：10+ 测试
- IDE 集成：5+ 测试
- 语音交互：5+ 测试

### 4.3 性能测试

- 任务执行时间
- 并发处理能力
- 内存使用

---

## 五、里程碑

| 阶段 | 里程碑 | 时间 |
|------|--------|------|
| Phase 1 | Skills 机制增强完成 | Week 1-2 |
| Phase 2 | 通用任务规划增强完成 | Week 3-5 |
| Phase 3 | 自主循环增强完成 | Week 6-7 |
| Phase 4 | 语音交互完成 | Week 8-10 |
| Phase 5 | IDE 集成完成 | Week 11-14 |

---

## 六、风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 语音识别延迟 | 体验差 | 异步处理，本地优先 |
| LLM 决策不稳定 | 任务失败 | 安全边界，重试机制 |
| IDE 兼容性 | 覆盖不全 | 先主流后扩展 |
| 性能瓶颈 | 响应慢 | 缓存，并行执行 |

---

## 七、优先级总结

### P0 - 必须完成
1. **Skills 机制增强** - 2周
2. **通用任务规划增强** - 3周

### P1 - 重要完成
3. **自主循环增强** - 2周
4. **语音交互** - 3周
5. **IDE 集成** - 4周

### P2 - 后续优化
- CLI/TUI 体验增强
- 用户协作模式
- 知识推理

---

**计划创建日期**：2026-03-04
**预计完成时间**：14 周
