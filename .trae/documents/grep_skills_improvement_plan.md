# Grep工具和Skills改进计划

## 一、背景分析

### 1.1 Claude Code的Skills机制核心特点

Claude Code的Skills机制是"工作流说明书"，不是简单的提示词模板：

**核心特性**：
1. **渐进式加载(Progressive Disclosure)**：按需加载，不是全部加载
2. **文件夹结构**：
   ```
   skill-name/
   ├── SKILL.md          # 必需：技能说明
   ├── scripts/          # 可选：辅助脚本
   ├── templates/        # 可选：文档模板
   └── resources/        # 可选：参考资料
   ```
3. **智能触发**：通过description匹配任务，自动加载相关Skill

### 1.2 当前PyUT Agent的实现状态

**Grep工具**：
- ✅ 基本正则搜索
- ✅ 文件过滤(glob)
- ❌ 语义搜索能力
- ❌ 上下文感知
- ❌ AI增强搜索

**Skills**：
- ✅ Skill基类和注册表
- ✅ Skill元数据和步骤
- ❌ 渐进式加载机制
- ❌ 外部脚本支持
- ❌ 模板系统
- ❌ 智能触发

---

## 二、改进目标

### 2.1 Grep工具增强

| 改进项 | 目标 |
|--------|------|
| 语义搜索 | 支持自然语言查询 |
| 上下文感知 | 理解代码语义 |
| AI增强 | LLM辅助理解搜索结果 |
| 过滤增强 | 更智能的文件过滤 |

### 2.2 Skills机制增强

| 改进项 | 目标 |
|--------|------|
| 渐进式加载 | 按需加载，减少上下文消耗 |
| 外部脚本 | 支持scripts/目录 |
| 模板系统 | 支持templates/目录 |
| 智能触发 | 通过description自动匹配Skill |

---

## 三、实施计划

### Phase 1: Grep工具增强

#### 1.1 创建语义搜索工具

**文件**: `pyutagent/tools/semantic_grep.py`

```python
class SemanticGrepTool(Tool):
    """AI增强的语义搜索工具"""

    async def execute(self, **kwargs) -> ToolResult:
        # 1. 将自然语言查询转换为技术搜索
        # 2. 执行搜索
        # 3. 使用LLM理解结果
        pass
```

**实施步骤**:
1. 创建SemanticGrepTool类
2. 实现自然语言→技术查询转换
3. 集成LLM结果理解
4. 添加测试

#### 1.2 增强现有GrepTool

**改进项**:
1. 添加上下文行数参数(context_lines)
2. 添加文件类型过滤增强
3. 添加结果分组功能
4. 添加输出格式选项

---

### Phase 2: Skills机制增强

#### 2.1 实现渐进式加载

**文件**: `pyutagent/agent/enhanced_skills.py`

```python
class EnhancedSkillLoader(SkillLoader):
    """增强版Skill加载器，支持渐进式加载"""

    def load_skill_summary(self, skill_name: str) -> SkillSummary:
        """只加载name和description"""
        pass

    def load_skill_full(self, skill_name: str) -> Skill:
        """完整加载Skill"""
        pass

    def auto_discover_skills(self) -> List[SkillSummary]:
        """自动发现Skills目录下的Skills"""
        pass
```

#### 2.2 实现Claude Code格式兼容

```python
class ClaudeCodeSkillAdapter:
    """Claude Code格式Skill适配器"""

    def load_from_folder(self, folder_path: Path) -> Skill:
        """从文件夹加载Skill"""
        # 读取 SKILL.md
        # 加载 scripts/ 下的脚本
        # 加载 templates/ 下的模板
        pass

    def export_to_claude_format(self, skill: Skill) -> Path:
        """导出为Claude Code格式"""
        pass
```

#### 2.3 实现智能触发

```python
class SkillMatcher:
    """Skill智能匹配器"""

    def match(self, user_request: str) -> List[Skill]:
        """根据用户请求匹配Skills"""
        # 1. 提取请求关键词
        # 2. 匹配Skill description
        # 3. 返回相关Skills
        pass
```

---

### Phase 3: 集成与测试

#### 3.1 集成到Agent

**步骤**:
1. 在Agent中集成SemanticGrepTool
2. 在Agent中集成EnhancedSkillLoader
3. 添加Skill自动发现功能

#### 3.2 测试覆盖

**测试文件**:
- `tests/unit/tools/test_semantic_grep.py`
- `tests/unit/agent/test_enhanced_skills.py`

---

## 四、详细实施步骤

### Step 1: Grep工具增强 (1小时)

- [ ] 1.1 创建semantic_grep.py
- [ ] 1.2 实现SemanticGrepTool类
- [ ] 1.3 添加单元测试

### Step 2: Skills渐进式加载 (2小时)

- [ ] 2.1 创建enhanced_skills.py
- [ ] 2.2 实现EnhancedSkillLoader
- [ ] 2.3 实现SkillSummary类

### Step 3: Claude Code格式兼容 (2小时)

- [ ] 3.1 实现ClaudeCodeSkillAdapter
- [ ] 3.2 支持SKILL.md解析
- [ ] 3.3 支持scripts/目录

### Step 4: 智能触发 (1小时)

- [ ] 4.1 实现SkillMatcher
- [ ] 4.2 实现自动发现
- [ ] 4.3 集成测试

### Step 5: Agent集成 (1小时)

- [ ] 5.1 在Agent中启用新功能
- [ ] 5.2 端到端测试

---

## 五、验收标准

### Grep工具
- [ ] 能执行语义搜索
- [ ] 能理解自然语言查询
- [ ] 原有功能不受影响

### Skills机制
- [ ] 支持渐进式加载
- [ ] 兼容Claude Code格式
- [ ] 能自动匹配Skills
- [ ] 原有Skills正常工作

---

**计划制定日期**: 2026-03-04
