# 功能整合验收标准 - 已实现但未充分集成的组件

## 阶段一：核心重复清理（优先级：高）

### 1.1 MessageBus 整合验收

- [ ] **C1.1.1** 分析完成：识别出 core/message_bus.py 和 multi_agent/message_bus.py 的功能差异
- [ ] **C1.1.2** 整合完成：multi_agent 版本继承或组合 core 版本
- [ ] **C1.1.3** 测试通过：所有 multi_agent 相关测试通过
- [ ] **C1.1.4** 无重复代码：搜索确认无 MessageBus 重复定义

### 1.2 AgentState 整合验收

- [ ] **C1.2.1** 定义完整：protocols.py 包含所有需要的 AgentState
- [ ] **C1.2.2** 使用确认：state_machine.py 中所有使用处已改为导入 protocols
- [ ] **C1.2.3** 删除完成：state_machine.py 中无重复 AgentState 定义
- [ ] **C1.2.4** 测试通过：运行测试无 AgentState 相关错误

### 1.3 Cache 整合验收

- [ ] **C1.3.1** 分析完成：识别出各缓存的实际用途
- [ ] **C1.3.2** 策略确定：确定保留 core/cache.py 和 llm/multi_level_cache.py
- [ ] **C1.3.3** 引用更新：所有对 tools/tool_cache.py 的引用已更新
- [ ] **C1.3.4** 文件删除：tools/tool_cache.py 已删除
- [ ] **C1.3.5** 引用更新：所有对 core/tool_cache.py 的引用已更新
- [ ] **C1.3.6** 文件删除：core/tool_cache.py 已删除
- [ ] **C1.3.7** 测试通过：缓存相关测试全部通过
- [ ] **C1.3.8** 搜索确认：无 ToolResultCache 在 tools/ 或 core/ 中的重复定义

---

## 阶段二：关键组件集成（优先级：中）

### 2.1 ThinkingEngine & ThinkingOrchestrator 集成验收

- [ ] **C2.1.1** 分析完成：明确 ThinkingEngine 的功能和应用场景
- [ ] **C2.1.2** 集成完成：ReActAgent 中添加可选 thinking 模式
- [ ] **C2.1.3** 调用集成：execution_steps.py 中可调用 thinking
- [ ] **C2.1.4** 配置支持：可通过配置开关启用/关闭 thinking
- [ ] **C2.1.5** 测试通过：thinking 相关测试通过

### 2.2 ToolOrchestrator 集成验收

- [ ] **C2.2.1** 分析完成：明确 ToolOrchestrator 可用功能
- [ ] **C2.2.2** 集成完成编译：/测试失败处理中调用 ToolOrchestrator
- [ ] **C2.2.3** 触发验证：失败时正确触发恢复逻辑
- [ ] **C2.2.4** 测试通过：ToolOrchestrator 相关测试通过

### 2.3 Agent 变体处理验收

- [ ] **C2.3.1** 分析完成：ToolEnabledReActAgent 与 ReActAgent 差异明确
- [ ] **C2.3.2** 分析完成：ToolUseAgent 与 ReActAgent 差异明确
- [ ] **C2.3.3** 决策明确：确定合并或删除方案
- [ ] **C2.3.4** 执行完成：合并或删除操作完成
- [ ] **C2.3.5** 测试通过：Agent 相关测试全部通过
- [ ] **C2.3.6** 无孤立代码：搜索确认无未使用的 Agent 类

---

## 阶段三：优化与精简（优先级：低）

### 3.1 Memory 模块优化验收

- [ ] **C3.1.1** 评估完成：ShortTermMemory 使用情况明确
- [ ] **C3.1.2** 评估完成：PatternLibrary 等使用情况明确
- [ ] **C3.1.3** 优化完成：Memory 模块已精简
- [ ] **C3.1.4** 测试通过：Memory 相关测试通过

### 3.2 Editor 整合验收

- [ ] **C3.2.1** 分析完成：4 个 Editor 功能边界明确
- [ ] **C3.2.2** 策略确定：合并到 1-2 个核心 Editor
- [ ] **C3.2.3** 合并完成：Editor 功能完整合并
- [ ] **C3.2.4** 测试通过：Editor 相关测试通过
- [ ] **C3.2.5** 搜索确认：无重复的 Editor 类定义

### 3.3 AutonomousLoop 处理验收

- [ ] **C3.3.1** 评估完成：确认是否为预留功能
- [ ] **C3.3.2** 决策明确：保留、集成或删除方案确定
- [ ] **C3.3.3** 执行完成：AutonomousLoop 处理完成
- [ ] **C3.3.4** 测试通过：无相关错误

### 3.4 最终验收

- [ ] **C3.4.1** 测试通过：完整测试套件通过
- [ ] **C3.4.2** 无循环依赖：运行依赖检查无问题
- [ ] **C3.4.3** 文档更新：ARCHITECTURE.md 反映实际实现
- [ ] **C3.4.4** 搜索验证：无未使用但存在的组件（除明确标记为可选的）
- [ ] **C3.4.5** 代码统计：代码重复率降至 5% 以下

---

## 验收检查方法

### 1. 测试验证

```bash
# 运行完整测试套件
python -m pytest tests/ -v --tb=short

# 运行特定模块测试
python -m pytest tests/unit/core/test_cache.py -v
python -m pytest tests/unit/agent/ -v
```

### 2. 代码搜索验证

```bash
# 搜索重复定义
grep -r "class MessageBus" pyutagent/
grep -r "class AgentState" pyutagent/
grep -r "class ToolResultCache" pyutagent/

# 搜索未使用的类
grep -r "class ToolEnabledReActAgent" pyutagent/agent/
grep -r "class ThinkingEngine" pyutagent/agent/
```

### 3. 循环依赖检查

```bash
# 使用 pylint 或 custom script 检查循环依赖
python -c "from pyutagent import agent; print('OK')"
```

---

## 验收签字

| 阶段 | 验收人 | 验收日期 | 备注 |
|------|--------|----------|------|
| 阶段一 | | | |
| 阶段二 | | | |
| 阶段三 | | | |
