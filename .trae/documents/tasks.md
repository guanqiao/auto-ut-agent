# PyUT Agent 代码整合 - 详细任务分解

## 任务总览

### 完整迁移任务 (3周)
- Phase A: 核心模块迁移 (Week 1) - 5个任务
- Phase B: Agent模块清理 (Week 2) - 4个任务
- Phase C: 验证和文档 (Week 3) - 3个任务

### 完美整合任务 (3个月)
- Month 1: 架构层面整合 - 4个任务
- Month 2: 代码层面优化 - 4个任务
- Month 3: 测试和文档 - 4个任务

---

## 第一部分：完整迁移任务

### Phase A: 核心模块迁移 (Week 1)

#### Task A1: 迁移 core/__init__.py
**任务ID:** TASK-A1  
**优先级:** P0 (最高)  
**预估工时:** 4小时  
**依赖:** 无

**描述:**
迁移 `pyutagent/core/__init__.py` 中的废弃导入，将 `EventBus` 和 `AsyncEventBus` 替换为 `UnifiedMessageBus`。

**子任务:**
- [ ] A1.1: 分析当前导入使用情况
- [ ] A1.2: 添加新的统一接口导出
- [ ] A1.3: 标记旧接口为废弃
- [ ] A1.4: 更新内部使用代码
- [ ] A1.5: 运行单元测试
- [ ] A1.6: 运行迁移检查脚本

**验收标准:**
- `scripts/check_migration.py` 不报告 `core/__init__.py` 有废弃导入
- 所有使用 `core.EventBus` 的代码正常工作
- 单元测试通过

**提交信息:**
```
Migrate core/__init__.py: EventBus -> UnifiedMessageBus

- Replace EventBus/AsyncEventBus imports with UnifiedMessageBus
- Add deprecation warnings for backward compatibility
- Update all internal usages
- All tests pass
```

---

#### Task A2: 迁移 agent/components/execution_steps.py
**任务ID:** TASK-A2  
**优先级:** P0  
**预估工时:** 3小时  
**依赖:** TASK-A1

**描述:**
迁移 `pyutagent/agent/components/execution_steps.py` 中的 `StepResult` 导入。

**子任务:**
- [ ] A2.1: 分析 `StepResult` 使用情况
- [ ] A2.2: 替换为 `AgentResult`
- [ ] A2.3: 更新类型注解
- [ ] A2.4: 验证功能等价
- [ ] A2.5: 运行单元测试

**验收标准:**
- 文件无废弃导入
- 类型检查通过
- 单元测试通过

**提交信息:**
```
Migrate execution_steps.py: StepResult -> AgentResult

- Replace StepResult import with AgentResult
- Update type annotations
- Maintain backward compatibility
```

---

#### Task A3: 迁移 agent/components/core_agent.py
**任务ID:** TASK-A3  
**优先级:** P0  
**预估工时:** 3小时  
**依赖:** TASK-A1

**描述:**
迁移 `pyutagent/agent/components/core_agent.py` 中的 `StepResult` 导入。

**子任务:**
- [ ] A3.1: 分析 `StepResult` 使用情况
- [ ] A3.2: 替换为 `AgentResult`
- [ ] A3.3: 更新类型注解
- [ ] A3.4: 验证功能等价
- [ ] A3.5: 运行单元测试

**验收标准:**
- 文件无废弃导入
- 类型检查通过
- 单元测试通过

**提交信息:**
```
Migrate core_agent.py: StepResult -> AgentResult

- Replace StepResult import with AgentResult
- Update type annotations
- Maintain backward compatibility
```

---

#### Task A4: 迁移 agent/components/agent_initialization.py
**任务ID:** TASK-A4  
**优先级:** P0  
**预估工时:** 4小时  
**依赖:** TASK-A1

**描述:**
迁移 `pyutagent/agent/components/agent_initialization.py` 中的 `ContextManager` 导入。

**子任务:**
- [ ] A4.1: 分析 `ContextManager` 使用情况
- [ ] A4.2: 替换为 `UnifiedContextManager`
- [ ] A4.3: 更新类型注解
- [ ] A4.4: 适配API差异
- [ ] A4.5: 验证功能等价
- [ ] A4.6: 运行单元测试

**验收标准:**
- 文件无废弃导入
- 类型检查通过
- 单元测试通过
- 功能等价

**提交信息:**
```
Migrate agent_initialization.py: ContextManager -> UnifiedContextManager

- Replace ContextManager import with UnifiedContextManager
- Update type annotations
- Adapt to API differences
- All tests pass
```

---

#### Task A5: 迁移 agent/capability_registry.py
**任务ID:** TASK-A5  
**优先级:** P0  
**预估工时:** 4小时  
**依赖:** TASK-A1

**描述:**
迁移 `pyutagent/agent/capability_registry.py` 中的 `SubAgent` 类型注解。

**子任务:**
- [ ] A5.1: 分析 `SubAgent` 使用情况
- [ ] A5.2: 替换为 `UnifiedAgentBase`
- [ ] A5.3: 更新类型注解
- [ ] A5.4: 验证类型检查
- [ ] A5.5: 运行单元测试

**验收标准:**
- 文件无废弃导入
- mypy 类型检查通过
- 单元测试通过

**提交信息:**
```
Migrate capability_registry.py: SubAgent -> UnifiedAgentBase

- Replace SubAgent type annotations with UnifiedAgentBase
- Update all references
- Type checking passes
```

---

### Phase B: Agent模块清理 (Week 2)

#### Task B1: 清理 agent/__init__.py 导出 - BaseAgent
**任务ID:** TASK-B1  
**优先级:** P0  
**预估工时:** 3小时  
**依赖:** TASK-A2, TASK-A3

**描述:**
在 `agent/__init__.py` 中为 `BaseAgent` 导出添加废弃警告。

**子任务:**
- [ ] B1.1: 实现 `__getattr__` 废弃警告
- [ ] B1.2: 添加废弃文档字符串
- [ ] B1.3: 测试废弃警告显示
- [ ] B1.4: 验证向后兼容

**验收标准:**
- 导入 `BaseAgent` 时显示废弃警告
- 向后兼容保持
- 测试通过

**提交信息:**
```
Deprecate BaseAgent export in agent/__init__.py

- Add deprecation warning via __getattr__
- Maintain backward compatibility
- Update documentation
```

---

#### Task B2: 清理 agent/__init__.py 导出 - AutonomousLoop
**任务ID:** TASK-B2  
**优先级:** P0  
**预估工时:** 3小时  
**依赖:** TASK-A1

**描述:**
在 `agent/__init__.py` 中为 `AutonomousLoop` 导出添加废弃警告。

**子任务:**
- [ ] B2.1: 实现 `__getattr__` 废弃警告
- [ ] B2.2: 添加废弃文档字符串
- [ ] B2.3: 测试废弃警告显示
- [ ] B2.4: 验证向后兼容

**验收标准:**
- 导入 `AutonomousLoop` 时显示废弃警告
- 向后兼容保持
- 测试通过

**提交信息:**
```
Deprecate AutonomousLoop export in agent/__init__.py

- Add deprecation warning via __getattr__
- Maintain backward compatibility
- Update documentation
```

---

#### Task B3: 清理 agent/__init__.py 导出 - ContextManager
**任务ID:** TASK-B3  
**优先级:** P0  
**预估工时:** 3小时  
**依赖:** TASK-A4

**描述:**
在 `agent/__init__.py` 中为 `ContextManager` 导出添加废弃警告。

**子任务:**
- [ ] B3.1: 实现 `__getattr__` 废弃警告
- [ ] B3.2: 添加废弃文档字符串
- [ ] B3.3: 测试废弃警告显示
- [ ] B3.4: 验证向后兼容

**验收标准:**
- 导入 `ContextManager` 时显示废弃警告
- 向后兼容保持
- 测试通过

**提交信息:**
```
Deprecate ContextManager export in agent/__init__.py

- Add deprecation warning via __getattr__
- Maintain backward compatibility
- Update documentation
```

---

#### Task B4: 更新测试用例
**任务ID:** TASK-B4  
**优先级:** P1  
**预估工时:** 8小时  
**依赖:** TASK-B1, TASK-B2, TASK-B3

**描述:**
更新测试用例，使用新的统一接口。

**子任务:**
- [ ] B4.1: 更新 `test_base_agent.py`
- [ ] B4.2: 更新 `test_autonomous_loop.py`
- [ ] B4.3: 更新 `test_context_manager.py`
- [ ] B4.4: 运行所有测试
- [ ] B4.5: 检查覆盖率

**验收标准:**
- 所有测试使用统一接口
- 测试覆盖率 > 90%
- 所有测试通过

**提交信息:**
```
Update tests to use unified interfaces

- Migrate test_base_agent.py to test UnifiedAgentBase
- Migrate test_autonomous_loop.py to test UnifiedAutonomousLoop
- Migrate test_context_manager.py to test UnifiedContextManager
- Coverage > 90%
```

---

### Phase C: 验证和文档 (Week 3)

#### Task C1: 运行完整测试套件
**任务ID:** TASK-C1  
**优先级:** P0  
**预估工时:** 4小时  
**依赖:** TASK-A5, TASK-B4

**描述:**
运行完整的测试套件，确保所有测试通过。

**子任务:**
- [ ] C1.1: 运行单元测试
- [ ] C1.2: 运行集成测试
- [ ] C1.3: 检查测试覆盖率
- [ ] C1.4: 修复失败的测试

**验收标准:**
- 单元测试 100% 通过
- 集成测试 100% 通过
- 覆盖率 > 90%

**提交信息:**
```
Complete test suite verification

- All unit tests pass
- All integration tests pass
- Coverage > 90%
```

---

#### Task C2: 验证无废弃导入
**任务ID:** TASK-C2  
**优先级:** P0  
**预估工时:** 2小时  
**依赖:** TASK-C1

**描述:**
运行迁移检查脚本，验证无废弃导入。

**子任务:**
- [ ] C2.1: 运行 `check_migration.py`
- [ ] C2.2: 修复发现的废弃导入
- [ ] C2.3: 重新验证

**验收标准:**
- `scripts/check_migration.py` 显示无废弃导入

**提交信息:**
```
Verify no deprecated imports remain

- Run check_migration.py
- Fix any remaining deprecated imports
- All checks pass
```

---

#### Task C3: 更新架构文档
**任务ID:** TASK-C3  
**优先级:** P1  
**预估工时:** 6小时  
**依赖:** TASK-C2

**描述:**
更新架构文档，反映新的统一接口。

**子任务:**
- [ ] C3.1: 更新 `docs/architecture.md`
- [ ] C3.2: 更新 `docs/unified_interfaces.md`
- [ ] C3.3: 更新 `README.md`
- [ ] C3.4: 添加迁移说明

**验收标准:**
- 文档与代码同步
- 迁移说明清晰
- 示例代码正确

**提交信息:**
```
Update documentation for unified interfaces

- Update architecture.md
- Update unified_interfaces.md
- Update README.md with migration notes
- Add code examples
```

---

## 第二部分：完美整合任务

### Month 1: 架构层面整合

#### Task M1: 统一依赖注入容器
**任务ID:** TASK-M1  
**优先级:** P1  
**预估工时:** 40小时  
**依赖:** 完整迁移完成

**描述:**
创建统一的依赖注入容器，取代分散的 Manager 类。

**子任务:**
- [ ] M1.1: 设计 DIContainer 接口
- [ ] M1.2: 实现 DIContainer 类
- [ ] M1.3: 注册所有 Manager
- [ ] M1.4: 更新 Agent 使用 DI
- [ ] M1.5: 编写测试
- [ ] M1.6: 编写文档

**验收标准:**
- DIContainer 功能完整
- 所有 Manager 通过 DI 注册
- 测试覆盖率 > 90%

---

#### Task M2: 统一事件系统
**任务ID:** TASK-M2  
**优先级:** P1  
**预估工时:** 32小时  
**依赖:** TASK-M1

**描述:**
整合所有事件/消息机制为统一事件系统。

**子任务:**
- [ ] M2.1: 设计统一 EventBus 接口
- [ ] M2.2: 实现 EventBus 类
- [ ] M2.3: 适配现有代码
- [ ] M2.4: 编写测试
- [ ] M2.5: 编写文档

**验收标准:**
- 支持同步/异步事件
- 支持请求-响应模式
- 向后兼容

---

#### Task M3: 统一配置系统
**任务ID:** TASK-M3  
**优先级:** P1  
**预估工时:** 24小时  
**依赖:** TASK-M1

**描述:**
创建统一的配置系统，集中管理所有配置。

**子任务:**
- [ ] M3.1: 设计 AppConfig 结构
- [ ] M3.2: 实现配置加载/保存
- [ ] M3.3: 实现配置验证
- [ ] M3.4: 迁移现有配置
- [ ] M3.5: 编写测试

**验收标准:**
- 配置加载/保存正常
- 配置验证通过
- 敏感信息保护

---

#### Task M4: 架构验证
**任务ID:** TASK-M4  
**优先级:** P1  
**预估工时:** 16小时  
**依赖:** TASK-M2, TASK-M3

**描述:**
验证架构整合效果。

**子任务:**
- [ ] M4.1: 运行完整测试
- [ ] M4.2: 性能测试
- [ ] M4.3: 代码审查
- [ ] M4.4: 文档审查

**验收标准:**
- 所有测试通过
- 性能无退化
- 架构审查通过

---

### Month 2: 代码层面优化

#### Task M5: 消除重复代码
**任务ID:** TASK-M5  
**优先级:** P1  
**预估工时:** 40小时  
**依赖:** TASK-M4

**描述:**
识别并消除重复代码模式。

**子任务:**
- [ ] M5.1: 代码重复分析
- [ ] M5.2: 提取公共工具类
- [ ] M5.3: 合并重复实现
- [ ] M5.4: 运行测试
- [ ] M5.5: 验证重复率 < 5%

**验收标准:**
- 代码重复率 < 5%
- 所有测试通过

---

#### Task M6: 统一接口契约
**任务ID:** TASK-M6  
**优先级:** P1  
**预估工时:** 32小时  
**依赖:** TASK-M5

**描述:**
定义并统一所有公共接口契约。

**子任务:**
- [ ] M6.1: 定义 IAgent 接口
- [ ] M6.2: 定义 ITool 接口
- [ ] M6.3: 定义 IExecutor 接口
- [ ] M6.4: 更新实现类
- [ ] M6.5: 编写接口文档

**验收标准:**
- 所有公共接口有明确契约
- 接口文档完整

---

#### Task M7: 优化导入结构
**任务ID:** TASK-M7  
**优先级:** P2  
**预估工时:** 24小时  
**依赖:** TASK-M6

**描述:**
优化导入结构，消除循环导入。

**子任务:**
- [ ] M7.1: 分析导入结构
- [ ] M7.2: 消除循环导入
- [ ] M7.3: 统一导入顺序
- [ ] M7.4: 运行测试

**验收标准:**
- 无循环导入
- 导入顺序规范

---

#### Task M8: 代码审查
**任务ID:** TASK-M8  
**优先级:** P1  
**预估工时:** 16小时  
**依赖:** TASK-M7

**描述:**
进行代码审查，确保代码质量。

**子任务:**
- [ ] M8.1: 代码风格检查
- [ ] M8.2: 类型检查
- [ ] M8.3: 安全审查
- [ ] M8.4: 修复问题

**验收标准:**
- 代码风格检查通过
- 类型检查通过

---

### Month 3: 测试和文档

#### Task M9: 统一测试基类
**任务ID:** TASK-M9  
**优先级:** P1  
**预估工时:** 24小时  
**依赖:** TASK-M8

**描述:**
创建统一的测试基类。

**子任务:**
- [ ] M9.1: 设计 AgentTestCase
- [ ] M9.2: 设计 IntegrationTestCase
- [ ] M9.3: 实现测试基类
- [ ] M9.4: 更新现有测试
- [ ] M9.5: 编写文档

**验收标准:**
- 测试基类功能完整
- 所有测试使用新基类

---

#### Task M10: 提高测试覆盖率
**任务ID:** TASK-M10  
**优先级:** P1  
**预估工时:** 40小时  
**依赖:** TASK-M9

**描述:**
提高测试覆盖率到 > 90%。

**子任务:**
- [ ] M10.1: 分析覆盖率报告
- [ ] M10.2: 补充缺失测试
- [ ] M10.3: 优化现有测试
- [ ] M10.4: 验证覆盖率 > 90%

**验收标准:**
- 整体覆盖率 > 90%
- 核心模块覆盖率 > 95%

---

#### Task M11: 性能测试自动化
**任务ID:** TASK-M11  
**优先级:** P2  
**预估工时:** 24小时  
**依赖:** TASK-M10

**描述:**
建立性能测试自动化。

**子任务:**
- [ ] M11.1: 设计性能测试
- [ ] M11.2: 实现性能测试
- [ ] M11.3: 建立性能基线
- [ ] M11.4: 集成到CI

**验收标准:**
- 性能测试自动化
- 性能报告生成

---

#### Task M12: 完善文档
**任务ID:** TASK-M12  
**优先级:** P1  
**预估工时:** 32小时  
**依赖:** TASK-M11

**描述:**
完善所有文档。

**子任务:**
- [ ] M12.1: 生成API文档
- [ ] M12.2: 创建ADR
- [ ] M12.3: 编写开发者指南
- [ ] M12.4: 编写贡献指南
- [ ] M12.5: 更新README

**验收标准:**
- API文档完整
- ADR记录完整
- 开发者指南清晰

---

## 任务依赖图

```
完整迁移:
    A1 (core/__init__.py)
        ├── A2 (execution_steps.py)
        ├── A3 (core_agent.py)
        ├── A4 (agent_initialization.py)
        └── A5 (capability_registry.py)
            ├── B1 (deprecate BaseAgent)
            ├── B2 (deprecate AutonomousLoop)
            └── B3 (deprecate ContextManager)
                └── B4 (update tests)
                    ├── C1 (test suite)
                    ├── C2 (check migration)
                    └── C3 (update docs)

完美整合:
    M1 (DI Container)
        ├── M2 (Event System)
        └── M3 (Config System)
            └── M4 (architecture validation)
                └── M5 (eliminate duplicates)
                    └── M6 (interface contracts)
                        └── M7 (optimize imports)
                            └── M8 (code review)
                                └── M9 (test base classes)
                                    └── M10 (coverage)
                                        └── M11 (performance)
                                            └── M12 (documentation)
```

---

## 任务跟踪表

| 任务ID | 任务名称 | 优先级 | 状态 | 负责人 | 开始日期 | 完成日期 | 工时 |
|--------|----------|--------|------|--------|----------|----------|------|
| TASK-A1 | 迁移 core/__init__.py | P0 | 待开始 | | | | 4h |
| TASK-A2 | 迁移 execution_steps.py | P0 | 待开始 | | | | 3h |
| TASK-A3 | 迁移 core_agent.py | P0 | 待开始 | | | | 3h |
| TASK-A4 | 迁移 agent_initialization.py | P0 | 待开始 | | | | 4h |
| TASK-A5 | 迁移 capability_registry.py | P0 | 待开始 | | | | 4h |
| TASK-B1 | 废弃 BaseAgent 导出 | P0 | 待开始 | | | | 3h |
| TASK-B2 | 废弃 AutonomousLoop 导出 | P0 | 待开始 | | | | 3h |
| TASK-B3 | 废弃 ContextManager 导出 | P0 | 待开始 | | | | 3h |
| TASK-B4 | 更新测试用例 | P1 | 待开始 | | | | 8h |
| TASK-C1 | 完整测试套件 | P0 | 待开始 | | | | 4h |
| TASK-C2 | 验证无废弃导入 | P0 | 待开始 | | | | 2h |
| TASK-C3 | 更新架构文档 | P1 | 待开始 | | | | 6h |
| TASK-M1 | 统一依赖注入 | P1 | 待开始 | | | | 40h |
| TASK-M2 | 统一事件系统 | P1 | 待开始 | | | | 32h |
| TASK-M3 | 统一配置系统 | P1 | 待开始 | | | | 24h |
| TASK-M4 | 架构验证 | P1 | 待开始 | | | | 16h |
| TASK-M5 | 消除重复代码 | P1 | 待开始 | | | | 40h |
| TASK-M6 | 统一接口契约 | P1 | 待开始 | | | | 32h |
| TASK-M7 | 优化导入结构 | P2 | 待开始 | | | | 24h |
| TASK-M8 | 代码审查 | P1 | 待开始 | | | | 16h |
| TASK-M9 | 统一测试基类 | P1 | 待开始 | | | | 24h |
| TASK-M10 | 提高测试覆盖率 | P1 | 待开始 | | | | 40h |
| TASK-M11 | 性能测试自动化 | P2 | 待开始 | | | | 24h |
| TASK-M12 | 完善文档 | P1 | 待开始 | | | | 32h |

**总工时:** 完整迁移 47小时，完美整合 320小时
