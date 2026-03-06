# PyUT Agent 代码整合 - 验收标准 (Checklist)

## 1. 完整迁移验收标准

### 1.1 代码迁移验收

#### AC-001: 核心模块迁移完成
- [ ] `core/__init__.py` 无废弃导入
- [ ] `EventBus` 和 `AsyncEventBus` 导入已替换为 `UnifiedMessageBus`
- [ ] 所有使用 `core.EventBus` 的代码正常工作
- [ ] 废弃警告正确显示

**验证方法:**
```bash
python scripts/check_migration.py
# 应显示: core/__init__.py 无废弃导入
```

#### AC-002: Agent组件迁移完成
- [ ] `agent/components/execution_steps.py` 无废弃导入
- [ ] `agent/components/core_agent.py` 无废弃导入
- [ ] `agent/components/agent_initialization.py` 无废弃导入
- [ ] `StepResult` 已替换为 `AgentResult`
- [ ] `ContextManager` 已替换为 `UnifiedContextManager`
- [ ] 组件功能等价

**验证方法:**
```bash
python -m pytest tests/unit/agent/components/ -v
python scripts/check_migration.py | grep "agent/components"
```

#### AC-003: Capability Registry迁移完成
- [ ] `agent/capability_registry.py` 无废弃导入
- [ ] `SubAgent` 类型注解已替换为 `UnifiedAgentBase`
- [ ] 类型检查通过
- [ ] 功能等价

**验证方法:**
```bash
mypy pyutagent/agent/capability_registry.py
python -m pytest tests/unit/agent/test_capability_registry.py -v
```

#### AC-004: Agent模块导出清理完成
- [ ] `agent/__init__.py` 中的 `BaseAgent` 导出添加废弃警告
- [ ] `agent/__init__.py` 中的 `AutonomousLoop` 导出添加废弃警告
- [ ] `agent/__init__.py` 中的 `ContextManager` 导出添加废弃警告
- [ ] 导入时显示废弃警告
- [ ] 向后兼容保持

**验证方法:**
```bash
python -c "from pyutagent.agent import BaseAgent" 2>&1 | grep -i "deprecat"
python -c "from pyutagent.agent import AutonomousLoop" 2>&1 | grep -i "deprecat"
```

### 1.2 测试验收

#### AC-005: 单元测试通过
- [ ] 所有单元测试通过 (目标: 100%)
- [ ] 新增测试覆盖迁移代码
- [ ] 测试运行时间 < 5分钟

**验证方法:**
```bash
python -m pytest tests/unit/ -v --tb=short
# 预期: 所有测试通过
```

#### AC-006: 集成测试通过
- [ ] 所有集成测试通过 (目标: 100%)
- [ ] 覆盖主要使用场景
- [ ] 测试运行时间 < 10分钟

**验证方法:**
```bash
python -m pytest tests/integration/ -v --tb=short
# 预期: 所有测试通过
```

#### AC-007: 覆盖率达标
- [ ] 代码覆盖率 > 90%
- [ ] 核心模块覆盖率 > 95%
- [ ] 无未覆盖的关键路径

**验证方法:**
```bash
python -m pytest tests/ --cov=pyutagent --cov-report=term-missing
# 预期: 覆盖率 > 90%
```

### 1.3 代码质量验收

#### AC-008: 无废弃导入
- [ ] `scripts/check_migration.py` 显示无废弃导入
- [ ] 所有文件通过迁移检查

**验证方法:**
```bash
python scripts/check_migration.py
# 预期: [OK] No deprecated imports found!
```

#### AC-009: 类型检查通过
- [ ] `mypy` 无错误
- [ ] 所有公共API有类型注解
- [ ] 类型覆盖率 > 95%

**验证方法:**
```bash
mypy pyutagent/
# 预期: Success: no issues found
```

#### AC-010: 代码风格检查通过
- [ ] `ruff` 或 `flake8` 无错误
- [ ] 符合 PEP 8 规范
- [ ] 文档字符串完整

**验证方法:**
```bash
ruff check pyutagent/
# 预期: 无错误
```

### 1.4 功能验收

#### AC-011: 功能等价
- [ ] 迁移前后功能一致
- [ ] 输出结果一致
- [ ] 边界条件处理一致

**验证方法:**
```bash
# 运行功能测试
python -m pytest tests/functional/ -v
# 对比迁移前后的输出
```

#### AC-012: 性能无退化
- [ ] 消息总线吞吐量 > 1000 msg/sec
- [ ] Agent执行延迟 < 100ms
- [ ] 内存使用无增长

**验证方法:**
```bash
python -m pytest tests/performance/ -v
# 预期: 性能指标达标
```

#### AC-013: 向后兼容
- [ ] 现有代码可运行
- [ ] 废弃警告正确显示
- [ ] 适配器功能正常

**验证方法:**
```bash
# 使用旧接口运行测试
python -c "
import warnings
warnings.filterwarnings('error', category=DeprecationWarning)
from pyutagent.core.event_bus import EventBus
"
# 预期: 抛出 DeprecationWarning 异常
```

---

## 2. 完美整合验收标准

### 2.1 架构整合验收

#### AC-014: 依赖注入系统
- [ ] 统一 `DIContainer` 实现
- [ ] 所有 Manager 通过 DIContainer 注册
- [ ] 单例模式正确
- [ ] 依赖解析正常

**验证方法:**
```bash
python -m pytest tests/unit/core/test_container.py -v
# 预期: 所有测试通过
```

#### AC-015: 统一事件系统
- [ ] 统一 `EventBus` 实现
- [ ] 支持同步/异步事件
- [ ] 支持请求-响应模式
- [ ] 向后兼容

**验证方法:**
```bash
python -m pytest tests/unit/core/messaging/test_event_bus.py -v
# 预期: 所有测试通过
```

#### AC-016: 统一配置系统
- [ ] 统一 `AppConfig` 实现
- [ ] 配置加载/保存功能正常
- [ ] 配置验证通过
- [ ] 敏感信息保护

**验证方法:**
```bash
python -m pytest tests/unit/core/test_config.py -v
# 预期: 所有测试通过
```

### 2.2 代码优化验收

#### AC-017: 代码重复率
- [ ] 代码重复率 < 5%
- [ ] 核心模块无重复代码
- [ ] 工具类已提取

**验证方法:**
```bash
# 使用代码分析工具
jscpd pyutagent/ --threshold 5
# 预期: 重复率 < 5%
```

#### AC-018: 接口契约统一
- [ ] 所有公共接口有明确契约
- [ ] 接口文档完整
- [ ] 向后兼容保证

**验证方法:**
```bash
# 检查接口文档
python -c "from pyutagent.agent import UnifiedAgentBase; help(UnifiedAgentBase)"
# 预期: 文档完整
```

#### AC-019: 导入结构优化
- [ ] 导入结构清晰
- [ ] 无循环导入
- [ ] 导入顺序规范

**验证方法:**
```bash
# 检查循环导入
python -c "import pyutagent"
# 预期: 无循环导入错误
```

### 2.3 测试完善验收

#### AC-020: 测试基类统一
- [ ] 统一 `AgentTestCase` 实现
- [ ] 统一 `IntegrationTestCase` 实现
- [ ] 测试基类功能完整

**验证方法:**
```bash
python -m pytest tests/unit/test_base.py -v
# 预期: 所有测试通过
```

#### AC-021: 测试覆盖率提升
- [ ] 核心模块覆盖率 > 95%
- [ ] 整体覆盖率 > 90%
- [ ] 关键路径全覆盖

**验证方法:**
```bash
python -m pytest tests/ --cov=pyutagent --cov-report=html
# 检查 HTML 报告
```

#### AC-022: 性能测试
- [ ] 性能测试基线建立
- [ ] 性能回归测试自动化
- [ ] 性能报告生成

**验证方法:**
```bash
python -m pytest tests/performance/ --benchmark-only
# 预期: 生成性能报告
```

### 2.4 文档完善验收

#### AC-023: API文档
- [ ] API文档自动生成
- [ ] 所有公共API有文档
- [ ] 文档与代码同步

**验证方法:**
```bash
cd docs && make html
# 预期: 文档生成成功，无警告
```

#### AC-024: 架构决策记录
- [ ] ADR目录创建
- [ ] 重大决策有ADR
- [ ] ADR格式规范

**验证方法:**
```bash
ls docs/adr/
# 预期: 有 ADR 文件
```

#### AC-025: 开发者指南
- [ ] 开发者入门指南
- [ ] 贡献指南
- [ ] 代码规范指南

**验证方法:**
```bash
ls docs/development_guide.md
ls docs/contributing.md
# 预期: 文件存在且内容完整
```

---

## 3. 最终验收清单

### 3.1 完整迁移最终验收

| 检查项 | 状态 | 验证人 | 日期 |
|--------|------|--------|------|
| 代码迁移完成 | [ ] | | |
| 单元测试通过 | [ ] | | |
| 集成测试通过 | [ ] | | |
| 覆盖率达标 | [ ] | | |
| 无废弃导入 | [ ] | | |
| 类型检查通过 | [ ] | | |
| 代码风格检查通过 | [ ] | | |
| 功能等价验证 | [ ] | | |
| 性能无退化 | [ ] | | |
| 向后兼容验证 | [ ] | | |
| 文档更新 | [ ] | | |

### 3.2 完美整合最终验收

| 检查项 | 状态 | 验证人 | 日期 |
|--------|------|--------|------|
| 依赖注入系统 | [ ] | | |
| 统一事件系统 | [ ] | | |
| 统一配置系统 | [ ] | | |
| 代码重复率 < 5% | [ ] | | |
| 接口契约统一 | [ ] | | |
| 导入结构优化 | [ ] | | |
| 测试基类统一 | [ ] | | |
| 测试覆盖率 > 90% | [ ] | | |
| 性能测试自动化 | [ ] | | |
| API文档完整 | [ ] | | |
| ADR记录完整 | [ ] | | |
| 开发者指南 | [ ] | | |

---

## 4. 验收流程

### 4.1 验收步骤

1. **自测阶段**
   - 开发者完成迁移后自测
   - 运行所有测试
   - 检查代码质量

2. **代码审查阶段**
   - 提交PR
   - 代码审查
   - 解决审查意见

3. **CI验证阶段**
   - CI自动运行测试
   - 检查覆盖率
   - 检查代码风格

4. **最终验收阶段**
   - 技术负责人验收
   - 填写验收清单
   - 合并代码

### 4.2 验收标准说明

- **必须满足**: 标记为"必须"的项必须全部通过
- **建议满足**: 标记为"建议"的项应尽量满足
- **例外处理**: 无法满足的项需要说明原因并获得批准

### 4.3 验收失败处理

1. **记录问题**: 详细记录验收失败的原因
2. **修复问题**: 修复导致验收失败的问题
3. **重新验收**: 修复后重新进行验收
4. **例外申请**: 如无法修复，提交例外申请

---

## 5. 验收工具

### 5.1 自动化工具

```bash
# 完整验收检查脚本
#!/bin/bash
set -e

echo "=== 运行验收检查 ==="

echo "1. 迁移检查..."
python scripts/check_migration.py

echo "2. 单元测试..."
python -m pytest tests/unit/ -v --tb=short

echo "3. 集成测试..."
python -m pytest tests/integration/ -v --tb=short

echo "4. 覆盖率检查..."
python -m pytest tests/ --cov=pyutagent --cov-report=term-missing

echo "5. 类型检查..."
mypy pyutagent/

echo "6. 代码风格检查..."
ruff check pyutagent/

echo "=== 验收检查完成 ==="
```

### 5.2 手动检查项

- [ ] 代码审查完成
- [ ] 文档审查完成
- [ ] 架构审查完成
- [ ] 安全审查完成

---

## 6. 验收报告模板

```markdown
# 验收报告

## 项目信息
- 项目名称: PyUT Agent 代码整合
- 验收日期: YYYY-MM-DD
- 验收人: [姓名]

## 验收范围
- [ ] 完整迁移
- [ ] 完美整合

## 验收结果

### 通过项
- [列出通过的验收项]

### 未通过项
- [列出未通过的验收项及原因]

### 例外项
- [列出获得批准的例外项]

## 结论
- [ ] 验收通过
- [ ] 验收不通过
- [ ] 有条件通过

## 建议
[改进建议]

## 签名
- 验收人: _______________
- 日期: _______________
```
