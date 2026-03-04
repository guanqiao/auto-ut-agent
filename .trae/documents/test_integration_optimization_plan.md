# PyUT Agent 测试与集成持续优化计划

## 项目概述
当前项目具有良好的测试基础（65个测试文件，1585+测试方法），但缺少关键的工程实践（CI/CD、代码质量工具、统一fixture管理）。本计划旨在系统性地提升测试质量和集成效率。

---

## 阶段一：CI/CD 基础架构搭建

### 1.1 GitHub Actions 工作流配置
**目标**: 建立自动化测试和部署流程

**任务**:
- [ ] 创建 `.github/workflows/test.yml`
  - 配置多Python版本测试矩阵（3.9, 3.10, 3.11, 3.12）
  - 配置缓存加速依赖安装
  - 添加测试报告生成
- [ ] 创建 `.github/workflows/lint.yml`
  - 代码格式检查
  - 静态类型检查
- [ ] 创建 `.github/workflows/coverage.yml`
  - 覆盖率报告生成
  - 覆盖率阈值检查（80%）

**配置示例**:
```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11, 3.12]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
          cache: 'pip'
      - run: pip install -e ".[dev]"
      - run: pytest --cov=pyutagent --cov-report=xml
      - uses: codecov/codecov-action@v3
```

### 1.2 测试环境标准化
**目标**: 确保测试环境一致性

**任务**:
- [ ] 创建 `docker-compose.test.yml`
  - 定义测试服务（如需要数据库）
- [ ] 创建 `Makefile` 或 `taskfile.yml`
  - 统一测试命令：`make test`, `make lint`, `make coverage`
- [ ] 更新 `pyproject.toml` 测试配置
  - 添加测试分类标记（unit, integration, e2e, slow）
  - 配置并行测试（pytest-xdist）

---

## 阶段二：代码质量工具链集成

### 2.1 代码格式化和检查工具
**目标**: 统一代码风格，提升代码质量

**任务**:
- [ ] 配置 **Ruff**（替代flake8 + isort + pydocstyle）
  - 行长度：100字符
  - 启用规则：E, F, I, N, W, UP, B, C4, SIM
  - 配置自动修复
- [ ] 配置 **Black** 代码格式化
  - 行长度：100字符
  - Python目标版本：3.9+
- [ ] 配置 **mypy** 静态类型检查
  - 严格模式
  - 忽略缺失的第三方库stub

**pyproject.toml 配置**:
```toml
[tool.ruff]
line-length = 100
select = ["E", "F", "I", "N", "W", "UP", "B", "C4", "SIM"]
ignore = ["E501"]

[tool.black]
line-length = 100
target-version = ['py39']

[tool.mypy]
python_version = "3.9"
strict = true
ignore_missing_imports = true
```

### 2.2 Pre-commit 钩子
**目标**: 在提交前自动检查代码质量

**任务**:
- [ ] 创建 `.pre-commit-config.yaml`
  - ruff 代码检查
  - black 代码格式化
  - mypy 类型检查
  - 检查大文件、合并冲突标记
- [ ] 添加提交信息规范检查
  - 使用 conventional commits 格式
- [ ] 文档化安装步骤
  ```bash
  pip install pre-commit
  pre-commit install
  ```

### 2.3 代码质量门禁
**目标**: 防止低质量代码进入主分支

**任务**:
- [ ] 配置分支保护规则
  - 要求PR通过所有CI检查
  - 要求代码审查
  - 要求覆盖率不下降
- [ ] 添加代码覆盖率评论机器人
  - 在PR中显示覆盖率变化
  - 标记未覆盖的新代码

---

## 阶段三：测试框架增强

### 3.1 统一 Fixture 管理
**目标**: 消除fixture重复定义，提升测试可维护性

**任务**:
- [ ] 创建 `tests/conftest.py`
  - 提取公共fixture（mock_llm_client, temp_project等）
  - 按模块组织fixture
- [ ] 创建 `tests/fixtures/` 目录
  - `java_code_samples.py` - Java代码样本
  - `maven_projects.py` - Maven项目模板
  - `mock_objects.py` - 常用Mock对象
- [ ] 文档化fixture使用规范

**fixture示例**:
```python
# tests/conftest.py
@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client."""
    client = AsyncMock()
    client.agenerate = AsyncMock(return_value="generated code")
    return client

@pytest.fixture
def sample_java_class():
    """Return a sample Java class code."""
    return """
    public class Calculator {
        public int add(int a, int b) {
            return a + b;
        }
    }
    """
```

### 3.2 测试分类和标记
**目标**: 支持选择性运行测试

**任务**:
- [ ] 添加测试标记到现有测试
  ```python
  @pytest.mark.unit
  @pytest.mark.integration
  @pytest.mark.e2e
  @pytest.mark.slow
  @pytest.mark.flaky
  ```
- [ ] 配置pytest.ini
  ```ini
  [tool.pytest.ini_options]
  markers = [
      "unit: Unit tests",
      "integration: Integration tests",
      "e2e: End-to-end tests",
      "slow: Slow tests (>5s)",
      "flaky: Flaky tests",
  ]
  addopts = "-m 'not slow and not e2e'"
  ```
- [ ] 更新CI配置以分类运行测试
  - PR：只运行unit + integration
  - 主分支：运行所有测试

### 3.3 测试报告和可视化
**目标**: 提升测试结果的可读性和可追溯性

**任务**:
- [ ] 配置 **Allure Report**
  - 添加 `@allure.feature` 和 `@allure.story` 装饰器
  - 配置Allure报告生成
  - 集成到CI/CD
- [ ] 配置 **pytest-html**
  - 生成HTML测试报告
  - 包含测试用例描述和日志
- [ ] 配置 **pytest-metadata**
  - 记录测试环境信息

---

## 阶段四：覆盖率提升

### 4.1 覆盖率基线建立
**目标**: 了解当前覆盖率状况

**任务**:
- [ ] 生成覆盖率报告
  ```bash
  pytest --cov=pyutagent --cov-report=html --cov-report=term-missing
  ```
- [ ] 识别未覆盖的关键代码路径
  - 核心模块：agent, core, tools
  - 边界条件处理
  - 错误处理路径
- [ ] 设置覆盖率基线（当前估计：60-70%）

### 4.2 关键模块覆盖率提升
**目标**: 核心模块覆盖率达到80%+

**任务**:
- [ ] **agent模块**（当前：~50%）
  - 补充ReActAgent测试
  - 补充Action执行测试
  - 补充状态转换测试
- [ ] **core模块**（当前：~65%）
  - 补充错误处理测试
  - 补充配置管理测试
  - 补充测试历史测试（新增）
- [ ] **tools模块**（当前：~70%）
  - 补充Java解析器边界测试
  - 补充Maven工具错误处理测试
- [ ] **ui模块**（当前：~30%）
  - 补充对话框单元测试
  - 使用pytest-qt进行GUI测试

### 4.3 覆盖率监控
**目标**: 防止覆盖率下降

**任务**:
- [ ] 配置覆盖率阈值检查
  ```toml
  [tool.coverage.report]
  fail_under = 80
  skip_covered = false
  ```
- [ ] 集成Codecov或Coveralls
  - PR覆盖率差异评论
  - 覆盖率趋势图表
- [ ] 添加覆盖率徽章到README

---

## 阶段五：集成测试增强

### 5.1 集成测试框架
**目标**: 提升集成测试的稳定性和覆盖度

**任务**:
- [ ] 创建集成测试基类
  - 统一测试环境设置和清理
  - 提供常用断言方法
- [ ] 添加更多集成测试场景
  - 完整工作流测试
  - 错误恢复测试
  - 并发测试
- [ ] 配置集成测试隔离
  - 使用临时目录
  - 清理测试数据

### 5.2 E2E测试增强
**目标**: 确保核心业务流程正常工作

**任务**:
- [ ] 扩展E2E测试覆盖
  - 单文件测试生成流程
  - 批量测试生成流程
  - 配置变更流程
- [ ] 添加E2E测试数据管理
  - 创建标准测试项目
  - 管理测试数据版本
- [ ] 配置E2E测试定时运行
  - 每天定时运行
  - 失败时发送通知

---

## 阶段六：GUI测试集成

### 6.1 GUI测试框架搭建
**目标**: 为新增的GUI组件添加自动化测试

**任务**:
- [ ] 配置 **pytest-qt**
  - 安装pytest-qt和Qt测试依赖
  - 配置Qt测试环境
- [ ] 创建GUI测试基类
  - 提供窗口创建和清理
  - 提供常用GUI操作封装
- [ ] 为核心对话框添加测试
  - TestHistoryDialog测试
  - SetupWizard测试
  - CommandPalette测试

### 6.2 GUI组件单元测试
**目标**: 确保GUI组件行为正确

**任务**:
- [ ] 测试样式管理器
  - 主题切换测试
  - 样式应用测试
- [ ] 测试通知管理器
  - Toast通知显示测试
  - 通知队列测试
- [ ] 测试快捷键系统
  - 快捷键注册测试
  - 快捷键触发测试

---

## 阶段七：性能测试和监控

### 7.1 性能基准测试
**目标**: 建立性能基线，防止性能回归

**任务**:
- [ ] 扩展现有基准测试
  - 添加更多关键路径测试
  - 添加内存使用测试
- [ ] 配置性能回归检测
  - 与历史结果对比
  - 超过阈值时警告
- [ ] 添加性能测试到CI
  - 只在主分支运行
  - 生成性能报告

### 7.2 测试性能优化
**目标**: 提升测试执行速度

**任务**:
- [ ] 识别慢测试
  ```bash
  pytest --durations=10
  ```
- [ ] 优化慢测试
  - 减少不必要的IO操作
  - 使用更轻量级的Mock
  - 并行执行独立测试
- [ ] 配置测试并行执行
  ```bash
  pytest -n auto
  ```

---

## 实施计划

### 第一阶段（1-2周）：基础架构
1. CI/CD工作流配置
2. 代码质量工具集成
3. Pre-commit钩子配置

### 第二阶段（2-3周）：测试框架增强
1. 统一fixture管理
2. 测试分类和标记
3. 测试报告配置

### 第三阶段（2-3周）：覆盖率提升
1. 覆盖率基线建立
2. 关键模块测试补充
3. 覆盖率监控配置

### 第四阶段（1-2周）：GUI测试
1. GUI测试框架搭建
2. 核心组件测试编写

### 第五阶段（1周）：性能优化
1. 性能基准测试
2. 测试执行速度优化

---

## 文件结构规划

```
.github/
├── workflows/
│   ├── test.yml              # 主测试工作流
│   ├── lint.yml              # 代码检查工作流
│   └── coverage.yml          # 覆盖率工作流
├── CODEOWNERS                # 代码审查配置
└── PULL_REQUEST_TEMPLATE.md  # PR模板

tests/
├── conftest.py               # 全局fixture
├── fixtures/                 # 测试数据
│   ├── __init__.py
│   ├── java_code_samples.py
│   ├── maven_projects.py
│   └── mock_objects.py
├── unit/                     # 单元测试
├── integration/              # 集成测试
├── e2e/                      # E2E测试
└── gui/                      # GUI测试（新增）
    ├── __init__.py
    ├── test_main_window.py
    ├── test_dialogs.py
    └── test_components.py

.pyproject.toml               # 更新配置
.pre-commit-config.yaml       # Pre-commit配置
Makefile                      # 常用命令
TESTING.md                    # 测试文档
```

---

## 验收标准

- [ ] CI/CD工作流正常运行
- [ ] 代码质量工具集成完成
- [ ] 所有测试通过（unit + integration）
- [ ] 核心模块覆盖率达到80%
- [ ] GUI组件有基本测试覆盖
- [ ] 性能基准测试建立
- [ ] 测试文档完善

---

## 风险评估与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| CI/CD配置复杂 | 中 | 分步实施，先基础后高级 |
| 测试执行时间过长 | 中 | 并行执行，分类运行 |
| GUI测试不稳定 | 高 | 使用可靠的选择器，增加重试机制 |
| 覆盖率提升困难 | 中 | 优先覆盖核心路径，逐步提升 |
| 工具链冲突 | 低 | 使用虚拟环境，隔离依赖 |
