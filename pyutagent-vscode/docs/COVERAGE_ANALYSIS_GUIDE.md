# JaCoCo 覆盖率分析使用指南

## 📋 概述

PyUT Agent v0.2.0 增强了 JaCoCo 覆盖率分析功能，提供以下高级特性:

- ✅ 覆盖率历史记录
- ✅ 覆盖率趋势分析
- ✅ 覆盖率阈值检查
- ✅ 覆盖率对比
- ✅ 测试建议生成
- ✅ Markdown 格式报告

---

## 🚀 快速开始

### 1. 基本使用

```python
from pyutagent.tools.maven_tools import CoverageAnalyzer, MavenRunner

# 创建分析器
analyzer = CoverageAnalyzer('/path/to/project')

# 生成覆盖率报告
maven_runner = MavenRunner('/path/to/project')
maven_runner.generate_coverage()

# 解析报告
report = analyzer.parse_report()
print(f"行覆盖率：{report.line_coverage:.1%}")
print(f"分支覆盖率：{report.branch_coverage:.1%}")
```

### 2. 保存覆盖率快照

```python
# 生成覆盖率后保存快照
report = analyzer.parse_report()
analyzer.save_coverage_snapshot(report, total_tests=100)

# 快照会自动保存到 .pyutagent/coverage_history.json
```

### 3. 查看覆盖率趋势

```python
# 获取最近 30 天的趋势
trends = analyzer.get_coverage_trend(days=30)

for trend in trends:
    print(f"{trend.timestamp}: 行覆盖率={trend.line_coverage:.1%}")
```

### 4. 对比覆盖率变化

```python
# 与上一次对比
changes = analyzer.compare_coverage(baseline_index=-2)

for change in changes:
    direction = "↑" if change.change > 0 else "↓" if change.change < 0 else "="
    print(f"{change.metric}: {change.old_value:.1%} → {change.new_value:.1%} {direction}")
```

### 5. 检查覆盖率阈值

```python
from pyutagent.tools.maven_tools import CoverageThreshold

# 定义阈值
thresholds = CoverageThreshold(
    line=0.8,      # 行覆盖率 >= 80%
    branch=0.7,    # 分支覆盖率 >= 70%
    method=0.9,    # 方法覆盖率 >= 90%
    class_=1.0     # 类覆盖率 >= 100%
)

# 检查是否达标
violations = analyzer.check_thresholds(thresholds)

if violations:
    print("覆盖率未达标:")
    for violation in violations:
        print(f"  ❌ {violation}")
else:
    print("✅ 所有覆盖率指标达标!")
```

### 6. 生成 Markdown 报告

```python
# 生成覆盖率摘要
summary = analyzer.get_coverage_summary()
print(summary)

# 保存到文件
with open('coverage_report.md', 'w') as f:
    f.write(summary)
```

---

## 📊 功能详解

### 覆盖率历史记录

**用途**: 跟踪项目覆盖率变化趋势

**API**:
```python
def save_coverage_snapshot(self, report: CoverageReport, total_tests: int = 0)
```

**示例**:
```python
# 每次生成测试后保存快照
result = await agent.generate_tests(target_file)
report = analyzer.parse_report()
analyzer.save_coverage_snapshot(report, total_tests=result.get('test_count', 0))
```

**存储位置**: `.pyutagent/coverage_history.json`

**数据结构**:
```json
[
  {
    "timestamp": "2026-04-27T10:30:00",
    "line_coverage": 0.75,
    "branch_coverage": 0.65,
    "instruction_coverage": 0.72,
    "method_coverage": 0.85,
    "class_coverage": 0.90,
    "total_tests": 100
  }
]
```

---

### 覆盖率趋势分析

**用途**: 可视化覆盖率变化趋势

**API**:
```python
def get_coverage_trend(self, days: int = 30) -> List[CoverageTrend]
```

**示例**:
```python
# 获取最近 7 天的趋势
trends = analyzer.get_coverage_trend(days=7)

# 绘制趋势图 (使用 matplotlib)
import matplotlib.pyplot as plt

timestamps = [t.timestamp for t in trends]
line_cov = [t.line_coverage * 100 for t in trends]
branch_cov = [t.branch_coverage * 100 for t in trends]

plt.figure(figsize=(10, 6))
plt.plot(timestamps, line_cov, label='Line Coverage')
plt.plot(timestamps, branch_cov, label='Branch Coverage')
plt.xlabel('Date')
plt.ylabel('Coverage (%)')
plt.title('Coverage Trend')
plt.legend()
plt.grid(True)
plt.savefig('coverage_trend.png')
```

---

### 覆盖率对比

**用途**: 比较不同时间点的覆盖率变化

**API**:
```python
def compare_coverage(self, baseline_index: int = -2) -> List[CoverageChange]
```

**参数**:
- `baseline_index`: 历史记录的索引
  - `-2`: 与上一次对比
  - `-3`: 与上上次对比
  - `-10`: 与 10 次前对比

**示例**:
```python
# 与上次对比
changes = analyzer.compare_coverage(baseline_index=-2)

print("覆盖率变化:")
for change in changes:
    if change.change > 0:
        print(f"✅ {change.metric}: +{change.change:.1%}")
    elif change.change < 0:
        print(f"❌ {change.metric}: {change.change:.1%}")
    else:
        print(f"➡️ {change.metric}: 无变化")
```

**输出**:
```
覆盖率变化:
✅ Line Coverage: +2.5%
✅ Branch Coverage: +1.8%
➡️ Instruction Coverage: 无变化
❌ Method Coverage: -0.5%
```

---

### 阈值检查

**用途**: 确保覆盖率达到质量标准

**API**:
```python
def check_thresholds(self, thresholds: CoverageThreshold) -> List[str]
```

**配置阈值**:
```python
from pyutagent.tools.maven_tools import CoverageThreshold

# 严格标准
strict = CoverageThreshold(
    line=0.9,      # 90%
    branch=0.8,    # 80%
    method=0.95,   # 95%
    class_=1.0     # 100%
)

# 宽松标准
relaxed = CoverageThreshold(
    line=0.6,      # 60%
    branch=0.5,    # 50%
    method=0.7,    # 70%
    class_=0.8     # 80%
)
```

**CI/CD 集成**:
```python
# 在 CI 流程中检查覆盖率
violations = analyzer.check_thresholds(strict)

if violations:
    print("❌ 覆盖率未达标，构建失败")
    for v in violations:
        print(f"  - {v}")
    exit(1)
else:
    print("✅ 覆盖率达标，构建成功")
```

---

### 测试建议

**用途**: 基于覆盖率分析提供测试改进建议

**API**:
```python
def suggest_tests_for_coverage(self, file_path: str) -> List[Dict[str, Any]]
```

**示例**:
```python
# 获取文件的测试建议
suggestions = analyzer.suggest_tests_for_coverage(
    'com/example/service/PaymentService.java'
)

for suggestion in suggestions:
    print(f"优先级：{suggestion['priority']}")
    print(f"类型：{suggestion['type']}")
    print(f"描述：{suggestion['description']}")
    print(f"未覆盖行：{suggestion['lines'][:10]}")
    print()
```

**输出**:
```
优先级：high
类型：cover_uncovered_lines
描述：为 25 行未覆盖的代码添加测试
未覆盖行：[15, 16, 17, 28, 29, 45, 46, 67, 68, 89]
```

---

### Markdown 报告

**用途**: 生成可读性强的覆盖率报告

**API**:
```python
def get_coverage_summary(self) -> str
```

**示例输出**:
```markdown
## 覆盖率报告

- **行覆盖率**: 75.3%
- **分支覆盖率**: 68.2%
- **指令覆盖率**: 72.5%
- **方法覆盖率**: 85.1%
- **类覆盖率**: 90.0%
- **覆盖文件数**: 42

### 覆盖率最低的文件

- `com/example/service/LegacyService.java`: 12.5%
- `com/example/util/ComplexUtil.java`: 23.8%
- `com/example/helper/BigHelper.java`: 35.2%
- `com/example/model/RichModel.java`: 42.1%
- `com/example/config/AdvancedConfig.java`: 48.9%
```

---

## 🔧 高级用法

### 1. 自动化覆盖率跟踪

```python
import schedule
import time

def track_coverage():
    """定期跟踪覆盖率"""
    maven_runner = MavenRunner('/path/to/project')
    analyzer = CoverageAnalyzer('/path/to/project')
    
    # 生成覆盖率
    maven_runner.generate_coverage()
    
    # 保存快照
    report = analyzer.parse_report()
    analyzer.save_coverage_snapshot(report)
    
    # 检查阈值
    violations = analyzer.check_thresholds(CoverageThreshold())
    if violations:
        print("⚠️ 覆盖率警告:")
        for v in violations:
            print(f"  - {v}")

# 每天上午 9 点检查
schedule.every().day.at("09:00").do(track_coverage)

while True:
    schedule.run_pending()
    time.sleep(60)
```

### 2. 覆盖率趋势可视化

```python
import plotly.graph_objects as go
from datetime import datetime

def plot_coverage_trend(project_path: str, days: int = 30):
    """绘制交互式覆盖率趋势图"""
    analyzer = CoverageAnalyzer(project_path)
    trends = analyzer.get_coverage_trend(days=days)
    
    if not trends:
        print("没有足够的历史数据")
        return
    
    # 准备数据
    timestamps = [t.timestamp.strftime('%Y-%m-%d') for t in trends]
    line_cov = [t.line_coverage * 100 for t in trends]
    branch_cov = [t.branch_coverage * 100 for t in trends]
    
    # 创建图表
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=line_cov,
        mode='lines+markers',
        name='行覆盖率',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=branch_cov,
        mode='lines+markers',
        name='分支覆盖率',
        line=dict(color='green', width=2)
    ))
    
    fig.update_layout(
        title='覆盖率趋势分析',
        xaxis_title='日期',
        yaxis_title='覆盖率 (%)',
        yaxis=dict(tickformat='.1f'),
        hovermode='x unified',
        height=500
    )
    
    fig.write_html('coverage_trend.html')
    print("趋势图已保存到 coverage_trend.html")

# 使用
plot_coverage_trend('/path/to/project', days=30)
```

### 3. 覆盖率驱动测试生成

```python
async def generate_tests_with_coverage_feedback(
    agent: TestGeneratorAgent,
    analyzer: CoverageAnalyzer,
    target_file: str,
    target_coverage: float = 0.8,
    max_iterations: int = 5
):
    """基于覆盖率反馈生成测试"""
    
    # 第 1 步：生成初始测试
    result = await agent.generate_tests(target_file)
    
    for iteration in range(max_iterations):
        # 分析覆盖率
        report = analyzer.parse_report()
        current_coverage = report.line_coverage if report else 0
        
        print(f"迭代 {iteration + 1}: 当前覆盖率 {current_coverage:.1%}")
        
        # 检查是否达标
        if current_coverage >= target_coverage:
            print("✅ 达到目标覆盖率!")
            break
        
        # 获取未覆盖行
        uncovered = analyzer.get_uncovered_lines(target_file)
        
        if not uncovered:
            print("所有行已覆盖")
            break
        
        # 生成补充测试
        suggestions = analyzer.suggest_tests_for_coverage(target_file)
        
        # 使用建议生成额外测试
        # ... (实现略)
    
    # 保存最终快照
    final_report = analyzer.parse_report()
    analyzer.save_coverage_snapshot(final_report)
```

---

## 📈 最佳实践

### 1. 设置合理的阈值

```python
# 新项目：宽松阈值
new_project = CoverageThreshold(
    line=0.6,
    branch=0.5,
    method=0.7,
    class_=0.8
)

# 成熟项目：严格阈值
mature_project = CoverageThreshold(
    line=0.85,
    branch=0.75,
    method=0.9,
    class_=1.0
)

# 关键系统：非常严格
critical_system = CoverageThreshold(
    line=0.95,
    branch=0.9,
    method=0.98,
    class_=1.0
)
```

### 2. 定期保存快照

```python
# 在以下时机保存快照:
# 1. 每次生成测试后
# 2. 每次代码提交前
# 3. 每天定时任务
# 4. CI/CD 流程中

# CI/CD 集成示例 (GitHub Actions)
# .github/workflows/coverage.yml
# name: Coverage Tracking
# on: [push]
# jobs:
#   coverage:
#     runs-on: ubuntu-latest
#     steps:
#       - uses: actions/checkout@v2
#       - name: Generate Coverage
#         run: mvn test jacoco:report
#       - name: Save Snapshot
#         run: python save_coverage.py
```

### 3. 分析覆盖率趋势

```python
def analyze_trend(trends: List[CoverageTrend]) -> Dict[str, Any]:
    """分析覆盖率趋势"""
    if len(trends) < 2:
        return {'status': 'insufficient_data'}
    
    # 计算变化率
    first = trends[0]
    last = trends[-1]
    
    line_change = last.line_coverage - first.line_coverage
    branch_change = last.branch_coverage - first.branch_coverage
    
    # 判断趋势
    if line_change > 0.05 and branch_change > 0.05:
        status = 'improving_fast'
    elif line_change > 0 or branch_change > 0:
        status = 'improving_slow'
    elif line_change == 0 and branch_change == 0:
        status = 'stable'
    else:
        status = 'declining'
    
    return {
        'status': status,
        'line_change': line_change,
        'branch_change': branch_change,
        'days': len(trends)
    }

# 使用
trends = analyzer.get_coverage_trend(days=30)
analysis = analyze_trend(trends)
print(f"覆盖率趋势：{analysis['status']}")
```

---

## 🐛 故障排除

### 问题 1: 找不到覆盖率报告

**症状**: `parse_report()` 返回 `None`

**解决方案**:
1. 确保已运行 `mvn test jacoco:report`
2. 检查报告路径:
```python
analyzer = CoverageAnalyzer('/path/to/project')
diag = analyzer._get_diagnostic_info()
print(f"搜索路径：{diag['searched_paths']}")
print(f"现有路径：{diag['existing_paths']}")
```

### 问题 2: 历史记录为空

**症状**: `get_coverage_trend()` 返回空列表

**解决方案**:
1. 确保已调用 `save_coverage_snapshot()`
2. 检查历史文件是否存在:
```python
history_file = analyzer.get_coverage_history_file()
print(f"历史文件：{history_file}")
print(f"是否存在：{history_file.exists()}")
```

### 问题 3: 阈值检查不准确

**症状**: 阈值检查通过但实际覆盖率很低

**解决方案**:
1. 调整阈值配置:
```python
# 使用更严格的阈值
strict = CoverageThreshold(line=0.9, branch=0.8)
violations = analyzer.check_thresholds(strict)
```

---

## 📚 相关文档

- [JaCoCo 官方文档](https://www.jacoco.org/jacoco/)
- [Maven JaCoCo 插件](https://www.jacoco.org/jacoco/trunk/doc/maven.html)
- [PyUT Agent v0.2.0 使用指南](USAGE_GUIDE_V0.2.md)

---

**文档版本**: v1.0  
**最后更新**: 2026-04-27  
**维护者**: PyUT Agent 开发团队
