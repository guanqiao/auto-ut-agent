# JaCoCo 代码覆盖率集成方案

## 1. 概述

本文档详细说明如何在 PyUT Agent VS Code 插件中集成 JaCoCo 代码覆盖率分析功能，实现覆盖率可视化、覆盖率驱动测试生成和测试优化建议。

## 2. 现状分析

### 2.1 当前覆盖率功能

根据代码分析，当前项目已有基础的覆盖率支持：

**后端 (pyutagent)**:
- `CoverageAnalyzer` 类：解析 JaCoCo XML 报告
- `MavenRunner.generate_coverage()`: 生成覆盖率报告
- `CoverageReport` / `FileCoverage` 数据类：存储覆盖率数据
- 支持多种报告路径检测

**VS Code 插件**:
- ❌ 暂无覆盖率相关功能

### 2.2 当前架构限制

1. **覆盖率数据未可视化**: 只在后端解析，前端无法查看
2. **覆盖率未驱动测试生成**: 没有基于覆盖率反馈自动补充测试
3. **缺少覆盖率趋势分析**: 无法查看历史覆盖率变化
4. **缺少覆盖率热力图**: 无法直观看到哪些代码被覆盖

## 3. JaCoCo 技术详解

### 3.1 JaCoCo 工作原理

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│  编译阶段   │ ───► │  测试执行    │ ───► │  报告生成   │
│  字节码插桩 │      │  收集执行数据│      │  XML/HTML   │
└─────────────┘      └──────────────┘      └─────────────┘
```

### 3.2 JaCoCo 报告结构

**XML 报告结构** (`target/site/jacoco/jacoco.xml`):

```xml
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<!DOCTYPE report PUBLIC "-//JACOCO//DTD Report 1.1//EN" "report.dtd">
<report name="My Project">
    <sessioninfo id="session-1" start="1234567890" dump="1234567899"/>
    
    <package name="com/example/service">
        <class name="com/example/service/PaymentService" sourcefilename="PaymentService.java">
            <method name="&lt;init&gt;" desc="()V" line="10">
                <counter type="INSTRUCTION" missed="0" covered="3"/>
                <counter type="LINE" missed="0" covered="1"/>
                <counter type="COMPLEXITY" missed="0" covered="1"/>
                <counter type="METHOD" missed="0" covered="1"/>
            </method>
            
            <method name="processPayment" desc="(LRequest;)Z" line="15">
                <counter type="INSTRUCTION" missed="5" covered="20"/>
                <counter type="BRANCH" missed="1" covered="3"/>
                <counter type="LINE" missed="2" covered="8"/>
                <counter type="COMPLEXITY" missed="1" covered="3"/>
                <counter type="METHOD" missed="0" covered="1"/>
                
                <line nr="15" mi="0" ci="3" mb="0" cb="2"/>
                <line nr="16" mi="0" ci="5" mb="0" cb="0"/>
                <line nr="17" mi="3" ci="0" mb="1" cb="0"/>
                <line nr="18" mi="2" ci="0" mb="0" cb="0"/>
            </method>
            
            <counter type="INSTRUCTION" missed="5" covered="23"/>
            <counter type="BRANCH" missed="1" covered="3"/>
            <counter type="LINE" missed="2" covered="9"/>
            <counter type="COMPLEXITY" missed="1" covered="4"/>
            <counter type="METHOD" missed="0" covered="2"/>
            <counter type="CLASS" missed="0" covered="1"/>
        </class>
    </package>
    
    <counter type="INSTRUCTION" missed="100" covered="900"/>
    <counter type="BRANCH" missed="20" covered="80"/>
    <counter type="LINE" missed="50" covered="450"/>
    <counter type="COMPLEXITY" missed="10" covered="90"/>
    <counter type="METHOD" missed="5" covered="95"/>
    <counter type="CLASS" missed="0" covered="10"/>
</report>
```

### 3.3 覆盖率指标说明

| 指标 | 说明 | 计算方式 |
|------|------|----------|
| **Instruction Coverage** | 字节码指令覆盖率 | covered / (covered + missed) |
| **Branch Coverage** | 分支覆盖率 | covered branches / total branches |
| **Line Coverage** | 行覆盖率 | covered lines / total lines |
| **Complexity Coverage** | 复杂度覆盖率 | 基于圈复杂度计算 |
| **Method Coverage** | 方法覆盖率 | covered methods / total methods |
| **Class Coverage** | 类覆盖率 | covered classes / total classes |

## 4. 实现方案

### 4.1 后端增强 (pyutagent)

#### 4.1.1 增强 CoverageAnalyzer

**文件**: `pyutagent/tools/maven_tools.py`

添加高级覆盖率分析功能：

```python
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import xml.etree.ElementTree as ET
from datetime import datetime

@dataclass
class CoverageChange:
    """覆盖率变化数据"""
    file_path: str
    method_name: str
    old_coverage: float
    new_coverage: float
    change: float
    timestamp: datetime

@dataclass
class CoverageThreshold:
    """覆盖率阈值配置"""
    line: float = 0.8
    branch: float = 0.7
    method: float = 0.9
    class_: float = 1.0

@dataclass
class CoverageTrend:
    """覆盖率趋势数据"""
    timestamp: datetime
    line_coverage: float
    branch_coverage: float
    instruction_coverage: float
    total_tests: int

class CoverageAnalyzer:
    def __init__(self, project_path: str):
        self.project_path = Path(project_path)
        self.possible_report_paths = [
            self.project_path / "target" / "site" / "jacoco" / "jacoco.xml",
            self.project_path / "target" / "jacoco" / "jacoco.xml",
            self.project_path / "build" / "reports" / "jacoco" / "test" / "jacocoTestReport.xml",
        ]
        self.history_file = self.project_path / ".pyutagent" / "coverage_history.json"
    
    def parse_report(self) -> Optional[CoverageReport]:
        """解析 JaCoCo 报告（已有）"""
        # ... 现有实现 ...
        pass
    
    def get_method_coverage(self, file_path: str, method_name: str) -> Optional[float]:
        """获取特定方法的覆盖率"""
        report = self.parse_report()
        if not report:
            return None
        
        for file_cov in report.files:
            if file_path in file_cov.path:
                # 需要增强 FileCoverage 以包含方法级别数据
                # TODO: 实现方法级别覆盖率提取
                pass
        
        return None
    
    def get_uncovered_methods(self) -> List[Dict[str, Any]]:
        """获取未覆盖或覆盖率不足的方法列表"""
        report = self.parse_report()
        if not report:
            return []
        
        uncovered = []
        for file_cov in report.files:
            # 分析方法覆盖率
            methods_with_low_coverage = self._analyze_method_coverage(file_cov)
            uncovered.extend(methods_with_low_coverage)
        
        return uncovered
    
    def _analyze_method_coverage(self, file_cov: FileCoverage) -> List[Dict[str, Any]]:
        """分析方法级别的覆盖率"""
        # 需要解析 XML 中的 method 标签
        # 这部分需要重新设计 FileCoverage 数据结构
        pass
    
    def get_coverage_by_package(self) -> Dict[str, CoverageReport]:
        """按包统计覆盖率"""
        report = self.parse_report()
        if not report:
            return {}
        
        packages = {}
        for file_cov in report.files:
            package_name = file_cov.path.rsplit('/', 1)[0] if '/' in file_cov.path else 'default'
            if package_name not in packages:
                packages[package_name] = {
                    'files': [],
                    'line_coverage': 0.0,
                    'branch_coverage': 0.0,
                }
            packages[package_name]['files'].append(file_cov)
        
        # 计算每个包的平均覆盖率
        for pkg_name, pkg_data in packages.items():
            if pkg_data['files']:
                total_line = sum(f.line_coverage for f in pkg_data['files'])
                total_branch = sum(f.branch_coverage for f in pkg_data['files'])
                pkg_data['line_coverage'] = total_line / len(pkg_data['files'])
                pkg_data['branch_coverage'] = total_branch / len(pkg_data['files'])
        
        return packages
    
    def get_coverage_trend(self) -> List[CoverageTrend]:
        """获取覆盖率趋势"""
        if not self.history_file.exists():
            return []
        
        import json
        try:
            history_data = json.loads(self.history_file.read_text())
            trends = []
            for entry in history_data:
                trends.append(CoverageTrend(
                    timestamp=datetime.fromisoformat(entry['timestamp']),
                    line_coverage=entry['line_coverage'],
                    branch_coverage=entry['branch_coverage'],
                    instruction_coverage=entry.get('instruction_coverage', 0.0),
                    total_tests=entry.get('total_tests', 0)
                ))
            return trends
        except Exception as e:
            logger.error(f"Failed to load coverage history: {e}")
            return []
    
    def save_coverage_snapshot(self, report: CoverageReport, total_tests: int):
        """保存覆盖率快照到历史记录"""
        import json
        
        # 创建历史目录
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 加载现有历史
        history = []
        if self.history_file.exists():
            try:
                history = json.loads(self.history_file.read_text())
            except:
                history = []
        
        # 添加新记录
        history.append({
            'timestamp': datetime.now().isoformat(),
            'line_coverage': report.line_coverage,
            'branch_coverage': report.branch_coverage,
            'instruction_coverage': report.instruction_coverage,
            'total_tests': total_tests
        })
        
        # 保留最近 100 条记录
        history = history[-100:]
        
        # 保存
        self.history_file.write_text(json.dumps(history, indent=2))
    
    def compare_coverage(self, baseline_date: datetime) -> Dict[str, CoverageChange]:
        """比较当前覆盖率与历史基准"""
        current = self.parse_report()
        if not current:
            return {}
        
        # 获取历史基准数据
        history = self.get_coverage_trend()
        baseline = None
        for trend in history:
            if trend.timestamp >= baseline_date:
                baseline = trend
                break
        
        if not baseline:
            return {}
        
        # 计算变化
        changes = {
            'line': CoverageChange(
                file_path='overall',
                method_name='N/A',
                old_coverage=baseline.line_coverage,
                new_coverage=current.line_coverage,
                change=current.line_coverage - baseline.line_coverage,
                timestamp=datetime.now()
            ),
            'branch': CoverageChange(
                file_path='overall',
                method_name='N/A',
                old_coverage=baseline.branch_coverage,
                new_coverage=current.branch_coverage,
                change=current.branch_coverage - baseline.branch_coverage,
                timestamp=datetime.now()
            )
        }
        
        return changes
    
    def check_thresholds(self, thresholds: CoverageThreshold) -> List[str]:
        """检查是否达到覆盖率阈值"""
        report = self.parse_report()
        if not report:
            return ["无法解析覆盖率报告"]
        
        violations = []
        
        if report.line_coverage < thresholds.line:
            violations.append(
                f"行覆盖率 {report.line_coverage:.1%} < 目标 {thresholds.line:.1%}"
            )
        
        if report.branch_coverage < thresholds.branch:
            violations.append(
                f"分支覆盖率 {report.branch_coverage:.1%} < 目标 {thresholds.branch:.1%}"
            )
        
        if report.method_coverage < thresholds.method:
            violations.append(
                f"方法覆盖率 {report.method_coverage:.1%} < 目标 {thresholds.method:.1%}"
            )
        
        if report.class_coverage < thresholds.class_:
            violations.append(
                f"类覆盖率 {report.class_coverage:.1%} < 目标 {thresholds.class_:.1%}"
            )
        
        return violations
    
    def generate_coverage_summary(self) -> str:
        """生成覆盖率摘要文本"""
        report = self.parse_report()
        if not report:
            return "无法获取覆盖率数据"
        
        summary = []
        summary.append("## 覆盖率报告\n")
        summary.append(f"- **行覆盖率**: {report.line_coverage:.1%}")
        summary.append(f"- **分支覆盖率**: {report.branch_coverage:.1%}")
        summary.append(f"- **指令覆盖率**: {report.instruction_coverage:.1%}")
        summary.append(f"- **方法覆盖率**: {report.method_coverage:.1%}")
        summary.append(f"- **类覆盖率**: {report.class_coverage:.1%}")
        summary.append(f"- **覆盖文件数**: {len(report.files)}\n")
        
        # 覆盖率最低的文件
        sorted_files = sorted(report.files, key=lambda f: f.line_coverage)
        if sorted_files:
            summary.append("### 覆盖率最低的文件\n")
            for file_cov in sorted_files[:5]:
                summary.append(f"- `{file_cov.path}`: {file_cov.line_coverage:.1%}")
        
        return "\n".join(summary)
    
    def get_lines_to_cover(self, file_path: str, limit: int = 10) -> List[int]:
        """获取指定文件中需要覆盖的代码行"""
        file_cov = self.get_file_coverage(file_path)
        if not file_cov:
            return []
        
        uncovered = [
            line_num for line_num, is_covered in file_cov.lines 
            if not is_covered
        ]
        
        return uncovered[:limit]
    
    def suggest_tests_for_coverage(self, file_path: str) -> List[Dict[str, Any]]:
        """基于覆盖率分析建议需要补充的测试"""
        uncovered_lines = self.get_lines_to_cover(file_path)
        if not uncovered_lines:
            return []
        
        # 读取源代码，分析未覆盖行的上下文
        source_file = self.project_path / file_path
        if not source_file.exists():
            return []
        
        import ast
        source = source_file.read_text(encoding='utf-8')
        
        try:
            tree = ast.parse(source)
            suggestions = []
            
            # 分析未覆盖行所属的方法
            for line_num in uncovered_lines[:10]:
                method_info = self._find_method_for_line(tree, line_num)
                if method_info:
                    suggestions.append({
                        'line': line_num,
                        'method': method_info['name'],
                        'suggestion': f"为方法 '{method_info['name']}' 添加测试用例",
                        'priority': self._calculate_priority(method_info, line_num)
                    })
            
            return sorted(suggestions, key=lambda s: s['priority'], reverse=True)
        except Exception as e:
            logger.error(f"Failed to analyze source: {e}")
            return []
    
    def _find_method_for_line(self, tree: ast.AST, line_num: int) -> Optional[Dict[str, Any]]:
        """查找包含指定行的方法"""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if node.lineno <= line_num <= node.end_lineno:
                    return {
                        'name': node.name,
                        'start_line': node.lineno,
                        'end_line': node.end_lineno
                    }
        return None
    
    def _calculate_priority(self, method_info: Dict[str, Any], line_num: int) -> int:
        """计算测试建议的优先级"""
        # 简单实现：基于方法大小和未覆盖行数
        method_size = method_info['end_line'] - method_info['start_line']
        if method_size > 50:
            return 3  # 大方法优先
        elif method_size > 20:
            return 2
        else:
            return 1
```

#### 4.1.2 覆盖率驱动测试生成

**文件**: `pyutagent/agent/services/test_generation_service.py`

添加覆盖率驱动的测试生成：

```python
class TestGenerationService:
    def __init__(self, project_path: str):
        self.project_path = Path(project_path)
        self.coverage_analyzer = CoverageAnalyzer(project_path)
    
    def generate_tests_with_coverage_feedback(
        self,
        target_file: str,
        initial_test: str,
        target_coverage: float = 0.8,
        max_iterations: int = 5
    ) -> Tuple[str, CoverageReport]:
        """基于覆盖率反馈生成测试"""
        
        # 第 1 步：保存初始测试
        test_file = self._get_test_file_path(target_file)
        self._save_test_file(test_file, initial_test)
        
        # 第 2 步：运行测试并生成覆盖率
        maven_runner = MavenRunner(str(self.project_path))
        maven_runner.generate_coverage()
        
        # 第 3 步：分析覆盖率
        coverage = self.coverage_analyzer.parse_report()
        if not coverage:
            return initial_test, None
        
        # 第 4 步：迭代改进覆盖率
        iteration = 0
        while coverage.line_coverage < target_coverage and iteration < max_iterations:
            # 获取未覆盖的代码
            uncovered_lines = self.coverage_analyzer.get_lines_to_cover(target_file)
            
            if not uncovered_lines:
                break
            
            # 生成补充测试
            additional_test = self._generate_additional_tests(
                target_file,
                uncovered_lines,
                coverage
            )
            
            # 合并测试
            merged_test = self._merge_tests(initial_test, additional_test)
            self._save_test_file(test_file, merged_test)
            
            # 重新运行测试
            maven_runner.generate_coverage()
            coverage = self.coverage_analyzer.parse_report()
            
            iteration += 1
        
        return merged_test, coverage
    
    def _generate_additional_tests(
        self,
        target_file: str,
        uncovered_lines: List[int],
        coverage: CoverageReport
    ) -> str:
        """生成补充测试以覆盖未覆盖的代码"""
        # 使用 LLM 生成针对性的测试
        # 基于未覆盖行的上下文
        pass
    
    def _merge_tests(self, original: str, additional: str) -> str:
        """合并原始测试和补充测试"""
        # 智能合并测试文件，避免重复
        pass
```

### 4.2 VS Code 插件实现

#### 4.2.1 创建覆盖率服务

**文件**: `pyutagent-vscode/src/coverage/coverageService.ts`

```typescript
import * as vscode from 'vscode';
import { ApiClient } from '../backend/apiClient';

export interface CoverageData {
    lineCoverage: number;
    branchCoverage: number;
    instructionCoverage: number;
    methodCoverage: number;
    classCoverage: number;
    files: FileCoverage[];
}

export interface FileCoverage {
    path: string;
    lineCoverage: number;
    branchCoverage: number;
    lines: LineCoverage[];
}

export interface LineCoverage {
    lineNumber: number;
    isCovered: boolean;
    instructionCount: number;
}

export class CoverageService {
    private api: ApiClient;
    private currentCoverage: CoverageData | null = null;
    private coverageHistory: CoverageData[] = [];
    
    constructor(api: ApiClient) {
        this.api = api;
    }
    
    async generateCoverage(): Promise<CoverageData | null> {
        try {
            // 调用后端生成覆盖率
            const response = await this.api.post('/coverage/generate', {});
            
            if (response.success) {
                this.currentCoverage = response.data;
                this.coverageHistory.push(response.data);
                
                // 保持历史记录最近 100 条
                if (this.coverageHistory.length > 100) {
                    this.coverageHistory.shift();
                }
                
                return response.data;
            }
            
            return null;
        } catch (error) {
            vscode.window.showErrorMessage(`生成覆盖率失败：${error}`);
            return null;
        }
    }
    
    async getCoverageSummary(): Promise<string> {
        try {
            const response = await this.api.get('/coverage/summary');
            return response.summary;
        } catch (error) {
            return `获取覆盖率摘要失败：${error}`;
        }
    }
    
    async getFileCoverage(filePath: string): Promise<FileCoverage | null> {
        try {
            const response = await this.api.get(`/coverage/file?path=${encodeURIComponent(filePath)}`);
            return response.fileCoverage;
        } catch (error) {
            return null;
        }
    }
    
    async getUncoveredLines(filePath: string, limit: number = 10): Promise<number[]> {
        try {
            const response = await this.api.get(
                `/coverage/uncovered?path=${encodeURIComponent(filePath)}&limit=${limit}`
            );
            return response.uncoveredLines;
        } catch (error) {
            return [];
        }
    }
    
    async getCoverageTrend(): Promise<CoverageData[]> {
        return this.coverageHistory;
    }
    
    getCoverageStatus(): 'good' | 'warning' | 'error' {
        if (!this.currentCoverage) {
            return 'error';
        }
        
        const lineCoverage = this.currentCoverage.lineCoverage;
        
        if (lineCoverage >= 0.8) {
            return 'good';
        } else if (lineCoverage >= 0.6) {
            return 'warning';
        } else {
            return 'error';
        }
    }
}
```

#### 4.2.2 创建覆盖率视图

**文件**: `pyutagent-vscode/src/coverage/coverageViewProvider.ts`

```typescript
import * as vscode from 'vscode';
import { CoverageService } from './coverageService';

export class CoverageViewProvider implements vscode.WebviewViewProvider {
    public static readonly viewType = 'pyutagent.coverageView';
    
    private _view?: vscode.WebviewView;
    private _coverageService: CoverageService;
    
    constructor(
        private readonly _extensionUri: vscode.Uri,
        coverageService: CoverageService
    ) {
        this._coverageService = coverageService;
    }
    
    resolveWebviewView(
        webviewView: vscode.WebviewView,
        context: vscode.WebviewViewResolveContext,
        _token: vscode.CancellationToken
    ) {
        this._view = webviewView;
        
        webviewView.webview.options = {
            enableScripts: true,
            localResourceRoots: [this._extensionUri]
        };
        
        webviewView.webview.html = this._getHtmlForWebview(webviewView.webview);
        
        // 监听刷新命令
        vscode.commands.registerCommand('pyutagent.refreshCoverage', () => {
            this.refreshCoverage();
        });
        
        // 监听覆盖率数据更新
        vscode.commands.registerCommand('pyutagent.updateCoverageData', (data: any) => {
            this.updateCoverageDisplay(data);
        });
    }
    
    private async refreshCoverage() {
        if (!this._view) {
            return;
        }
        
        // 显示加载状态
        this._view.webview.postMessage({ type: 'loading' });
        
        // 生成覆盖率
        const coverage = await this._coverageService.generateCoverage();
        
        if (coverage) {
            this._view.webview.postMessage({
                type: 'coverage_update',
                data: coverage
            });
        } else {
            this._view.webview.postMessage({
                type: 'error',
                message: '无法生成覆盖率报告'
            });
        }
    }
    
    private updateCoverageDisplay(data: any) {
        if (this._view) {
            this._view.webview.postMessage({
                type: 'coverage_update',
                data
            });
        }
    }
    
    private _getHtmlForWebview(webview: vscode.Webview): string {
        return `<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>覆盖率视图</title>
    <style>
        :root {
            --vscode-foreground: #cccccc;
            --vscode-editor-background: #1e1e1e;
            --vscode-badge-background: #0e639c;
            --vscode-badge-foreground: #ffffff;
            --coverage-good: #4ec9b0;
            --coverage-warning: #dcdcaa;
            --coverage-error: #f44747;
        }
        
        body {
            font-family: var(--vscode-font-family);
            font-size: var(--vscode-font-size);
            color: var(--vscode-foreground);
            background-color: var(--vscode-editor-background);
            padding: 10px;
        }
        
        .coverage-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }
        
        .coverage-title {
            font-size: 16px;
            font-weight: bold;
        }
        
        .refresh-button {
            background-color: var(--vscode-badge-background);
            color: var(--vscode-badge-foreground);
            border: none;
            padding: 5px 10px;
            cursor: pointer;
            border-radius: 2px;
        }
        
        .coverage-summary {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
            margin-bottom: 20px;
        }
        
        .coverage-metric {
            background-color: rgba(255, 255, 255, 0.05);
            padding: 10px;
            border-radius: 4px;
        }
        
        .metric-label {
            font-size: 12px;
            opacity: 0.8;
        }
        
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            margin-top: 5px;
        }
        
        .metric-value.good { color: var(--coverage-good); }
        .metric-value.warning { color: var(--coverage-warning); }
        .metric-value.error { color: var(--coverage-error); }
        
        .coverage-trend {
            margin-top: 20px;
        }
        
        .trend-chart {
            width: 100%;
            height: 150px;
            background-color: rgba(255, 255, 255, 0.05);
            border-radius: 4px;
        }
        
        .file-list {
            margin-top: 20px;
        }
        
        .file-item {
            display: flex;
            justify-content: space-between;
            padding: 8px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .file-path {
            flex: 1;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        
        .file-coverage {
            font-weight: bold;
        }
        
        .loading {
            text-align: center;
            padding: 20px;
            opacity: 0.6;
        }
    </style>
</head>
<body>
    <div class="coverage-header">
        <div class="coverage-title">代码覆盖率</div>
        <button class="refresh-button" onclick="refreshCoverage()">刷新</button>
    </div>
    
    <div id="loading" class="loading" style="display: none;">
        正在生成覆盖率报告...
    </div>
    
    <div id="content">
        <div class="coverage-summary">
            <div class="coverage-metric">
                <div class="metric-label">行覆盖率</div>
                <div class="metric-value" id="line-coverage">-</div>
            </div>
            <div class="coverage-metric">
                <div class="metric-label">分支覆盖率</div>
                <div class="metric-value" id="branch-coverage">-</div>
            </div>
            <div class="coverage-metric">
                <div class="metric-label">方法覆盖率</div>
                <div class="metric-value" id="method-coverage">-</div>
            </div>
            <div class="coverage-metric">
                <div class="metric-label">类覆盖率</div>
                <div class="metric-value" id="class-coverage">-</div>
            </div>
        </div>
        
        <div class="coverage-trend">
            <h3>覆盖率趋势</h3>
            <canvas id="trend-chart" class="trend-chart"></canvas>
        </div>
        
        <div class="file-list">
            <h3>文件覆盖率</h3>
            <div id="file-list"></div>
        </div>
    </div>
    
    <script>
        const vscode = acquireVsCodeApi();
        
        function refreshCoverage() {
            vscode.postMessage({ command: 'refresh' });
        }
        
        window.addEventListener('message', event => {
            const message = event.data;
            
            if (message.type === 'loading') {
                document.getElementById('loading').style.display = 'block';
                document.getElementById('content').style.display = 'none';
            } else if (message.type === 'coverage_update') {
                document.getElementById('loading').style.display = 'none';
                document.getElementById('content').style.display = 'block';
                updateCoverageDisplay(message.data);
            } else if (message.type === 'error') {
                document.getElementById('loading').style.display = 'none';
                alert(message.message);
            }
        });
        
        function updateCoverageDisplay(data) {
            // 更新覆盖率指标
            document.getElementById('line-coverage').textContent = 
                (data.lineCoverage * 100).toFixed(1) + '%';
            document.getElementById('branch-coverage').textContent = 
                (data.branchCoverage * 100).toFixed(1) + '%';
            document.getElementById('method-coverage').textContent = 
                (data.methodCoverage * 100).toFixed(1) + '%';
            document.getElementById('class-coverage').textContent = 
                (data.classCoverage * 100).toFixed(1) + '%';
            
            // 设置颜色
            setMetricColor('line-coverage', data.lineCoverage);
            setMetricColor('branch-coverage', data.branchCoverage);
            setMetricColor('method-coverage', data.methodCoverage);
            setMetricColor('class-coverage', data.classCoverage);
            
            // 更新文件列表
            updateFileList(data.files);
            
            // 绘制趋势图
            drawTrendChart();
        }
        
        function setMetricColor(elementId, value) {
            const element = document.getElementById(elementId);
            element.classList.remove('good', 'warning', 'error');
            
            if (value >= 0.8) {
                element.classList.add('good');
            } else if (value >= 0.6) {
                element.classList.add('warning');
            } else {
                element.classList.add('error');
            }
        }
        
        function updateFileList(files) {
            const fileList = document.getElementById('file-list');
            fileList.innerHTML = '';
            
            files.forEach(file => {
                const fileItem = document.createElement('div');
                fileItem.className = 'file-item';
                fileItem.innerHTML = \`
                    <div class="file-path">\${file.path}</div>
                    <div class="file-coverage">\${(file.lineCoverage * 100).toFixed(1)}%</div>
                \`;
                fileList.appendChild(fileItem);
            });
        }
        
        function drawTrendChart() {
            // 使用 Chart.js 或其他库绘制趋势图
            // TODO: 实现趋势图绘制
        }
    </script>
</body>
</html>`;
    }
}
```

#### 4.2.3 代码装饰器（Gutter Decoration）

**文件**: `pyutagent-vscode/src/coverage/coverageDecorator.ts`

```typescript
import * as vscode from 'vscode';
import { CoverageService, FileCoverage } from './coverageService';

export class CoverageDecorator {
    private _decorationTypeCovered: vscode.TextEditorDecorationType;
    private _decorationTypeUncovered: vscode.TextEditorDecorationType;
    private _coverageService: CoverageService;
    private _disposable: vscode.Disposable;
    
    constructor(coverageService: CoverageService) {
        this._coverageService = coverageService;
        
        // 定义已覆盖代码的装饰样式
        this._decorationTypeCovered = vscode.window.createTextEditorDecorationType({
            isWholeLine: true,
            backgroundColor: 'rgba(78, 201, 176, 0.2)', // 绿色半透明
            gutterIconPath: this._createIcon('covered')
        });
        
        // 定义未覆盖代码的装饰样式
        this._decorationTypeUncovered = vscode.window.createTextEditorDecorationType({
            isWholeLine: true,
            backgroundColor: 'rgba(244, 71, 71, 0.2)', // 红色半透明
            gutterIconPath: this._createIcon('uncovered')
        });
        
        this._disposable = vscode.Disposable.from(
            this._decorationTypeCovered,
            this._decorationTypeUncovered
        );
    }
    
    private _createIcon(type: 'covered' | 'uncovered'): vscode.Uri {
        // 创建 SVG 图标
        const svg = `
            <svg width="20" height="20" xmlns="http://www.w3.org/2000/svg">
                <circle cx="10" cy="10" r="8" fill="${type === 'covered' ? '#4ec9b0' : '#f44747'}"/>
            </svg>
        `;
        const encoded = Buffer.from(svg).toString('base64');
        return vscode.Uri.parse(`data:image/svg+xml;base64,${encoded}`);
    }
    
    async decorateActiveEditor(): Promise<void> {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            return;
        }
        
        const filePath = editor.document.fileName;
        const coverage = await this._coverageService.getFileCoverage(filePath);
        
        if (!coverage) {
            return;
        }
        
        const coveredRanges: vscode.DecorationOptions[] = [];
        const uncoveredRanges: vscode.DecorationOptions[] = [];
        
        coverage.lines.forEach(line => {
            const range = new vscode.Range(
                line.lineNumber - 1, 0,
                line.lineNumber - 1, editor.document.lineAt(line.lineNumber - 1).text.length
            );
            
            const decoration = {
                range,
                hoverMessage: line.isCovered 
                    ? '✅ 已覆盖' 
                    : '❌ 未覆盖'
            };
            
            if (line.isCovered) {
                coveredRanges.push(decoration);
            } else {
                uncoveredRanges.push(decoration);
            }
        });
        
        editor.setDecorations(this._decorationTypeCovered, coveredRanges);
        editor.setDecorations(this._decorationTypeUncovered, uncoveredRanges);
    }
    
    dispose() {
        this._disposable.dispose();
    }
}
```

#### 4.2.4 更新 package.json

```json
{
  "contributes": {
    "views": {
      "pyutagent": [
        {
          "type": "webview",
          "id": "pyutagent.coverageView",
          "name": "代码覆盖率",
          "icon": "resources/coverage-icon.svg",
          "contextualTitle": "PyUT Agent"
        }
      ]
    },
    "commands": [
      {
        "command": "pyutagent.refreshCoverage",
        "title": "刷新覆盖率",
        "category": "PyUT Agent",
        "icon": "$(refresh)"
      },
      {
        "command": "pyutagent.showCoverage",
        "title": "显示代码覆盖率",
        "category": "PyUT Agent"
      },
      {
        "command": "pyutagent.hideCoverage",
        "title": "隐藏代码覆盖率",
        "category": "PyUT Agent"
      }
    ],
    "menus": {
      "editor/title": [
        {
          "when": "resourceLangId == java",
          "command": "pyutagent.showCoverage",
          "group": "navigation"
        }
      ]
    }
  }
}
```

## 5. 实现优先级

### P0 - 核心功能 (Week 3)

1. ✅ 增强 `CoverageAnalyzer` 类
2. ✅ 实现覆盖率历史记录
3. ✅ 实现覆盖率阈值检查
4. ✅ 创建 VS Code `CoverageService`
5. ✅ 创建覆盖率视图 (`CoverageViewProvider`)
6. ✅ 实现代码装饰器（Gutter Decoration）

### P1 - 增强功能 (Week 3-4)

1. ⏭️ 实现覆盖率趋势图表
2. ⏭️ 实现覆盖率驱动测试生成
3. ⏭️ 实现未覆盖代码建议
4. ⏭️ 实现覆盖率对比功能

### P2 - 优化功能 (Week 4)

1. ⏭️ 实现覆盖率热力图
2. ⏭️ 实现批量文件覆盖率分析
3. ⏭️ 实现覆盖率报告导出

## 6. 测试计划

### 6.1 单元测试

- [ ] 测试 `CoverageAnalyzer` 解析 XML
- [ ] 测试覆盖率阈值检查
- [ ] 测试历史记录保存和加载
- [ ] 测试趋势数据计算

### 6.2 集成测试

- [ ] 测试与 Maven JaCoCo 集成
- [ ] 测试 VS Code 视图更新
- [ ] 测试代码装饰器显示

### 6.3 端到端测试

- [ ] 测试完整覆盖率生成流程
- [ ] 测试覆盖率驱动测试生成
- [ ] 测试用户交互

## 7. 验收标准

### 功能验收

1. ✅ 能够生成 JaCoCo 覆盖率报告
2. ✅ 能够在 VS Code 中可视化显示覆盖率
3. ✅ 能够用颜色标记已覆盖/未覆盖的代码行
4. ✅ 能够显示覆盖率趋势
5. ✅ 能够提供未覆盖代码的测试建议

### 质量验收

1. ✅ 覆盖率数据准确
2. ✅ UI 响应流畅
3. ✅ 装饰器不影响正常编辑
4. ✅ 支持大型项目（1000+ 文件）

## 8. 性能和优化

### 8.1 性能考虑

1. **XML 解析优化**: 使用增量解析，避免重复解析整个文件
2. **缓存策略**: 缓存覆盖率数据，避免频繁调用后端
3. **懒加载**: 只在需要时加载文件级别的覆盖率
4. **异步处理**: 所有覆盖率操作都使用异步

### 8.2 优化措施

1. **增量更新**: 只更新变化的覆盖率数据
2. **后台处理**: 覆盖率生成在后台进行，不阻塞 UI
3. **数据压缩**: 压缩历史数据，减少存储空间

## 9. 参考资料

- [JaCoCo 官方文档](https://www.jacoco.org/jacoco/)
- [JaCoCo Maven 插件](https://www.jacoco.org/jacoco/trunk/doc/maven.html)
- [JaCoCo XML 格式](https://www.jacoco.org/jacoco/trunk/doc/report.dtd)
- [VS Code Decorations API](https://code.visualstudio.com/api/references/vscode-api#TextEditorDecorationType)
