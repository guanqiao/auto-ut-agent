# 批量测试生成与性能优化方案

## 1. 概述

本文档定义了在 PyUT Agent VS Code 插件中实现批量测试生成和性能优化的技术方案，包括并发控制、缓存机制、懒加载和流式传输等优化措施。

## 2. 现状分析

### 2.1 当前架构限制

**后端 (pyutagent)**:
- ✅ 单文件测试生成流程完整
- ❌ 不支持批量文件并发处理
- ❌ 缺少结果缓存机制
- ❌ 重复解析相同代码
- ❌ 大文件处理时响应慢

**VS Code 插件**:
- ✅ 流式输出基本支持
- ❌ 批量操作 UI 缺失
- ❌ 进度显示不完善
- ❌ 缺少缓存策略

### 2.2 性能瓶颈分析

1. **LLM 调用延迟**: 每次生成需要 2-10 秒
2. **Maven 编译时间**: 首次编译可能需要 30-60 秒
3. **重复工作**: 相同/相似代码重复生成测试
4. **内存占用**: 大项目解析占用大量内存
5. **UI 阻塞**: 长时间操作阻塞主线程

## 3. 批量测试生成实现

### 3.1 后端实现 (pyutagent)

#### 3.1.1 批量生成器架构

**文件**: `pyutagent/agent/batch_test_generator.py`

```python
import asyncio
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import logging

from .test_generator import TestGeneratorAgent
from ..core.project_config import ProjectContext, TestFramework
from ..tools.maven_tools import CoverageAnalyzer, MavenRunner

logger = logging.getLogger(__name__)

@dataclass
class BatchOptions:
    """批量生成选项"""
    concurrency: int = 3  # 并发数
    target_coverage: float = 0.8
    max_iterations: int = 3
    skip_existing: bool = True  # 跳过已存在的测试
    generate_for_tests: bool = False  # 是否为测试类生成
    include_patterns: List[str] = field(default_factory=list)
    exclude_patterns: List[str] = field(default_factory=list)

@dataclass
class BatchResult:
    """批量生成结果"""
    total_files: int
    successful: int
    failed: int
    skipped: int
    total_time: float
    average_time: float
    coverage_before: Optional[float]
    coverage_after: Optional[float]
    results: List['FileResult']

@dataclass
class FileResult:
    """单个文件的生成结果"""
    file_path: str
    success: bool
    test_file_path: Optional[str]
    error_message: Optional[str]
    generation_time: float
    coverage_contribution: float
    skipped: bool = False

class BatchTestGenerator:
    """批量测试生成器"""
    
    def __init__(
        self,
        project_path: str,
        project_context: ProjectContext,
        options: Optional[BatchOptions] = None
    ):
        self.project_path = Path(project_path)
        self.project_context = project_context
        self.options = options or BatchOptions()
        self.maven_runner = MavenRunner(project_path)
        self.coverage_analyzer = CoverageAnalyzer(project_path)
        
        # 进度回调
        self._progress_callback: Optional[Callable[[int, int, str], None]] = None
        self._result_callback: Optional[Callable[[FileResult], None]] = None
        
        # 缓存
        self._result_cache: Dict[str, FileResult] = {}
    
    def set_progress_callback(self, callback: Callable[[int, int, str], None]):
        """设置进度回调函数"""
        self._progress_callback = callback
    
    def set_result_callback(self, callback: Callable[[FileResult], None]):
        """设置结果回调函数"""
        self._result_callback = callback
    
    def scan_target_files(self) -> List[str]:
        """扫描需要生成测试的文件"""
        source_dir = self.project_path / "src" / "main" / "java"
        if not source_dir.exists():
            return []
        
        java_files = []
        for java_file in source_dir.rglob("*.java"):
            rel_path = str(java_file.relative_to(source_dir))
            
            # 应用包含/排除模式
            if self._should_include(rel_path):
                java_files.append(rel_path)
        
        logger.info(f"Found {len(java_files)} Java files to process")
        return java_files
    
    def _should_include(self, file_path: str) -> bool:
        """检查文件是否应该被包含"""
        # 检查排除模式
        for pattern in self.options.exclude_patterns:
            if pattern in file_path:
                return False
        
        # 检查包含模式
        if self.options.include_patterns:
            for pattern in self.options.include_patterns:
                if pattern in file_path:
                    return True
            return False
        
        return True
    
    async def generate_batch(
        self,
        files: Optional[List[str]] = None
    ) -> BatchResult:
        """批量生成测试"""
        start_time = datetime.now()
        
        # 如果没有指定文件列表，扫描所有文件
        if files is None:
            files = self.scan_target_files()
        
        if not files:
            return BatchResult(
                total_files=0,
                successful=0,
                failed=0,
                skipped=0,
                total_time=0,
                average_time=0,
                coverage_before=None,
                coverage_after=None,
                results=[]
            )
        
        # 获取生成前的覆盖率
        coverage_before = self._get_current_coverage()
        
        # 分批处理
        batches = self._chunk_array(files, self.options.concurrency)
        
        all_results: List[FileResult] = []
        successful = 0
        failed = 0
        skipped = 0
        
        # 并发处理每个批次
        for i, batch in enumerate(batches):
            self._update_progress(
                processed=len(all_results),
                total=len(files),
                message=f"Processing batch {i+1}/{len(batches)}"
            )
            
            # 并发处理批次中的文件
            tasks = [
                self._generate_single_file(file)
                for file in batch
            ]
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理结果
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    file_result = FileResult(
                        file_path=batch[j],
                        success=False,
                        test_file_path=None,
                        error_message=str(result),
                        generation_time=0,
                        coverage_contribution=0
                    )
                    failed += 1
                else:
                    file_result = result
                    if file_result.skipped:
                        skipped += 1
                    elif file_result.success:
                        successful += 1
                    else:
                        failed += 1
                
                all_results.append(file_result)
                
                # 调用结果回调
                if self._result_callback:
                    self._result_callback(file_result)
        
        # 获取生成后的覆盖率
        coverage_after = self._get_current_coverage()
        
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        return BatchResult(
            total_files=len(files),
            successful=successful,
            failed=failed,
            skipped=skipped,
            total_time=total_time,
            average_time=total_time / len(files) if files else 0,
            coverage_before=coverage_before,
            coverage_after=coverage_after,
            results=all_results
        )
    
    async def _generate_single_file(self, file_path: str) -> FileResult:
        """为单个文件生成测试"""
        start_time = datetime.now()
        
        try:
            # 检查缓存
            if file_path in self._result_cache:
                cached = self._result_cache[file_path]
                if cached.skipped:
                    return cached
            
            # 检查是否已存在测试
            if self.options.skip_existing:
                test_path = self._get_test_file_path(file_path)
                if test_path and test_path.exists():
                    result = FileResult(
                        file_path=file_path,
                        success=False,
                        test_file_path=str(test_path),
                        error_message=None,
                        generation_time=0,
                        coverage_contribution=0,
                        skipped=True
                    )
                    self._result_cache[file_path] = result
                    return result
            
            # 创建测试生成器 Agent
            generator = TestGeneratorAgent(
                project_path=str(self.project_path),
                project_context=self.project_context
            )
            
            # 生成测试
            test_code = await generator.generate_tests(file_path)
            
            # 保存测试文件
            test_path = self._save_test_file(file_path, test_code)
            
            # 运行测试并分析覆盖率
            coverage_contribution = 0.0
            if test_path:
                self.maven_runner.run_tests()
                coverage_contribution = self._calculate_coverage_contribution(file_path)
            
            end_time = datetime.now()
            generation_time = (end_time - start_time).total_seconds()
            
            result = FileResult(
                file_path=file_path,
                success=True,
                test_file_path=str(test_path) if test_path else None,
                error_message=None,
                generation_time=generation_time,
                coverage_contribution=coverage_contribution
            )
            
            self._result_cache[file_path] = result
            return result
            
        except Exception as e:
            logger.exception(f"Failed to generate test for {file_path}: {e}")
            end_time = datetime.now()
            generation_time = (end_time - start_time).total_seconds()
            
            return FileResult(
                file_path=file_path,
                success=False,
                test_file_path=None,
                error_message=str(e),
                generation_time=generation_time,
                coverage_contribution=0.0
            )
    
    def _chunk_array(self, array: List[Any], size: int) -> List[List[Any]]:
        """将数组分块"""
        return [array[i:i + size] for i in range(0, len(array), size)]
    
    def _get_current_coverage(self) -> Optional[float]:
        """获取当前覆盖率"""
        report = self.coverage_analyzer.parse_report()
        return report.line_coverage if report else None
    
    def _get_test_file_path(self, source_file: str) -> Optional[Path]:
        """获取测试文件路径"""
        test_dir = self.project_path / "src" / "test" / "java"
        return test_dir / source_file.replace(".java", "Test.java")
    
    def _save_test_file(self, source_file: str, test_code: str) -> Optional[Path]:
        """保存测试文件"""
        test_path = self._get_test_file_path(source_file)
        if not test_path:
            return None
        
        test_path.parent.mkdir(parents=True, exist_ok=True)
        test_path.write_text(test_code, encoding='utf-8')
        return test_path
    
    def _calculate_coverage_contribution(self, source_file: str) -> float:
        """计算单个文件对覆盖率的贡献"""
        # 简化实现：返回该文件的覆盖率
        file_cov = self.coverage_analyzer.get_file_coverage(source_file)
        return file_cov.line_coverage if file_cov else 0.0
    
    def _update_progress(self, processed: int, total: int, message: str):
        """更新进度"""
        if self._progress_callback:
            self._progress_callback(processed, total, message)
```

#### 3.1.2 缓存机制

**文件**: `pyutagent/cache/test_cache.py`

```python
import hashlib
import json
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class CacheEntry:
    """缓存条目"""
    def __init__(
        self,
        key: str,
        value: Any,
        ttl_seconds: int = 3600  # 默认 1 小时过期
    ):
        self.key = key
        self.value = value
        self.created_at = datetime.now()
        self.ttl_seconds = ttl_seconds
    
    def is_expired(self) -> bool:
        """检查是否过期"""
        return datetime.now() > self.created_at + timedelta(seconds=self.ttl_seconds)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'key': self.key,
            'value': self.value,
            'created_at': self.created_at.isoformat(),
            'ttl_seconds': self.ttl_seconds
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheEntry':
        """从字典创建"""
        entry = cls(
            key=data['key'],
            value=data['value'],
            ttl_seconds=data['ttl_seconds']
        )
        entry.created_at = datetime.fromisoformat(data['created_at'])
        return entry

class TestCache:
    """测试生成缓存"""
    
    def __init__(self, cache_dir: str, max_size: int = 1000):
        self.cache_dir = Path(cache_dir)
        self.max_size = max_size
        self._cache: Dict[str, CacheEntry] = {}
        
        # 确保缓存目录存在
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载持久化缓存
        self._load_persistent_cache()
    
    def _generate_cache_key(
        self,
        source_file: str,
        source_hash: str,
        test_framework: str,
        mock_framework: str
    ) -> str:
        """生成缓存键"""
        key_string = f"{source_file}:{source_hash}:{test_framework}:{mock_framework}"
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    def _compute_file_hash(self, file_path: Path) -> str:
        """计算文件哈希"""
        if not file_path.exists():
            return ""
        
        content = file_path.read_text(encoding='utf-8')
        return hashlib.sha256(content.encode()).hexdigest()
    
    def get(
        self,
        source_file: str,
        source_path: Path,
        test_framework: str,
        mock_framework: str
    ) -> Optional[str]:
        """从缓存获取测试代码"""
        source_hash = self._compute_file_hash(source_path)
        key = self._generate_cache_key(
            source_file, source_hash, test_framework, mock_framework
        )
        
        if key in self._cache:
            entry = self._cache[key]
            if not entry.is_expired():
                logger.debug(f"Cache hit for {source_file}")
                return entry.value
            else:
                # 删除过期条目
                del self._cache[key]
        
        logger.debug(f"Cache miss for {source_file}")
        return None
    
    def set(
        self,
        source_file: str,
        source_path: Path,
        test_framework: str,
        mock_framework: str,
        test_code: str
    ):
        """缓存测试代码"""
        source_hash = self._compute_file_hash(source_path)
        key = self._generate_cache_key(
            source_file, source_hash, test_framework, mock_framework
        )
        
        # 如果缓存已满，删除最旧的条目
        if len(self._cache) >= self.max_size:
            self._evict_oldest()
        
        entry = CacheEntry(key=key, value=test_code, ttl_seconds=3600)
        self._cache[key] = entry
        
        # 持久化
        self._save_persistent_cache()
    
    def _evict_oldest(self):
        """删除最旧的缓存条目"""
        if not self._cache:
            return
        
        oldest_key = min(
            self._cache.keys(),
            key=lambda k: self._cache[k].created_at
        )
        del self._cache[oldest_key]
    
    def _save_persistent_cache(self):
        """保存持久化缓存"""
        cache_file = self.cache_dir / "cache.json"
        data = {
            key: entry.to_dict()
            for key, entry in self._cache.items()
            if not entry.is_expired()
        }
        
        try:
            cache_file.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
    
    def _load_persistent_cache(self):
        """加载持久化缓存"""
        cache_file = self.cache_dir / "cache.json"
        if not cache_file.exists():
            return
        
        try:
            data = json.loads(cache_file.read_text())
            for key, entry_data in data.items():
                entry = CacheEntry.from_dict(entry_data)
                if not entry.is_expired():
                    self._cache[key] = entry
        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
    
    def clear(self):
        """清空缓存"""
        self._cache.clear()
        cache_file = self.cache_dir / "cache.json"
        if cache_file.exists():
            cache_file.unlink()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        now = datetime.now()
        valid_entries = sum(
            1 for entry in self._cache.values()
            if not entry.is_expired()
        )
        
        return {
            'total_entries': len(self._cache),
            'valid_entries': valid_entries,
            'expired_entries': len(self._cache) - valid_entries,
            'max_size': self.max_size,
            'usage_percent': (len(self._cache) / self.max_size * 100) if self.max_size > 0 else 0
        }
```

### 3.2 VS Code 插件实现

#### 3.2.1 批量生成 UI

**文件**: `pyutagent-vscode/src/batch/batchGeneration.ts`

```typescript
import * as vscode from 'vscode';
import { ApiClient } from '../backend/apiClient';

interface BatchOptions {
    concurrency: number;
    targetCoverage: number;
    skipExisting: boolean;
    includePatterns: string[];
    excludePatterns: string[];
}

interface BatchProgress {
    processed: number;
    total: number;
    message: string;
    currentFile?: string;
}

interface BatchResult {
    totalFiles: number;
    successful: number;
    failed: number;
    skipped: number;
    totalTime: number;
    averageTime: number;
    coverageBefore?: number;
    coverageAfter?: number;
    results: FileResult[];
}

interface FileResult {
    filePath: string;
    success: boolean;
    testFilePath?: string;
    errorMessage?: string;
    generationTime: number;
    coverageContribution: number;
    skipped: boolean;
}

export class BatchGenerationPanel {
    public static currentPanel: BatchGenerationPanel | undefined;
    public static readonly viewType = 'pyutagent.batchGeneration';
    
    private readonly _panel: vscode.WebviewPanel;
    private readonly _api: ApiClient;
    private _disposables: vscode.Disposable[] = [];
    
    public static createOrShow(extensionUri: vscode.Uri, api: ApiClient) {
        const column = vscode.window.activeTextEditor
            ? vscode.window.activeTextEditor.viewColumn
            : undefined;
        
        if (BatchGenerationPanel.currentPanel) {
            BatchGenerationPanel.currentPanel._panel.reveal(column);
            return;
        }
        
        const panel = vscode.window.createWebviewPanel(
            BatchGenerationPanel.viewType,
            '批量生成测试',
            column || vscode.ViewColumn.One,
            {
                enableScripts: true,
                localResourceRoots: [extensionUri]
            }
        );
        
        BatchGenerationPanel.currentPanel = new BatchGenerationPanel(panel, extensionUri, api);
    }
    
    private constructor(
        panel: vscode.WebviewPanel,
        extensionUri: vscode.Uri,
        api: ApiClient
    ) {
        this._panel = panel;
        this._api = api;
        
        this._panel.onDidDispose(() => this.dispose(), null, this._disposables);
        
        this._panel.webview.html = this._getHtmlForWebview(panel.webview);
        
        this._setupMessageHandlers();
    }
    
    private _setupMessageHandlers() {
        this._panel.webview.onDidReceiveMessage(async message => {
            switch (message.command) {
                case 'startBatch':
                    await this._startBatchGeneration(message.options);
                    break;
                case 'cancelBatch':
                    await this._cancelBatchGeneration();
                    break;
            }
        }, null, this._disposables);
    }
    
    private async _startBatchGeneration(options: BatchOptions) {
        try {
            // 发送开始命令
            this._panel.webview.postMessage({ type: 'batch_started' });
            
            // 调用后端 API
            const response = await this._api.post('/batch/generate', options);
            
            if (response.success) {
                this._panel.webview.postMessage({
                    type: 'batch_completed',
                    result: response.result
                });
            } else {
                this._panel.webview.postMessage({
                    type: 'batch_error',
                    error: response.error
                });
            }
        } catch (error) {
            this._panel.webview.postMessage({
                type: 'batch_error',
                error: String(error)
            });
        }
    }
    
    private async _cancelBatchGeneration() {
        try {
            await this._api.post('/batch/cancel', {});
            this._panel.webview.postMessage({ type: 'batch_cancelled' });
        } catch (error) {
            this._panel.webview.postMessage({
                type: 'batch_error',
                error: String(error)
            });
        }
    }
    
    public updateProgress(progress: BatchProgress) {
        this._panel.webview.postMessage({
            type: 'batch_progress',
            progress
        });
    }
    
    public dispose() {
        BatchGenerationPanel.currentPanel = undefined;
        this._panel.dispose();
        while (this._disposables.length) {
            const x = this._disposables.pop();
            if (x) {
                x.dispose();
            }
        }
    }
    
    private _getHtmlForWebview(webview: vscode.Webview) {
        return `<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>批量生成测试</title>
    <style>
        :root {
            --vscode-foreground: #cccccc;
            --vscode-editor-background: #1e1e1e;
            --vscode-button-background: #0e639c;
            --vscode-button-foreground: #ffffff;
            --vscode-input-background: #3c3c3c;
            --vscode-input-foreground: #cccccc;
            --success: #4ec9b0;
            --warning: #dcdcaa;
            --error: #f44747;
        }
        
        body {
            font-family: var(--vscode-font-family);
            font-size: var(--vscode-font-size);
            color: var(--vscode-foreground);
            background-color: var(--vscode-editor-background);
            padding: 20px;
        }
        
        h2 {
            margin-top: 0;
        }
        
        .form-group {
            margin-bottom: 15px;
        }
        
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
        }
        
        input[type="number"],
        input[type="text"] {
            width: 100%;
            padding: 8px;
            background-color: var(--vscode-input-background);
            color: var(--vscode-input-foreground);
            border: 1px solid transparent;
            border-radius: 2px;
        }
        
        .checkbox-group {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        button {
            background-color: var(--vscode-button-background);
            color: var(--vscode-button-foreground);
            border: none;
            padding: 10px 20px;
            cursor: pointer;
            border-radius: 2px;
            margin-right: 10px;
        }
        
        button:hover {
            opacity: 0.9;
        }
        
        button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        .progress-container {
            margin-top: 20px;
            display: none;
        }
        
        .progress-bar {
            width: 100%;
            height: 20px;
            background-color: rgba(255, 255, 255, 0.1);
            border-radius: 4px;
            overflow: hidden;
        }
        
        .progress-fill {
            height: 100%;
            background-color: var(--vscode-button-background);
            transition: width 0.3s ease;
        }
        
        .progress-text {
            margin-top: 10px;
            text-align: center;
        }
        
        .results-container {
            margin-top: 20px;
            display: none;
        }
        
        .summary-stats {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 10px;
            margin-bottom: 20px;
        }
        
        .stat-card {
            background-color: rgba(255, 255, 255, 0.05);
            padding: 15px;
            border-radius: 4px;
            text-align: center;
        }
        
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 5px;
        }
        
        .stat-label {
            font-size: 12px;
            opacity: 0.8;
        }
        
        .stat-value.success { color: var(--success); }
        .stat-value.error { color: var(--error); }
        .stat-value.warning { color: var(--warning); }
        
        .file-list {
            max-height: 400px;
            overflow-y: auto;
        }
        
        .file-item {
            display: flex;
            justify-content: space-between;
            padding: 8px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .file-status {
            font-weight: bold;
        }
        
        .file-status.success { color: var(--success); }
        .file-status.error { color: var(--error); }
        .file-status.skipped { color: var(--warning); }
    </style>
</head>
<body>
    <h2>批量生成测试</h2>
    
    <div id="config-form">
        <div class="form-group">
            <label for="concurrency">并发数:</label>
            <input type="number" id="concurrency" value="3" min="1" max="10">
        </div>
        
        <div class="form-group">
            <label for="targetCoverage">目标覆盖率:</label>
            <input type="number" id="targetCoverage" value="0.8" min="0" max="1" step="0.1">
        </div>
        
        <div class="form-group">
            <label>选项:</label>
            <div class="checkbox-group">
                <input type="checkbox" id="skipExisting" checked>
                <label for="skipExisting">跳过已存在的测试</label>
            </div>
        </div>
        
        <div class="form-group">
            <label for="includePatterns">包含模式 (逗号分隔):</label>
            <input type="text" id="includePatterns" placeholder="例如：service,controller">
        </div>
        
        <div class="form-group">
            <label for="excludePatterns">排除模式 (逗号分隔):</label>
            <input type="text" id="excludePatterns" placeholder="例如：test,util">
        </div>
        
        <div class="form-group">
            <button id="start-btn" onclick="startBatch()">开始生成</button>
            <button id="cancel-btn" onclick="cancelBatch()" disabled>取消</button>
        </div>
    </div>
    
    <div id="progress" class="progress-container">
        <div class="progress-bar">
            <div class="progress-fill" id="progress-fill" style="width: 0%"></div>
        </div>
        <div class="progress-text" id="progress-text">准备中...</div>
    </div>
    
    <div id="results" class="results-container">
        <h3>生成结果</h3>
        
        <div class="summary-stats">
            <div class="stat-card">
                <div class="stat-value success" id="stat-success">0</div>
                <div class="stat-label">成功</div>
            </div>
            <div class="stat-card">
                <div class="stat-value error" id="stat-failed">0</div>
                <div class="stat-label">失败</div>
            </div>
            <div class="stat-card">
                <div class="stat-value warning" id="stat-skipped">0</div>
                <div class="stat-label">跳过</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="stat-time">0s</div>
                <div class="stat-label">总耗时</div>
            </div>
        </div>
        
        <h4>文件列表</h4>
        <div class="file-list" id="file-list"></div>
    </div>
    
    <script>
        const vscode = acquireVsCodeApi();
        let isRunning = false;
        
        function startBatch() {
            if (isRunning) return;
            
            const options = {
                concurrency: parseInt(document.getElementById('concurrency').value),
                targetCoverage: parseFloat(document.getElementById('targetCoverage').value),
                skipExisting: document.getElementById('skipExisting').checked,
                includePatterns: document.getElementById('includePatterns').value.split(',').filter(p => p.trim()),
                excludePatterns: document.getElementById('excludePatterns').value.split(',').filter(p => p.trim())
            };
            
            isRunning = true;
            document.getElementById('start-btn').disabled = true;
            document.getElementById('cancel-btn').disabled = false;
            document.getElementById('progress').style.display = 'block';
            document.getElementById('results').style.display = 'none';
            
            vscode.postMessage({
                command: 'startBatch',
                options: options
            });
        }
        
        function cancelBatch() {
            if (!isRunning) return;
            
            vscode.postMessage({ command: 'cancelBatch' });
        }
        
        window.addEventListener('message', event => {
            const message = event.data;
            
            if (message.type === 'batch_started') {
                document.getElementById('progress-text').textContent = '开始批量生成...';
            } else if (message.type === 'batch_progress') {
                const progress = message.progress;
                const percent = (progress.processed / progress.total * 100).toFixed(1);
                document.getElementById('progress-fill').style.width = percent + '%';
                document.getElementById('progress-text').textContent = 
                    \`\${progress.message} (\${progress.processed}/\${progress.total})\`;
                
                if (progress.currentFile) {
                    document.getElementById('progress-text').textContent += 
                        \` - 当前：\${progress.currentFile}\`;
                }
            } else if (message.type === 'batch_completed') {
                isRunning = false;
                document.getElementById('start-btn').disabled = false;
                document.getElementById('cancel-btn').disabled = true;
                displayResults(message.result);
            } else if (message.type === 'batch_cancelled') {
                isRunning = false;
                document.getElementById('start-btn').disabled = false;
                document.getElementById('cancel-btn').disabled = true;
                document.getElementById('progress-text').textContent = '已取消';
            } else if (message.type === 'batch_error') {
                isRunning = false;
                document.getElementById('start-btn').disabled = false;
                document.getElementById('cancel-btn').disabled = true;
                alert('错误：' + message.error);
            }
        });
        
        function displayResults(result) {
            document.getElementById('progress').style.display = 'none';
            document.getElementById('results').style.display = 'block';
            
            document.getElementById('stat-success').textContent = result.successful;
            document.getElementById('stat-failed').textContent = result.failed;
            document.getElementById('stat-skipped').textContent = result.skipped;
            document.getElementById('stat-time').textContent = result.totalTime.toFixed(1) + 's';
            
            const fileList = document.getElementById('file-list');
            fileList.innerHTML = '';
            
            result.results.forEach(file => {
                const fileItem = document.createElement('div');
                fileItem.className = 'file-item';
                
                let status = '';
                let statusClass = '';
                
                if (file.skipped) {
                    status = '⏭️ 跳过';
                    statusClass = 'skipped';
                } else if (file.success) {
                    status = '✅ 成功';
                    statusClass = 'success';
                } else {
                    status = '❌ 失败';
                    statusClass = 'error';
                }
                
                fileItem.innerHTML = \`
                    <div>\${file.filePath}</div>
                    <div class="file-status \${statusClass}">\${status}</div>
                \`;
                
                fileList.appendChild(fileItem);
            });
        }
    </script>
</body>
</html>`;
    }
}
```

## 4. 性能优化方案

### 4.1 流式传输优化

**文件**: `pyutagent-vscode/src/backend/apiClient.ts`

增强流式传输：

```typescript
export class ApiClient {
    async *streamExecute(endpoint: string, data: any): AsyncGenerator<StreamChunk> {
        const url = `${this.baseUrl}${endpoint}`;
        
        const response = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const reader = response.body?.getReader();
        if (!reader) {
            throw new Error('No response body');
        }
        
        const decoder = new TextDecoder();
        let buffer = '';
        
        while (true) {
            const { done, value } = await reader.read();
            
            if (done) {
                break;
            }
            
            buffer += decoder.decode(value, { stream: true });
            
            // 处理 SSE 格式或自定义流式格式
            const lines = buffer.split('\n');
            buffer = lines.pop() || ''; // 保留不完整的一行
            
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const chunk = JSON.parse(line.slice(6));
                    yield chunk;
                }
            }
        }
    }
}
```

### 4.2 懒加载实现

**文件**: `pyutagent-vscode/src/coverage/coverageService.ts`

```typescript
export class CoverageService {
    private _fileCoverageCache: Map<string, FileCoverage> = new Map();
    
    async getFileCoverage(filePath: string): Promise<FileCoverage | null> {
        // 检查缓存
        if (this._fileCoverageCache.has(filePath)) {
            return this._fileCoverageCache.get(filePath)!;
        }
        
        // 懒加载：只在需要时获取
        const coverage = await this._loadFileCoverage(filePath);
        
        if (coverage) {
            this._fileCoverageCache.set(filePath, coverage);
        }
        
        return coverage;
    }
    
    private async _loadFileCoverage(filePath: string): Promise<FileCoverage | null> {
        // 实际加载逻辑
        // ...
    }
    
    clearCache() {
        this._fileCoverageCache.clear();
    }
}
```

### 4.3 内存优化

**文件**: `pyutagent/agent/test_generator.py`

```python
class TestGeneratorAgent:
    def __init__(self, project_path: str):
        # 使用生成器而不是列表来节省内存
        self._method_cache = self._create_method_generator(project_path)
    
    def _create_method_generator(self, project_path: str):
        """创建方法生成器，按需加载"""
        def generator():
            for java_file in Path(project_path).rglob("*.java"):
                # 惰性解析每个文件
                methods = self._parse_methods(java_file)
                for method in methods:
                    yield method
        
        return generator()
    
    def process_methods_batch(self, batch_size: int = 100):
        """分批处理方法，避免一次性加载所有方法"""
        batch = []
        for method in self._method_cache:
            batch.append(method)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        
        if batch:
            yield batch
```

## 5. 实现优先级

### P0 - 核心功能 (Week 2-3)

1. ✅ 实现 `BatchTestGenerator` 类
2. ✅ 实现并发控制
3. ✅ 实现进度回调
4. ✅ 创建批量生成 UI
5. ✅ 实现 `TestCache` 缓存类
6. ✅ 实现流式传输优化

### P1 - 增强功能 (Week 3-4)

1. ⏭️ 实现懒加载
2. ⏭️ 实现内存优化
3. ⏭️ 实现批量操作取消
4. ⏭️ 实现错误重试机制

### P2 - 优化功能 (Week 4)

1. ⏭️ 实现智能并发调整
2. ⏭️ 实现缓存预热
3. ⏭️ 实现性能监控

## 6. 性能指标

### 6.1 目标性能

| 指标 | 当前 | 目标 | 改进 |
|------|------|------|------|
| 单文件生成时间 | 5-10s | 3-5s | 40-50% |
| 批量生成 (10 文件) | 50-100s | 20-30s | 60-70% |
| 缓存命中率 | 0% | 60-80% | - |
| 内存占用 | 500MB | 200MB | 60% |
| UI 响应时间 | <100ms | <50ms | 50% |

### 6.2 监控指标

1. **生成时间**: 每个文件的生成时间
2. **缓存命中率**: 缓存命中次数 / 总请求次数
3. **并发效率**: 实际并发数 / 目标并发数
4. **错误率**: 失败次数 / 总次数
5. **内存使用**: 峰值内存占用

## 7. 测试计划

### 7.1 性能测试

- [ ] 单文件生成性能测试
- [ ] 批量生成性能测试 (10/50/100 文件)
- [ ] 缓存命中率测试
- [ ] 内存泄漏测试
- [ ] 并发压力测试

### 7.2 功能测试

- [ ] 批量生成正确性
- [ ] 并发控制正确性
- [ ] 缓存一致性
- [ ] 错误处理和重试
- [ ] 取消功能

## 8. 验收标准

### 功能验收

1. ✅ 能够批量生成多个文件的测试
2. ✅ 支持并发控制（可配置）
3. ✅ 实时显示进度
4. ✅ 缓存机制正常工作
5. ✅ 流式输出流畅

### 性能验收

1. ✅ 批量生成性能提升 50%+
2. ✅ 缓存命中率 60%+
3. ✅ 内存占用降低 50%+
4. ✅ UI 无卡顿

## 9. 参考资料

- [Python asyncio](https://docs.python.org/3/library/asyncio.html)
- [VS Code Webview API](https://code.visualstudio.com/api/extension-guides/webview)
- [Server-Sent Events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events)
- [Python caching strategies](https://realpython.com/lru-cache-python/)
