# Monaco Editor 学习笔记

## 一、Monaco Editor 基础

### 1.1 简介

Monaco Editor 是 VS Code 使用的代码编辑器组件，支持：
- 语法高亮
- 智能提示
- Diff 对比
- 多语言支持
- 主题切换

### 1.2 安装

```bash
npm install monaco-editor
```

### 1.3 基础使用

**创建编辑器**:
```typescript
import * as monaco from 'monaco-editor';

// 创建编辑器实例
const editor = monaco.editor.create(
    document.getElementById('container'),
    {
        value: 'console.log("Hello, world!");',
        language: 'javascript',
        theme: 'vs-dark',
        automaticLayout: true,
        minimap: { enabled: true },
        fontSize: 14,
        lineNumbers: 'on',
        renderWhitespace: 'selection',
        wordWrap: 'on'
    }
);

// 获取内容
const content = editor.getValue();

// 设置内容
editor.setValue('New content');

// 监听变化
editor.onDidChangeModelContent((e) => {
    console.log('Content changed', e);
});
```

---

## 二、Diff 编辑器

### 2.1 创建 Diff 编辑器

```typescript
import * as monaco from 'monaco-editor';

// 创建 Diff 编辑器
const diffEditor = monaco.editor.createDiffEditor(
    document.getElementById('diff-container'),
    {
        originalEditable: false,  // 原始内容不可编辑
        renderSideBySide: true,   // 并排显示
        theme: 'vs-dark',
        automaticLayout: true,
        diffWordWrap: 'off'
    }
);

// 创建模型
const originalModel = monaco.editor.createModel(
    originalCode,
    'java'  // 语言
);

const modifiedModel = monaco.editor.createModel(
    modifiedCode,
    'java'
);

// 设置模型
diffEditor.setModel({
    original: originalModel,
    modified: modifiedModel
});

// 获取变更
const lineChanges = diffEditor.getLineChanges();
console.log('Line changes:', lineChanges);
```

### 2.2 Diff 编辑器配置

```typescript
const diffEditorOptions: monaco.editor.IDiffEditorOptions = {
    originalEditable: false,
    renderSideBySide: true,
    theme: 'vs-dark',
    automaticLayout: true,
    
    // Diff 相关
    diffWordWrap: 'off',
    diffAlgorithm: 'legacy',
    enableSplitViewResizing: true,
    
    // 滚动
    scrollBeyondLastLine: false,
    minimap: { enabled: false },
    
    // 行号
    lineNumbers: 'on',
    renderLineHighlight: 'all',
    
    // 只读
    readOnly: true,
    
    // 其他
    fontSize: 14,
    fontFamily: 'Consolas, "Courier New", monospace',
    wordWrap: 'off'
};
```

### 2.3 内联 Diff 模式

```typescript
// 切换内联模式
diffEditor.updateOptions({
    renderSideBySide: false  // false = 内联模式
});
```

---

## 三、Webview 中集成 Monaco

### 3.1 HTML 模板

**diff-view.html**:
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Diff Preview</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            padding: 10px;
            background: #1e1e1e;
            color: #d4d4d4;
        }
        #header {
            padding: 10px 0;
            border-bottom: 1px solid #333;
            margin-bottom: 10px;
        }
        #header h2 {
            font-size: 16px;
            font-weight: normal;
        }
        #diff-container {
            height: 600px;
            border: 1px solid #333;
        }
        #buttons {
            margin-top: 15px;
            text-align: center;
        }
        button {
            padding: 10px 25px;
            margin: 0 10px;
            border: none;
            border-radius: 3px;
            cursor: pointer;
            font-size: 14px;
        }
        #accept {
            background: #4CAF50;
            color: white;
        }
        #reject {
            background: #f44336;
            color: white;
        }
        button:hover { opacity: 0.9; }
    </style>
</head>
<body>
    <div id="header">
        <h2>Preview Changes</h2>
    </div>
    <div id="diff-container"></div>
    <div id="buttons">
        <button id="accept">✓ Accept</button>
        <button id="reject">✗ Reject</button>
    </div>
    
    <script>
        // 从 VS Code 获取 Monaco
        const vscode = acquireVsCodeApi();
        
        // 等待 Monaco 加载
        window.addEventListener('load', () => {
            initDiffEditor();
        });
        
        function initDiffEditor() {
            // 创建 Diff 编辑器
            const diffEditor = monaco.editor.createDiffEditor(
                document.getElementById('diff-container'),
                {
                    originalEditable: false,
                    renderSideBySide: true,
                    theme: 'vs-dark',
                    automaticLayout: true,
                    minimap: { enabled: false },
                    fontSize: 14,
                    lineNumbers: 'on',
                    wordWrap: 'off'
                }
            );
            
            // 从消息中获取代码
            window.addEventListener('message', (event) => {
                const data = event.data;
                if (data.type === 'setCode') {
                    const originalModel = monaco.editor.createModel(
                        data.originalCode,
                        data.language
                    );
                    const modifiedModel = monaco.editor.createModel(
                        data.modifiedCode,
                        data.language
                    );
                    
                    diffEditor.setModel({
                        original: originalModel,
                        modified: modifiedModel
                    });
                }
            });
            
            // 按钮事件
            document.getElementById('accept').onclick = () => {
                vscode.postMessage({ type: 'accept' });
            };
            
            document.getElementById('reject').onclick = () => {
                vscode.postMessage({ type: 'reject' });
            };
        }
    </script>
</body>
</html>
```

### 3.2 VS Code 扩展中使用

**diffProvider.ts**:
```typescript
import * as vscode from 'vscode';
import { join } from 'path';

export class DiffViewProvider {
    private panel: vscode.WebviewPanel | undefined;
    
    constructor(private extensionUri: vscode.Uri) {}
    
    public async showDiff(
        originalCode: string,
        modifiedCode: string,
        language: string = 'java'
    ): Promise<'accept' | 'reject' | undefined> {
        // 创建面板
        this.panel = vscode.window.createWebviewPanel(
            'pyutagent.diff',
            'Preview Changes',
            vscode.ViewColumn.One,
            {
                enableScripts: true,
                retainContextWhenHidden: true,
                localResourceRoots: [this.extensionUri]
            }
        );
        
        // 设置 HTML
        this.panel.webview.html = this.getHtmlContent();
        
        // 发送代码
        this.panel.webview.postMessage({
            type: 'setCode',
            originalCode,
            modifiedCode,
            language
        });
        
        // 监听用户操作
        return new Promise((resolve) => {
            const disposable = this.panel!.webview.onDidReceiveMessage(
                (message) => {
                    switch (message.type) {
                        case 'accept':
                            resolve('accept');
                            disposable.dispose();
                            break;
                        case 'reject':
                            resolve('reject');
                            disposable.dispose();
                            break;
                    }
                }
            );
            
            // 面板关闭
            this.panel.onDidDispose(() => {
                resolve(undefined);
            });
        });
    }
    
    private getHtmlContent(): string {
        const htmlPath = join(this.extensionUri.fsPath, 'webviews', 'diff-view.html');
        // 读取 HTML 文件并返回
        return require('fs').readFileSync(htmlPath, 'utf-8');
    }
}
```

---

## 四、高级功能

### 4.1 自定义主题

```typescript
monaco.editor.defineTheme('my-theme', {
    base: 'vs-dark',
    inherit: true,
    rules: [
        { token: 'comment', foreground: '6A9955', fontStyle: 'italic' },
        { token: 'keyword', foreground: '569CD6' },
        { token: 'string', foreground: 'CE9178' },
        { token: 'function', foreground: 'DCDCAA' },
        { token: 'class', foreground: '4EC9B0' }
    ],
    colors: {
        'editor.background': '#1e1e1e',
        'editor.foreground': '#d4d4d4',
        'editor.lineHighlightBackground': '#2d2d2d',
        'editorCursor.foreground': '#aeafad'
    }
});

monaco.editor.setTheme('my-theme');
```

### 4.2 代码补全

```typescript
monaco.languages.registerCompletionItemProvider('java', {
    provideCompletionItems: (model, position) => {
        const word = model.getWordUntilPosition(position);
        const range = {
            startLineNumber: position.lineNumber,
            endLineNumber: position.lineNumber,
            startColumn: word.startColumn,
            endColumn: word.endColumn
        };
        
        return {
            suggestions: [
                {
                    label: 'JUnit5',
                    kind: monaco.languages.CompletionItemKind.Snippet,
                    insertText: '@Test\nvoid testMethod() {\n    $0\n}',
                    insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
                    range: range
                }
            ]
        };
    }
});
```

### 4.3 代码诊断

```typescript
// 添加诊断信息
monaco.editor.setModelMarkers(model, 'owner', [
    {
        startLineNumber: 10,
        startColumn: 1,
        endLineNumber: 10,
        endColumn: 20,
        message: 'This is a warning',
        severity: monaco.MarkerSeverity.Warning
    }
]);
```

---

## 五、性能优化

### 5.1 懒加载 Monaco

```typescript
// 按需加载
async function loadMonaco() {
    const monaco = await import('monaco-editor');
    return monaco;
}

// 使用时加载
loadMonaco().then((monaco) => {
    monaco.editor.create(/* ... */);
});
```

### 5.2 Worker 配置

```javascript
// webpack.config.js
module.exports = {
    // ...
    module: {
        rules: [
            {
                test: /monaco-editor.*worker.*\.js$/,
                use: 'worker-loader'
            }
        ]
    }
};
```

---

## 六、常见问题

### Q1: Monaco 在 Webview 中不显示？

**A**: 确保配置了正确的 Content Security Policy：

```typescript
webview.html = `
    <meta http-equiv="Content-Security-Policy" 
          content="default-src 'none'; 
                   style-src ${webview.cspSource} 'unsafe-inline'; 
                   script-src ${webview.cspSource} 'unsafe-inline';">
`;
```

### Q2: 如何获取所有变更？

**A**: 使用 `getLineChanges()`:

```typescript
const changes = diffEditor.getLineChanges();
changes.forEach(change => {
    console.log('Modified lines:', change.originalStart, '-', change.originalEnd);
    console.log('New lines:', change.modifiedStart, '-', change.modifiedEnd);
});
```

### Q3: 如何高亮特定行？

**A**: 使用 Decorations:

```typescript
const decorations = editor.createDecorationsCollection([
    {
        range: new monaco.Range(10, 1, 10, 1),
        options: {
            isWholeLine: true,
            backgroundColor: { dark: '#ff000040', light: '#ff000040' },
            hoverMessage: { value: 'This line was modified' }
        }
    }
]);
```

---

## 七、参考资源

- **官方文档**: https://microsoft.github.io/monaco-editor/
- **API 参考**: https://microsoft.github.io/monaco-editor/api/index.html
- **Playground**: https://microsoft.github.io/monaco-editor/playground.html
- **GitHub**: https://github.com/microsoft/monaco-editor

---

**学习时间**: 2026-03-04  
**完成状态**: ✅ 已完成
