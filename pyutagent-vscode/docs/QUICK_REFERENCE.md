# VS Code Extension API 快速参考

## 核心 API

### 窗口和 UI

```typescript
// 显示信息消息
vscode.window.showInformationMessage('Hello World');

// 显示警告消息
vscode.window.showWarningMessage('Warning!');

// 显示错误消息
vscode.window.showErrorMessage('Error occurred');

// 创建终端
const terminal = vscode.window.createTerminal('My Terminal');
terminal.sendText('npm install');
terminal.show();

// 创建状态栏
const statusBarItem = vscode.window.createStatusBarItem();
statusBarItem.text = 'PyUT Agent';
statusBarItem.show();

// 显示进度条
vscode.window.withProgress({
    location: vscode.ProgressLocation.Notification,
    title: 'Generating tests...',
    cancellable: true
}, async (progress, token) => {
    progress.report({ increment: 0 });
    // 执行任务
    progress.report({ increment: 100 });
});
```

### 工作空间和文件

```typescript
// 获取当前文档
const editor = vscode.window.activeTextEditor;
const document = editor?.document;
const text = document?.getText();

// 获取工作区根目录
const rootPath = vscode.workspace.rootPath;

// 读取配置
const config = vscode.workspace.getConfiguration('pyutagent');
const apiUrl = config.get<string>('apiUrl');

// 监听文件保存
vscode.workspace.onDidSaveTextDocument((doc) => {
    console.log('File saved:', doc.fileName);
});
```

### 命令注册

```typescript
// 注册命令
context.subscriptions.push(
    vscode.commands.registerCommand('myapp.hello', () => {
        vscode.window.showInformationMessage('Hello!');
    })
);

// 执行命令
vscode.commands.executeCommand('myapp.hello');
```

### Webview

```typescript
// 创建 Webview 面板
const panel = vscode.window.createWebviewPanel(
    'myapp.webview',
    'My Webview',
    vscode.ViewColumn.One,
    {
        enableScripts: true,
        retainContextWhenHidden: true
    }
);

panel.webview.html = '<html><body><h1>Hello</h1></body></html>';

// 监听消息
panel.webview.onDidReceiveMessage(message => {
    console.log('Received:', message);
});

// 发送消息
panel.webview.postMessage({ type: 'greeting', content: 'Hi!' });
```

### WebviewView（侧边栏）

```typescript
class MyViewProvider implements vscode.WebviewViewProvider {
    resolveWebviewView(webviewView: vscode.WebviewView) {
        webviewView.webview.html = '<html><body>View</body></html>';
        
        webviewView.webview.onDidReceiveMessage(message => {
            // 处理消息
        });
    }
}

// 注册
context.subscriptions.push(
    vscode.window.registerWebviewViewProvider('myapp.view', new MyViewProvider())
);
```

## 常用类型

### Range - 文本范围

```typescript
const range = new vscode.Range(
    0, 0,  // 起始行、列
    10, 20 // 结束行、列
);
```

### Position - 文本位置

```typescript
const position = new vscode.Position(5, 10); // 第 5 行，第 10 列
```

### Uri - 资源路径

```typescript
const uri = vscode.Uri.file('/path/to/file');
const doc = await vscode.workspace.openTextDocument(uri);
```

### CompletionItem - 代码补全

```typescript
const item = new vscode.CompletionItem('JUnit5', vscode.CompletionItemKind.Snippet);
item.insertText = new vscode.SnippetString('@Test\nvoid test() {\n    $0\n}');
```

### Diagnostic - 代码诊断

```typescript
const diagnostic = new vscode.Diagnostic(
    new vscode.Range(0, 0, 0, 10),
    'Error message',
    vscode.DiagnosticSeverity.Error
);
```

## 事件监听

```typescript
// 文档变化
vscode.workspace.onDidChangeTextDocument(event => {
    console.log('Document changed:', event.document.fileName);
});

// 文档打开
vscode.workspace.onDidOpenTextDocument(doc => {
    console.log('Document opened:', doc.fileName);
});

// 活动编辑器变化
vscode.window.onDidChangeActiveTextEditor(editor => {
    console.log('Active editor changed');
});

// 配置变化
vscode.workspace.onDidChangeConfiguration(event => {
    if (event.affectsConfiguration('pyutagent.apiUrl')) {
        console.log('API URL changed');
    }
});
```

## 最佳实践

### 1. 资源清理

```typescript
export function activate(context: vscode.ExtensionContext) {
    const disposable = vscode.commands.registerCommand('myapp.hello', () => {});
    
    // 添加到订阅列表，停用时自动清理
    context.subscriptions.push(disposable);
}
```

### 2. 异步操作

```typescript
async function longRunningTask() {
    await vscode.window.withProgress({
        location: vscode.ProgressLocation.Notification,
        title: 'Processing...',
        cancellable: true
    }, async (progress, token) => {
        // 异步任务
        await someAsyncOperation();
    });
}
```

### 3. 错误处理

```typescript
try {
    await someOperation();
} catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    vscode.window.showErrorMessage(`Operation failed: ${message}`);
}
```

### 4. 日志记录

```typescript
const outputChannel = vscode.window.createOutputChannel('PyUT Agent');
outputChannel.appendLine('Starting operation...');
outputChannel.show();
```

## 调试技巧

### launch.json

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Run Extension",
            "type": "extensionHost",
            "request": "launch",
            "args": [
                "--extensionDevelopmentPath=${workspaceFolder}"
            ],
            "outFiles": ["${workspaceFolder}/dist/**/*.js"],
            "preLaunchTask": "${defaultBuildTask}"
        }
    ]
}
```

### tasks.json

```json
{
    "version": "2.0.0",
    "tasks": [
        {
            "type": "npm",
            "script": "watch",
            "problemMatcher": "$tsc-watch",
            "isBackground": true,
            "presentation": {
                "reveal": "never"
            },
            "group": {
                "kind": "build",
                "isDefault": true
            }
        }
    ]
}
```

## 参考资源

- [VS Code API 文档](https://code.visualstudio.com/api/references/vscode-api)
- [Extension 指南](https://code.visualstudio.com/api/extension-guides/overview)
- [示例项目](https://github.com/microsoft/vscode-extension-samples)
