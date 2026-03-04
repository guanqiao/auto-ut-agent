# VS Code Extension API 学习笔记

## 一、Extension 基础结构

### 1.1 核心文件

**package.json** - 插件配置文件
```json
{
  "name": "pyutagent-vscode",
  "displayName": "PyUT Agent",
  "description": "AI-driven unit test generation for Java",
  "version": "0.1.0",
  "publisher": "pyutagent",
  "engines": {
    "vscode": "^1.80.0"
  },
  "categories": [
    "Programming Languages",
    "Testing"
  ],
  "activationEvents": [
    "onLanguage:java",
    "onView:pyutagent.chatView"
  ],
  "main": "./dist/extension.js",
  "contributes": {
    "commands": [],
    "views": [],
    "menus": []
  },
  "scripts": {
    "vscode:prepublish": "npm run package",
    "compile": "webpack",
    "watch": "webpack --watch",
    "package": "webpack --mode production --devtool hidden-source-map",
    "test": "vscode-test"
  },
  "devDependencies": {
    "@types/vscode": "^1.80.0",
    "@types/node": "^20.0.0",
    "typescript": "^5.0.0",
    "webpack": "^5.0.0",
    "ts-loader": "^9.0.0"
  },
  "dependencies": {
    "axios": "^1.6.0",
    "react": "^18.0.0",
    "react-dom": "^18.0.0"
  }
}
```

**tsconfig.json** - TypeScript 配置
```json
{
  "compilerOptions": {
    "module": "commonjs",
    "target": "ES2020",
    "lib": ["ES2020"],
    "sourceMap": true,
    "rootDir": "src",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "resolveJsonModule": true,
    "declaration": true,
    "declarationMap": true
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", ".vscode-test"]
}
```

**webpack.config.js** - 构建配置
```javascript
const path = require('path');

module.exports = {
  target: 'node',
  mode: 'none',
  entry: './src/extension.ts',
  output: {
    path: path.resolve(__dirname, 'dist'),
    filename: 'extension.js',
    libraryTarget: 'commonjs2'
  },
  externals: {
    vscode: 'commonjs vscode'
  },
  resolve: {
    extensions: ['.ts', '.js']
  },
  module: {
    rules: [
      {
        test: /\.ts$/,
        exclude: /node_modules/,
        use: 'ts-loader'
      }
    ]
  }
};
```

---

### 1.2 Extension 入口文件

**src/extension.ts** - 主入口
```typescript
import * as vscode from 'vscode';

// 激活函数 - 插件激活时调用
export function activate(context: vscode.ExtensionContext) {
    console.log('PyUT Agent is now active!');
    
    // 注册命令
    const disposable = vscode.commands.registerCommand(
        'pyutagent.helloWorld',
        () => {
            vscode.window.showInformationMessage('Hello from PyUT Agent!');
        }
    );
    
    context.subscriptions.push(disposable);
}

// 停用函数 - 插件停用时调用
export function deactivate() {
    console.log('PyUT Agent is now deactivated.');
}
```

---

## 二、核心概念

### 2.1 Activation Events（激活事件）

插件在特定事件触发时激活：

```json
"activationEvents": [
    "onCommand:pyutagent.generateTest",    // 命令执行时
    "onLanguage:java",                      // Java 文件打开时
    "onView:pyutagent.chatView",           // 视图打开时
    "onStartupFinished"                     // VS Code 启动完成
]
```

### 2.2 Contribution Points（贡献点）

插件可以向 VS Code 贡献各种 UI 元素：

#### Commands（命令）
```json
"contributes": {
    "commands": [
        {
            "command": "pyutagent.generateTest",
            "title": "Generate Unit Test",
            "category": "PyUT Agent",
            "icon": "$(beaker)"
        }
    ]
}
```

#### Menus（菜单）
```json
"contributes": {
    "menus": {
        "editor/context": [
            {
                "command": "pyutagent.generateTest",
                "when": "resourceLang == java",
                "group": "PyUTAgent@1"
            }
        ],
        "explorer/context": [
            {
                "command": "pyutagent.generateTest",
                "when": "resourceLang == java",
                "group": "PyUTAgent"
            }
        ]
    }
}
```

#### Views（视图）
```json
"contributes": {
    "views": {
        "explorer": [
            {
                "type": "webview",
                "id": "pyutagent.chatView",
                "name": "PyUT Agent Chat",
                "icon": "resources/icon.png",
                "contextualTitle": "PyUT Agent",
                "visibility": "visible"
            }
        ]
    }
}
```

---

## 三、Webview 开发

### 3.1 WebviewViewProvider（侧边栏视图）

**src/chat/chatViewProvider.ts**
```typescript
import * as vscode from 'vscode';

export class ChatViewProvider implements vscode.WebviewViewProvider {
    public static readonly viewType = 'pyutagent.chatView';
    
    private _view?: vscode.WebviewView;
    
    constructor(private readonly _extensionUri: vscode.Uri) {}
    
    public resolveWebviewView(
        webviewView: vscode.WebviewView,
        context: vscode.WebviewViewResolveContext,
        _token: vscode.CancellationToken
    ) {
        this._view = webviewView;
        
        // 配置 Webview
        webviewView.webview.options = {
            enableScripts: true,
            localResourceRoots: [this._extensionUri]
        };
        
        // 设置 HTML 内容
        webviewView.webview.html = this._getHtmlForWebview(webviewView);
        
        // 监听消息
        webviewView.webview.onDidReceiveMessage((message) => {
            this._handleMessage(message);
        });
    }
    
    private _getHtmlForWebview(webview: vscode.Webview): string {
        return `
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>PyUT Agent Chat</title>
                <style>
                    body { padding: 10px; font-family: var(--vscode-font-family); }
                    .message { margin: 10px 0; padding: 8px; border-radius: 4px; }
                    .user { background: var(--vscode-button-background); color: white; }
                    .agent { background: var(--vscode-editor-background); }
                    .input-area { display: flex; gap: 5px; margin-top: 10px; }
                    input { flex: 1; padding: 5px; }
                    button { padding: 5px 15px; }
                </style>
            </head>
            <body>
                <div id="messages"></div>
                <div class="input-area">
                    <input id="input" placeholder="Type a message..." />
                    <button id="send">Send</button>
                </div>
                <script>
                    const vscode = acquireVsCodeApi();
                    
                    document.getElementById('send').onclick = () => {
                        const input = document.getElementById('input');
                        const message = input.value;
                        vscode.postMessage({ type: 'message', content: message });
                        input.value = '';
                    };
                    
                    window.addEventListener('message', event => {
                        const message = event.data;
                        const messagesDiv = document.getElementById('messages');
                        const msgDiv = document.createElement('div');
                        msgDiv.className = 'message ' + message.role;
                        msgDiv.textContent = message.content;
                        messagesDiv.appendChild(msgDiv);
                    });
                </script>
            </body>
            </html>
        `;
    }
    
    private _handleMessage(message: any) {
        switch (message.type) {
            case 'message':
                // 处理用户消息
                this._sendMessage({
                    role: 'agent',
                    content: 'Received: ' + message.content
                });
                break;
        }
    }
    
    public sendMessage(message: any) {
        this._view?.webview.postMessage(message);
    }
}
```

---

### 3.2 WebviewPanel（独立面板）

**src/diff/diffPanel.ts**
```typescript
import * as vscode from 'vscode';

export class DiffPanel {
    public static createOrShow(extensionUri: vscode.Uri) {
        const column = vscode.window.activeTextEditor
            ? vscode.window.activeTextEditor.viewColumn
            : undefined;
        
        const panel = vscode.window.createWebviewPanel(
            'pyutagent.diff',
            'Preview Changes',
            column || vscode.ViewColumn.One,
            {
                enableScripts: true,
                retainContextWhenHidden: true,
                localResourceRoots: [extensionUri]
            }
        );
        
        panel.webview.html = DiffPanel._getHtmlForWebview(panel, extensionUri);
        
        // 监听消息
        panel.webview.onDidReceiveMessage(
            (message) => {
                switch (message.type) {
                    case 'accept':
                        // 接受变更
                        vscode.window.showInformationMessage('Changes accepted');
                        break;
                    case 'reject':
                        // 拒绝变更
                        vscode.window.showInformationMessage('Changes rejected');
                        break;
                }
            },
            undefined,
            []
        );
    }
    
    private static _getHtmlForWebview(
        panel: vscode.Webview,
        extensionUri: vscode.Uri
    ): string {
        return `
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>Diff Preview</title>
                <style>
                    body { margin: 0; padding: 10px; }
                    #diff-container { height: 600px; }
                    .buttons { margin-top: 10px; text-align: center; }
                    button { padding: 10px 20px; margin: 0 10px; }
                </style>
            </head>
            <body>
                <div id="diff-container"></div>
                <div class="buttons">
                    <button id="accept" style="background: green; color: white;">Accept</button>
                    <button id="reject" style="background: red; color: white;">Reject</button>
                </div>
                <script>
                    const vscode = acquireVsCodeApi();
                    
                    document.getElementById('accept').onclick = () => {
                        vscode.postMessage({ type: 'accept' });
                    };
                    
                    document.getElementById('reject').onclick = () => {
                        vscode.postMessage({ type: 'reject' });
                    };
                </script>
            </body>
            </html>
        `;
    }
}
```

---

## 四、命令实现

### 4.1 基础命令

**src/commands/generateTest.ts**
```typescript
import * as vscode from 'vscode';

export async function generateTest() {
    const editor = vscode.window.activeTextEditor;
    
    if (!editor) {
        vscode.window.showErrorMessage('No active editor');
        return;
    }
    
    const document = editor.document;
    const filePath = document.fileName;
    
    vscode.window.showInformationMessage(
        `Generating test for: ${filePath}`
    );
    
    // TODO: 调用后端 API 生成测试
}
```

---

## 五、终端集成

### 5.1 Terminal 使用

```typescript
import * as vscode from 'vscode';

export class TerminalManager {
    private terminal: vscode.Terminal | undefined;
    
    public executeCommand(command: string, cwd?: string) {
        if (!this.terminal) {
            this.terminal = vscode.window.createTerminal('PyUT Agent');
        }
        
        this.terminal.show();
        
        if (cwd) {
            this.terminal.sendText(`cd ${cwd}`, false);
        }
        
        this.terminal.sendText(command, false);
    }
    
    public dispose() {
        this.terminal?.dispose();
    }
}
```

---

## 六、配置管理

### 6.1 package.json 配置

```json
"contributes": {
    "configuration": {
        "title": "PyUT Agent",
        "properties": {
            "pyutagent.apiUrl": {
                "type": "string",
                "default": "http://localhost:8000",
                "description": "PyUT Agent API URL"
            },
            "pyutagent.autoApprove": {
                "type": "boolean",
                "default": false,
                "description": "Automatically approve changes"
            }
        }
    }
}
```

### 6.2 读取配置

```typescript
const config = vscode.workspace.getConfiguration('pyutagent');
const apiUrl = config.get<string>('apiUrl');
const autoApprove = config.get<boolean>('autoApprove');
```

---

## 七、调试和测试

### 7.1 launch.json 配置

**.vscode/launch.json**
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
            "outFiles": [
                "${workspaceFolder}/dist/**/*.js"
            ],
            "preLaunchTask": "${defaultBuildTask}"
        },
        {
            "name": "Extension Tests",
            "type": "extensionHost",
            "request": "launch",
            "args": [
                "--extensionDevelopmentPath=${workspaceFolder}",
                "--extensionTestsPath=${workspaceFolder}/dist/test/suite/index"
            ],
            "outFiles": [
                "${workspaceFolder}/dist/test/**/*.js"
            ]
        }
    ]
}
```

---

## 八、发布流程

### 8.1 打包

```bash
# 安装 vsce
npm install -g @vscode/vsce

# 打包
vsce package

# 发布
vsce publish
```

### 8.2 Marketplace 配置

**README.md** - 插件描述
**CHANGELOG.md** - 变更日志
**LICENSE** - 许可证

---

## 九、最佳实践

### 9.1 性能优化

1. **懒加载**: 按需加载模块
2. **缓存**: 缓存 API 响应
3. **异步**: 使用 async/await
4. **Progress**: 显示进度条

### 9.2 用户体验

1. **状态栏**: 显示当前状态
2. **通知**: 及时通知用户
3. **错误处理**: 友好的错误消息
4. **快捷键**: 提供快捷键

### 9.3 安全性

1. **输入验证**: 验证用户输入
2. **HTTPS**: 使用 HTTPS 连接 API
3. **敏感信息**: 使用 SecretStorage 存储密钥

---

## 十、参考资源

- **官方文档**: https://code.visualstudio.com/api
- **示例项目**: https://github.com/microsoft/vscode-extension-samples
- **Cline 源码**: https://github.com/cline/cline
- **API 参考**: https://code.visualstudio.com/api/references/vscode-api

---

**学习时间**: 2026-03-04  
**完成状态**: ✅ 已完成
