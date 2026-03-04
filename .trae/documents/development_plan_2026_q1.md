# PyUT Agent 核心功能开发计划（2026 Q1-Q2）

## 执行摘要

基于对标分析报告，本计划聚焦**3 大 P0 级核心差距**，制定详细的开发路线图。

**开发目标**：
- 3 个月内补齐 P0 级短板（IDE 集成、Diff 预览、终端集成）
- 达到 Cursor 级别的用户体验
- 保持 Java 测试生成的专业优势

**时间跨度**：2026-03-04 至 2026-06-04（3 个月）

**资源需求**：
- 人力：1-2 名全栈开发者
- 技术栈：TypeScript (VS Code 插件) + Python (后端)

---

## 一、开发优先级

### P0 级任务（1-3 个月）

| 优先级 | 功能模块 | 重要性 | 预计时间 | 依赖关系 |
|--------|---------|--------|----------|----------|
| **P0-1** | VS Code 插件框架 | ⭐⭐⭐⭐⭐ | 2 周 | 无 |
| **P0-2** | Diff 预览组件 | ⭐⭐⭐⭐⭐ | 2 周 | P0-1 |
| **P0-3** | 通用终端集成 | ⭐⭐⭐⭐ | 2 周 | P0-1 |
| **P0-4** | Chat 面板 | ⭐⭐⭐⭐⭐ | 2 周 | P0-1 |
| **P0-5** | 后端通信协议 | ⭐⭐⭐⭐⭐ | 1 周 | 无 |

### P1 级任务（3-6 个月）

| 优先级 | 功能模块 | 重要性 | 预计时间 | 依赖关系 |
|--------|---------|--------|----------|----------|
| **P1-1** | 全局语义索引 | ⭐⭐⭐⭐ | 2 周 | 无 |
| **P1-2** | 智能上下文选择 | ⭐⭐⭐⭐ | 2 周 | P1-1 |
| **P1-3** | 长期规划优化 | ⭐⭐⭐ | 3 周 | 无 |
| **P1-4** | 代码风格学习 | ⭐⭐⭐ | 2 周 | P1-1 |

---

## 二、阶段一：VS Code 插件开发（Week 1-4）

### 目标
开发 VS Code 插件 MVP，实现基础功能

### Week 1: 技术预研和环境搭建

#### Task 1.1: VS Code Extension API 学习
**负责人**: TBD  
**时间**: 3 天  
**交付物**: 学习笔记 + Demo

**具体内容**:
- [ ] 阅读官方文档：https://code.visualstudio.com/api
- [ ] 学习 Extension 结构（package.json, extension.ts）
- [ ] 理解 Activation Events 和 Contribution Points
- [ ] 创建 Hello World 插件

**学习资源**:
- VS Code API 文档
- Cline 开源插件代码（参考）
- Cursor 插件分析

**验收标准**:
- ✅ 能创建并运行简单插件
- ✅ 理解 Command、View、Webview 概念

---

#### Task 1.2: Monaco Editor 研究
**负责人**: TBD  
**时间**: 2 天  
**交付物**: Diff 组件 Demo

**具体内容**:
- [ ] 学习 Monaco Editor API
- [ ] 实现 Diff 编辑器组件
- [ ] 支持行级高亮
- [ ] 支持接受/拒绝变更

**技术要点**:
```typescript
// Monaco Diff Editor 示例
const diffEditor = monaco.editor.createDiffEditor(container, {
    originalEditable: false,
    renderSideBySide: true,
    theme: 'vs-dark'
});

diffEditor.setModel({
    original: monaco.editor.createModel(originalCode, 'java'),
    modified: monaco.editor.createModel(modifiedCode, 'java')
});
```

**验收标准**:
- ✅ 能展示代码差异
- ✅ 支持行号显示
- ✅ 支持语法高亮

---

#### Task 1.3: 项目初始化
**负责人**: TBD  
**时间**: 2 天  
**交付物**: 插件项目框架

**目录结构**:
```
pyutagent-vscode/
├── src/
│   ├── extension.ts          # 入口文件
│   ├── chat/
│   │   ├── chatPanel.ts      # Chat 面板
│   │   └── chatView.tsx      # React 组件
│   ├── diff/
│   │   ├── diffPreview.ts    # Diff 预览
│   │   └── diffEditor.tsx    # Diff 编辑器
│   ├── terminal/
│   │   └── terminalManager.ts # 终端管理
│   └── backend/
│       └── apiClient.ts      # 后端 API 客户端
├── package.json              # 插件配置
├── tsconfig.json             # TypeScript 配置
└── webpack.config.js         # 构建配置
```

**技术栈**:
- TypeScript 4.9+
- React 18 (Webview)
- Webpack 5
- VS Code Extension API 1.80+

**验收标准**:
- ✅ 项目能编译运行
- ✅ 基础目录结构完整
- ✅ 配置正确

---

### Week 2: 基础功能开发

#### Task 2.1: Extension 入口和命令注册
**负责人**: TBD  
**时间**: 2 天  
**交付物**: 可运行的 Extension

**实现内容**:

**package.json 配置**:
```json
{
  "contributes": {
    "commands": [
      {
        "command": "pyutagent.generateTest",
        "title": "Generate Unit Test",
        "category": "PyUT Agent"
      },
      {
        "command": "pyutagent.showChatPanel",
        "title": "Show Chat Panel",
        "category": "PyUT Agent"
      }
    ],
    "menus": {
      "editor/context": [
        {
          "command": "pyutagent.generateTest",
          "when": "resourceLang == java",
          "group": "PyUTAgent"
        }
      ]
    },
    "views": {
      "explorer": [
        {
          "type": "webview",
          "id": "pyutagent.chatView",
          "name": "PyUT Agent Chat",
          "icon": "resources/icon.png"
        }
      ]
    }
  }
}
```

**extension.ts 实现**:
```typescript
import * as vscode from 'vscode';
import { ChatViewProvider } from './chat/chatViewProvider';
import { generateTest } from './commands/generateTest';

export function activate(context: vscode.ExtensionContext) {
    console.log('PyUT Agent is now active!');
    
    // 注册 Chat View Provider
    const chatProvider = new ChatViewProvider(context.extensionUri);
    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider(
            'pyutagent.chatView',
            chatProvider
        )
    );
    
    // 注册命令
    context.subscriptions.push(
        vscode.commands.registerCommand(
            'pyutagent.generateTest',
            generateTest
        )
    );
}
```

**验收标准**:
- ✅ 右键菜单显示"Generate Unit Test"
- ✅ Chat 面板可打开
- ✅ 命令可执行

---

#### Task 2.2: Chat 面板开发
**负责人**: TBD  
**时间**: 3 天  
**交付物**: 可对话的 Chat 面板

**Webview 实现**:

**chatView.tsx**:
```tsx
import React, { useState, useEffect } from 'react';

export const ChatPanel: React.FC = () => {
    const [messages, setMessages] = useState<Message[]>([]);
    const [input, setInput] = useState('');
    
    const sendMessage = async () => {
        // 发送消息到后端
        vscode.postMessage({
            type: 'chat_message',
            content: input
        });
        
        setMessages([...messages, { role: 'user', content: input }]);
        setInput('');
    };
    
    return (
        <div className="chat-container">
            <div className="messages">
                {messages.map((msg, idx) => (
                    <div key={idx} className={`message ${msg.role}`}>
                        {msg.content}
                    </div>
                ))}
            </div>
            <div className="input-area">
                <input
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
                    placeholder="输入指令..."
                />
                <button onClick={sendMessage}>发送</button>
            </div>
        </div>
    );
};
```

**验收标准**:
- ✅ 可发送消息
- ✅ 可接收回复
- ✅ 支持 Markdown 渲染
- ✅ 支持代码块高亮

---

### Week 3: 核心功能开发

#### Task 3.1: 后端通信协议设计
**负责人**: TBD  
**时间**: 2 天  
**交付物**: API 协议文档

**API 设计**:

```typescript
// API 客户端
class PyUTAgentAPI {
    private baseUrl: string;
    
    constructor(baseUrl: string = 'http://localhost:8000') {
        this.baseUrl = baseUrl;
    }
    
    // 生成测试
    async generateTest(filePath: string): Promise<GenerationResult> {
        const response = await fetch(`${this.baseUrl}/api/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                action: 'generate_test',
                file_path: filePath
            })
        });
        return response.json();
    }
    
    // 执行任务
    async executeTask(request: string, context: any): Promise<TaskResult> {
        const response = await fetch(`${this.baseUrl}/api/execute`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                request: request,
                context: context
            })
        });
        return response.json();
    }
    
    // 流式输出
    async streamExecute(request: string): AsyncGenerator<StreamChunk> {
        const response = await fetch(`${this.baseUrl}/api/stream`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ request })
        });
        
        const reader = response.body?.getReader();
        while (true) {
            const { done, value } = await reader!.read();
            if (done) break;
            yield JSON.parse(new TextDecoder().decode(value));
        }
    }
}
```

**验收标准**:
- ✅ API 文档完整
- ✅ TypeScript 类型定义完整
- ✅ 支持流式输出

---

#### Task 3.2: Diff 预览组件
**负责人**: TBD  
**时间**: 3 天  
**交付物**: Diff 预览功能

**diffPreview.ts**:
```typescript
import * as vscode from 'vscode';
import * as monaco from 'monaco-editor';

export class DiffPreviewProvider {
    public async showDiff(
        originalCode: string,
        modifiedCode: string,
        language: string = 'java'
    ): Promise<DiffResult> {
        const panel = vscode.window.createWebviewPanel(
            'pyutagent.diff',
            'Preview Changes',
            vscode.ViewColumn.One,
            {
                enableScripts: true,
                retainContextWhenHidden: true
            }
        );
        
        panel.webview.html = this.getDiffHtml(originalCode, modifiedCode, language);
        
        // 监听用户操作
        return new Promise((resolve) => {
            panel.webview.onDidReceiveMessage((message) => {
                if (message.type === 'accept') {
                    resolve({ action: 'accept' });
                    panel.dispose();
                } else if (message.type === 'reject') {
                    resolve({ action: 'reject' });
                    panel.dispose();
                }
            });
        });
    }
    
    private getDiffHtml(original: string, modified: string, lang: string): string {
        return `
            <!DOCTYPE html>
            <html>
            <head>
                <script src="${monacoUri}"></script>
            </head>
            <body>
                <div id="diff-container" style="width:100%;height:600px;"></div>
                <script>
                    const diffEditor = monaco.editor.createDiffEditor(
                        document.getElementById('diff-container'),
                        { originalEditable: false, renderSideBySide: true }
                    );
                    diffEditor.setModel({
                        original: monaco.editor.createModel(\`${original}\`, '${lang}'),
                        modified: monaco.editor.createModel(\`${modified}\`, '${lang}')
                    });
                </script>
                <button onclick="accept()">Accept</button>
                <button onclick="reject()">Reject</button>
            </body>
            </html>
        `;
    }
}
```

**验收标准**:
- ✅ 可展示代码差异
- ✅ 支持接受/拒绝操作
- ✅ 支持语法高亮

---

### Week 4: 终端集成和测试

#### Task 4.1: 通用终端集成
**负责人**: TBD  
**时间**: 3 天  
**交付物**: 终端面板

**terminalManager.ts**:
```typescript
import * as vscode from 'vscode';
import { spawn } from 'child_process';

export class TerminalManager {
    private terminal: vscode.Terminal | undefined;
    
    public async executeCommand(
        command: string,
        cwd?: string
    ): Promise<CommandResult> {
        return new Promise((resolve) => {
            const shell = process.platform === 'win32' ? 'powershell.exe' : 'bash';
            const proc = spawn(shell, ['-c', command], {
                cwd: cwd || vscode.workspace.rootPath
            });
            
            let stdout = '';
            let stderr = '';
            
            proc.stdout.on('data', (data) => {
                stdout += data.toString();
                this.writeToTerminal(data.toString(), 'stdout');
            });
            
            proc.stderr.on('data', (data) => {
                stderr += data.toString();
                this.writeToTerminal(data.toString(), 'stderr');
            });
            
            proc.on('close', (code) => {
                resolve({
                    success: code === 0,
                    stdout,
                    stderr,
                    exitCode: code
                });
            });
        });
    }
    
    private writeToTerminal(text: string, type: 'stdout' | 'stderr') {
        if (!this.terminal) {
            this.terminal = vscode.window.createTerminal('PyUT Agent');
        }
        this.terminal.show();
        this.terminal.sendText(text, false);
    }
}
```

**验收标准**:
- ✅ 可执行任意命令
- ✅ 实时显示输出
- ✅ 错误高亮显示

---

#### Task 4.2: 集成测试
**负责人**: TBD  
**时间**: 2 天  
**交付物**: 测试报告

**测试用例**:
- [ ] 插件安装和激活
- [ ] Chat 面板对话
- [ ] Diff 预览功能
- [ ] 终端命令执行
- [ ] 右键菜单生成测试

**验收标准**:
- ✅ 所有测试通过
- ✅ 无严重 Bug

---

## 三、阶段二：功能完善（Week 5-8）

### Week 5-6: 审批流程完善

#### Task 5.1: 自主/手动模式切换
**负责人**: TBD  
**时间**: 2 天  

**实现内容**:
- [ ] 模式切换 UI
- [ ] 模式状态管理
- [ ] 根据模式控制审批粒度

**验收标准**:
- ✅ 可切换模式
- ✅ 自主模式自动执行
- ✅ 手动模式每步确认

---

#### Task 5.2: 变更审查界面
**负责人**: TBD  
**时间**: 3 天  

**实现内容**:
- [ ] 变更列表展示
- [ ] 逐个审查功能
- [ ] 批量接受/拒绝

**验收标准**:
- ✅ 可查看所有变更
- ✅ 可选择性接受

---

### Week 7-8: 回滚和历史

#### Task 6.1: Git 回滚集成
**负责人**: TBD  
**时间**: 3 天  

**实现内容**:
- [ ] 一键回滚功能
- [ ] 历史记录查看
- [ ] 版本对比

**验收标准**:
- ✅ 可回滚到任意版本
- ✅ 历史记录完整

---

## 四、阶段三：P1 功能开发（Week 9-12）

### Week 9-10: 语义理解

#### Task 7.1: 全局语义索引
**负责人**: TBD  
**时间**: 1 周  

**技术实现**:
```python
from sentence_transformers import SentenceTransformer
import faiss

class SemanticIndexer:
    def __init__(self, project_path: str):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = faiss.IndexFlatL2(384)
        self.project_path = Path(project_path)
        
    def index_project(self):
        # 扫描所有 Java 文件
        for java_file in self.project_path.rglob('*.java'):
            content = java_file.read_text()
            # 生成 embedding
            embedding = self.model.encode([content])
            # 添加到索引
            self.index.add(embedding)
```

**验收标准**:
- ✅ 可索引整个项目
- ✅ 支持语义搜索

---

#### Task 7.2: 智能上下文选择
**负责人**: TBD  
**时间**: 1 周  

**实现内容**:
- [ ] 相关性评分算法
- [ ] Top-K 选择策略
- [ ] 上下文压缩

**验收标准**:
- ✅ 自动选择最相关代码
- ✅ 上下文大小合理

---

### Week 11-12: 长期规划

#### Task 8.1: 增加子任务上限
**负责人**: TBD  
**时间**: 1 周  

**实现内容**:
- [ ] 修改 max_subtasks (10 -> 100)
- [ ] 分层规划引擎
- [ ] 进度追踪

**验收标准**:
- ✅ 可处理 50+ 步骤任务
- ✅ 进度可视化

---

## 五、交付计划

### Milestone 1 (Week 4): VS Code 插件 MVP
- ✅ 基础插件框架
- ✅ Chat 面板
- ✅ Diff 预览
- ✅ 终端集成

### Milestone 2 (Week 8): 完整审批流程
- ✅ 自主/手动模式
- ✅ 变更审查
- ✅ Git 回滚

### Milestone 3 (Week 12): P1 功能完成
- ✅ 语义索引
- ✅ 智能上下文
- ✅ 长期规划

---

## 六、风险管理

### 技术风险

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| VS Code 插件开发经验不足 | 高 | 高 | 参考 Cline 等开源项目 |
| Monaco Editor 学习曲线 | 中 | 中 | 提前预研，制作 Demo |
| 后端通信性能问题 | 中 | 中 | 使用流式输出，优化协议 |

### 时间风险

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| 3 个月周期不够 | 中 | 高 | 先发布 MVP，快速迭代 |
| 人力资源不足 | 高 | 高 | 招聘或外包部分功能 |

---

## 七、资源需求

### 人力资源
- **TypeScript 开发者**: 1-2 名（VS Code 插件）
- **Python 开发者**: 1 名（后端 API）
- **UI 设计师**: 0.5 名（兼职）

### 技术资源
- **开发工具**: VS Code, Node.js, TypeScript
- **测试环境**: Windows, macOS, Linux
- **部署平台**: VS Code Marketplace

### 预算估算
- **人力成本**: 3 个月 × 2 人 = 6 人月
- **工具和云服务**: $500/月
- **总计**: 取决于人力成本

---

## 八、成功标准

### 技术指标
- ✅ VS Code 插件下载量 > 1000（首月）
- ✅ 用户评分 > 4.5/5
- ✅ Bug 数 < 10（严重 Bug 为 0）
- ✅ 响应时间 < 2 秒

### 用户体验指标
- ✅ Diff 预览使用率 > 80%
- ✅ 终端日均使用 > 50 次
- ✅ 用户留存率 > 60%

---

## 九、下一步行动

### 立即行动（本周）
1. **组建团队** - 招聘 TypeScript 开发者
2. **技术预研** - VS Code Extension API 学习
3. **项目初始化** - 创建插件项目框架

### Week 1 目标
- ✅ 完成技术预研
- ✅ 创建 Hello World 插件
- ✅ 设计 UI 原型

---

**计划制定时间**: 2026-03-04  
**下次评审**: 2026-03-11（Week 1 结束）  
**负责人**: TBD
