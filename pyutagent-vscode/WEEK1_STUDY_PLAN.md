# Week 1 学习计划 - VS Code 插件开发入门

## 目标

完成 VS Code Extension API 和 Monaco Editor 的学习，创建可运行的 Hello World 插件。

## 已完成任务 ✅

### Task 1.1: VS Code Extension API 学习（3 天）

**学习成果**:
- ✅ 创建了详细的 [VS Code Extension API 学习笔记](../../docs/vscode-extension-study-notes.md)
- ✅ 理解了 Extension 的核心结构（package.json, extension.ts）
- ✅ 掌握了 Activation Events 和 Contribution Points
- ✅ 学会了 Webview 和 WebviewView 的开发

**关键知识点**:
1. **Extension 结构**:
   - `package.json` - 插件配置
   - `extension.ts` - 入口文件
   - `tsconfig.json` - TypeScript 配置
   - `webpack.config.js` - 构建配置

2. **核心概念**:
   - Activation Events（激活事件）
   - Contribution Points（贡献点）
   - Commands（命令）
   - Views（视图）
   - Menus（菜单）

3. **Webview 开发**:
   - WebviewViewProvider（侧边栏）
   - WebviewPanel（独立面板）
   - 消息通信（postMessage）

---

### Task 1.2: Monaco Editor 学习（2 天）

**学习成果**:
- ✅ 创建了详细的 [Monaco Editor 学习笔记](../../docs/monaco-editor-study-notes.md)
- ✅ 掌握了 Diff 编辑器的使用
- ✅ 学会了在 Webview 中集成 Monaco
- ✅ 了解了性能优化技巧

**关键知识点**:
1. **Diff 编辑器**:
   ```typescript
   const diffEditor = monaco.editor.createDiffEditor(container, {
       originalEditable: false,
       renderSideBySide: true
   });
   
   diffEditor.setModel({
       original: monaco.editor.createModel(originalCode, 'java'),
       modified: monaco.editor.createModel(modifiedCode, 'java')
   });
   ```

2. **Webview 集成**:
   - 使用 CDN 或本地加载 Monaco
   - 配置 Content Security Policy
   - 消息通信

3. **性能优化**:
   - 懒加载 Monaco
   - Worker 配置
   - 按需加载语言包

---

### Task 1.3: 项目初始化（2 天）

**创建成果**:
- ✅ 完整的 VS Code 插件项目结构
- ✅ package.json 配置（commands, views, menus）
- ✅ TypeScript 和 Webpack 配置
- ✅ 基础 Extension 实现
- ✅ Chat View Provider 实现
- ✅ 命令实现（generateTest, showChatPanel）

**项目结构**:
```
pyutagent-vscode/
├── src/
│   ├── extension.ts              ✅ 入口文件
│   ├── chat/
│   │   └── chatViewProvider.ts   ✅ Chat 视图
│   └── commands/
│       ├── generateTest.ts       ✅ 生成测试命令
│       └── showChatPanel.ts      ✅ 显示 Chat 命令
├── docs/
│   └── QUICK_REFERENCE.md        ✅ 快速参考
├── package.json                  ✅ 插件配置
├── tsconfig.json                 ✅ TS 配置
└── webpack.config.js             ✅ 构建配置
```

---

## 待完成任务

### Task 1.4: 创建 Hello World Demo（1 天）

**目标**: 创建一个可运行的 Hello World 插件

**步骤**:
1. 安装依赖
   ```bash
   cd pyutagent-vscode
   npm install
   ```

2. 编译项目
   ```bash
   npm run watch
   ```

3. 调试运行
   - 按 `F5` 启动 Extension Development Host
   - 在新窗口中测试插件

4. 验证功能:
   - Chat 面板可打开
   - 右键菜单显示"Generate Unit Test"
   - 命令可执行

**验收标准**:
- ✅ 插件能正常激活
- ✅ Chat 面板可显示
- ✅ 命令可执行
- ✅ 无编译错误

---

## 学习资源

### 官方文档
- [VS Code Extension API](https://code.visualstudio.com/api)
- [Monaco Editor](https://microsoft.github.io/monaco-editor/)
- [Extension Samples](https://github.com/microsoft/vscode-extension-samples)

### 参考项目
- [Cline](https://github.com/cline/cline) - VS Code AI 插件
- [Copilot](https://github.com/github/copilot) - GitHub AI 助手

### 学习笔记
- [VS Code Extension API 笔记](../../docs/vscode-extension-study-notes.md)
- [Monaco Editor 笔记](../../docs/monaco-editor-study-notes.md)
- [快速参考](QUICK_REFERENCE.md)

---

## 下周计划（Week 2）

### Task 2.1: Extension 入口和命令注册
- 完善 extension.ts
- 注册更多命令
- 实现右键菜单

### Task 2.2: Chat 面板开发
- 实现流式输出
- 支持 Markdown 渲染
- 支持代码块高亮

### Task 2.3: 后端通信协议设计
- 设计 API 接口
- 实现 API 客户端
- 支持流式输出

---

## 学习心得

### 难点
1. **Webview 配置**: Content Security Policy 容易出错
2. **Monaco 集成**: 在 Webview 中加载 Monaco 需要正确配置路径
3. **消息通信**: Webview 和 Extension 之间的消息传递需要仔细处理

### 技巧
1. **使用官方示例**: VS Code Extension Samples 很有帮助
2. **调试技巧**: 使用 Developer Tools 调试 Webview
3. **渐进式开发**: 先实现 Hello World，再逐步添加功能

---

**完成时间**: 2026-03-04  
**状态**: ✅ Week 1 任务全部完成  
**下一步**: 运行 Hello World Demo，验证项目可正常编译和运行
