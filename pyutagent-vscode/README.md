# PyUT Agent VS Code 插件项目

AI-driven unit test generation for Java developers.

## 功能特性

- 🤖 **智能测试生成**: 基于 AI 的 Java 单元测试生成
- 💬 **Chat 面板**: 自然语言交互
- 👀 **Diff 预览**: 可视化代码变更预览
- ⌨️ **终端集成**: 内置终端执行命令
- ✅ **审批流程**: 自主/手动模式切换

## 快速开始

### 安装依赖

```bash
npm install
```

### 开发模式

```bash
npm run watch
```

按 `F5` 启动 Extension Development Host

### 打包发布

```bash
npm run package
vsce package
vsce publish
```

## 项目结构

```
pyutagent-vscode/
├── src/
│   ├── extension.ts          # 入口文件
│   ├── chat/
│   │   ├── chatViewProvider.ts
│   │   └── chatPanel.tsx
│   ├── diff/
│   │   ├── diffProvider.ts
│   │   └── diffPanel.tsx
│   ├── terminal/
│   │   └── terminalManager.ts
│   ├── commands/
│   │   ├── generateTest.ts
│   │   └── executeCommand.ts
│   └── backend/
│       └── apiClient.ts
├── webviews/
│   ├── diff-view.html
│   └── chat-view.html
├── resources/
│   └── icon.png
├── package.json
├── tsconfig.json
└── webpack.config.js
```

## 开发指南

参考文档：
- [VS Code Extension API 笔记](docs/vscode-extension-study-notes.md)
- [Monaco Editor 笔记](docs/monaco-editor-study-notes.md)

## License

MIT
