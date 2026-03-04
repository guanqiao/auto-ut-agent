# PyUT Agent - AI 驱动的 Java 单元测试生成器

[![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)](https://marketplace.visualstudio.com/items?itemName=pyutagent.pyutagent-vscode)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**PyUT Agent** 是一款强大的 VS Code 插件，专为 Java 开发者提供 AI 驱动的单元测试生成服务。通过集成先进的 AI 模型，它能够理解您的代码并自动生成高质量的 JUnit 测试用例。

![PyUT Agent Demo](resources/demo.gif)

## ✨ 核心功能

### 🤖 智能测试生成
- **AI 驱动**: 基于先进的 AI 模型理解代码逻辑
- **高质量测试**: 生成符合最佳实践的 JUnit 测试
- **上下文感知**: 考虑项目结构和依赖关系
- **实时预览**: Diff 视图预览，支持接受/拒绝

### 💬 智能对话助手
- **自然语言交互**: 像与同事交流一样简单
- **Markdown 支持**: 完美的格式渲染
- **流式输出**: 实时查看 AI 思考过程
- **代码高亮**: 语法高亮显示

### 🎯 完整开发流程
- **一键生成**: 右键菜单快速生成测试
- **自动创建**: 自动创建测试文件结构
- **运行测试**: 集成终端运行测试
- **结果反馈**: 实时查看测试结果

### ⚙️ 灵活配置
- **多种模式**: 自主/交互/监督模式可选
- **API 配置**: 自定义后端 API 地址
- **超时控制**: 灵活的任务超时设置
- **重试机制**: 自动重试失败任务

## 🚀 快速开始

### 安装

1. 打开 VS Code
2. 按 `Ctrl+P` (Windows/Linux) 或 `Cmd+P` (Mac)
3. 输入 `ext install pyutagent-vscode`
4. 点击安装

### 配置

1. 打开命令面板 (`Ctrl+Shift+P` / `Cmd+Shift+P`)
2. 输入 `PyUT Agent: Open Configuration`
3. 配置后端 API URL（默认：`http://localhost:8000`）
4. 选择运行模式：
   - **Autonomous**: 自动执行，无需确认
   - **Interactive**: 关键步骤确认（推荐）
   - **Supervised**: 每步确认

### 使用

#### 方法 1: 右键菜单
1. 打开 Java 文件
2. 右键点击编辑器
3. 选择 `Generate Unit Test`
4. 预览生成的测试
5. 接受或拒绝变更

#### 方法 2: Chat 面板
1. 打开 Chat 面板（侧边栏 PyUT Agent 图标）
2. 输入您的需求，例如：
   ```
   为当前文件生成测试，覆盖所有边界条件
   ```
3. AI 会理解并生成测试

#### 方法 3: 命令面板
1. 按 `Ctrl+Shift+P` / `Cmd+Shift+P`
2. 输入 `PyUT Agent: Generate Unit Test`
3. 按照提示操作

## 📖 使用示例

### 示例 1: 生成单元测试

```java
// 原始代码：UserService.java
public class UserService {
    public User findById(Long id) {
        // ...
    }
    
    public List<User> findAll() {
        // ...
    }
}
```

**右键 → Generate Unit Test**

生成的测试：
```java
// UserServiceTest.java
@Test
void testFindById() {
    // 测试正常情况
    User user = userService.findById(1L);
    assertNotNull(user);
    
    // 测试 null 情况
    User nullUser = userService.findById(null);
    assertNull(nullUser);
}

@Test
void testFindAll() {
    List<User> users = userService.findAll();
    assertNotNull(users);
    assertFalse(users.isEmpty());
}
```

### 示例 2: Chat 对话

在 Chat 面板中输入：
```
为这个类添加边界条件测试，特别是 null 值和空集合的情况
```

AI 会理解您的需求并生成相应的测试。

### 示例 3: 运行测试

生成测试后，插件会自动在终端中运行：
```bash
mvn test -Dtest=UserServiceTest
```

## 🛠️ 配置选项

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `pyutagent.apiUrl` | String | `http://localhost:8000` | 后端 API 地址 |
| `pyutagent.mode` | String | `interactive` | 运行模式（autonomous/interactive/supervised） |
| `pyutagent.timeout` | Number | `30000` | 任务超时时间（毫秒） |
| `pyutagent.maxRetries` | Number | `3` | 最大重试次数 |
| `pyutagent.autoApprove` | Boolean | `false` | 自动审批变更（⚠️ 谨慎使用） |

## 🔧 开发指南

### 本地开发

```bash
# 克隆仓库
git clone https://github.com/coding-agent/pyutagent-vscode.git
cd pyutagent-vscode

# 安装依赖
npm install

# 编译
npm run watch

# 调试
按 F5 启动 Extension Development Host
```

### 运行测试

```bash
# 运行所有测试
npm test

# 运行单元测试
npm run test:unit

# 运行集成测试
npm run test:integration
```

### 构建发布

```bash
# 打包
vsce package

# 发布
vsce publish
```

## 📚 架构说明

### 核心模块

```
pyutagent-vscode/
├── src/
│   ├── extension.ts              # 入口文件
│   ├── chat/
│   │   └── enhancedChatProvider.ts  # Chat 面板
│   ├── config/
│   │   └── configPanel.ts           # 配置面板
│   ├── backend/
│   │   └── apiClient.ts             # API 客户端
│   ├── diff/
│   │   └── diffProvider.ts          # Diff 预览
│   ├── terminal/
│   │   └── terminalManager.ts       # 终端管理
│   └── commands/
│       └── generateTest.ts          # 测试生成命令
├── test/
│   ├── integration.test.ts          # 集成测试
│   └── unit/
│       └── apiClient.test.ts        # 单元测试
└── package.json
```

### 技术栈

- **TypeScript**: 类型安全的 JavaScript
- **Monaco Editor**: VS Code 同款编辑器
- **Marked.js**: Markdown 渲染
- **Axios**: HTTP 客户端
- **VS Code Extension API**: 插件开发框架

## 🤝 贡献

欢迎贡献！请遵循以下步骤：

1. Fork 仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

### 开发规范

- 使用 TypeScript 严格模式
- 遵循 ESLint 规则
- 编写单元测试
- 更新文档

## 📝 更新日志

### [0.1.0] - 2026-03-31

**Added**
- ✨ 智能测试生成功能
- 💬 Chat 面板（支持 Markdown 和流式输出）
- 👀 Diff 预览组件
- ⌨️ 终端集成
- ⚙️ 配置管理面板
- 🧪 单元测试和集成测试

**Changed**
- 优化 UI/UX 设计
- 改进错误处理机制

**Fixed**
- 修复已知的 Bug

## 🙏 致谢

感谢以下开源项目：

- [VS Code](https://code.visualstudio.com/api) - Extension API
- [Monaco Editor](https://microsoft.github.io/monaco-editor/) - 代码编辑器
- [Marked](https://marked.js.org/) - Markdown 解析器
- [Axios](https://axios-http.com/) - HTTP 客户端

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 📞 联系方式

- **项目主页**: https://github.com/coding-agent/pyutagent-vscode
- **问题反馈**: https://github.com/coding-agent/pyutagent-vscode/issues
- **文档**: https://github.com/coding-agent/pyutagent-vscode/wiki

## 🌟 特性路线图

### v0.2.0 (计划中)
- [ ] 支持更多测试框架（TestNG, Spock）
- [ ] 代码覆盖率分析
- [ ] 测试用例优化建议
- [ ] 批量测试生成

### v0.3.0 (计划中)
- [ ] 支持其他语言（Python, JavaScript）
- [ ] 机器学习模型优化
- [ ] 团队协作功能
- [ ] 云端同步配置

---

**Made with ❤️ by PyUT Team**

如果这个插件对您有帮助，请给个 ⭐️ 星标！
