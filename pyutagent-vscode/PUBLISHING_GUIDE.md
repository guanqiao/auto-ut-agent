# PyUT Agent 发布指南

## 发布前检查清单

### ✅ 代码质量
- [ ] 所有测试通过 (`npm test`)
- [ ] 无 TypeScript 编译错误 (`npm run compile`)
- [ ] 代码格式化完成
- [ ] ESLint 检查通过

### ✅ 文档
- [ ] README.md 已更新
- [ ] CHANGELOG.md 已更新
- [ ] 使用示例已添加
- [ ] 配置说明完整

### ✅ 资源文件
- [ ] 图标文件存在 (`resources/icon.png`)
- [ ] Demo GIF 已准备 (`resources/demo.gif`)
- [ ] 截图已准备（可选）

### ✅ 版本信息
- [ ] package.json 版本号已更新
- [ ] CHANGELOG.md 已添加新版本
- [ ] Git tag 已创建

---

## 发布流程

### 1. 本地测试

```bash
# 安装依赖
npm install

# 运行测试
npm test

# 编译检查
npm run compile

# 打包测试
vsce package
```

### 2. 准备发布

#### 更新版本号

编辑 `package.json`:
```json
{
  "version": "0.1.0"  // 更新版本号
}
```

#### 更新 CHANGELOG

编辑 `CHANGELOG.md`, 添加新版本更新内容。

#### 创建 Git Tag

```bash
git add .
git commit -m "Release v0.1.0"
git tag v0.1.0
git push origin main --tags
```

### 3. 发布到 VS Code Marketplace

#### 方法 1: 使用 vsce 命令行工具

```bash
# 安装 vsce (如果未安装)
npm install -g @vscode/vsce

# 登录到 Marketplace
vsce login pyutagent

# 发布
vsce publish
```

#### 方法 2: 使用 Azure DevOps Pipeline

1. 创建 Azure DevOps 项目
2. 配置发布管道
3. 连接 GitHub 仓库
4. 自动发布

### 4. 验证发布

1. 打开 VS Code
2. 按 `Ctrl+P` / `Cmd+P`
3. 输入 `ext install pyutagent-vscode`
4. 验证插件可正常安装
5. 测试所有功能

---

## 发布后任务

### 监控和反馈

- [ ] 监控下载量
- [ ] 收集用户反馈
- [ ] 响应 Issue
- [ ] 更新文档

### 持续改进

- [ ] 修复报告的 Bug
- [ ] 实现用户请求的功能
- [ ] 性能优化
- [ ] 准备下一个版本

---

## 故障排除

### 问题 1: 发布失败 - 认证错误

**解决方案**:
```bash
# 清除缓存的凭据
vsce logout pyutagent

# 重新登录
vsce login pyutagent
```

### 问题 2: 打包失败 - 文件缺失

**错误**: `The following files are missing: resources/icon.png`

**解决方案**:
```bash
# 创建占位图标
mkdir -p resources
# 添加 128x128 PNG 图标到 resources/icon.png
```

### 问题 3: 激活失败 - 入口点错误

**错误**: `Activating extension failed: Cannot find module`

**解决方案**:
```bash
# 重新编译
npm run compile

# 检查 webpack 配置
cat webpack.config.js
```

---

## 发布检查表

### 发布前
- [ ] 代码审查完成
- [ ] 所有测试通过
- [ ] 文档更新完成
- [ ] 版本号已更新
- [ ] CHANGELOG 已更新
- [ ] Git tag 已创建

### 发布中
- [ ] 打包成功
- [ ] 发布成功
- [ ] Marketplace 页面正常显示

### 发布后
- [ ] 安装测试通过
- [ ] 功能验证完成
- [ ] 用户通知已发送
- [ ] GitHub Release 已创建

---

## 市场推广

### 渠道
- [ ] GitHub README
- [ ] Twitter/X
- [ ] LinkedIn
- [ ] 开发者社区（Reddit, Hacker News）
- [ ] 技术博客

### 内容
- 功能介绍
- 使用示例
- 截图/Demo
- 安装链接

---

## 联系支持

- **技术问题**: 创建 GitHub Issue
- **商务咨询**: 发送邮件至 support@pyutagent.com
- **社区讨论**: 加入 Discord 频道

---

**最后更新**: 2026-03-31  
**版本**: 0.1.0
