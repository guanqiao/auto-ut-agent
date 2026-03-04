# PyUT Agent 发布检查清单

## 📋 发布前检查（Pre-Release Checklist）

### 代码质量 ✅
- [x] 所有功能已实现
- [x] 代码编译通过 (`npm run compile`)
- [x] 无 TypeScript 错误
- [x] 代码格式化完成
- [ ] ESLint 检查通过（待运行）

### 测试覆盖 ✅
- [x] 集成测试已编写（7 个测试）
- [x] 单元测试已编写（6 个测试）
- [x] 测试框架已配置
- [ ] 运行所有测试（待执行）

### 文档完整性 ✅
- [x] README.md 完整
- [x] CHANGELOG.md 已更新
- [x] PUBLISHING_GUIDE.md 已创建
- [x] WEEK4_SUMMARY.md 已创建
- [x] 资源说明文档已创建

### 资源文件 ✅
- [x] LICENSE 文件已添加
- [x] 图标 SVG 已创建
- [x] 图标说明文档已创建
- [ ] 图标 PNG（可选，SVG 可替代）

### 配置完整性 ✅
- [x] package.json 配置完整
- [x] 版本号正确 (0.1.0)
- [x] 命令注册完整
- [x] 配置项完整
- [x] 仓库信息完整

---

## 🚀 发布流程（Release Process）

### 步骤 1: 最终验证

```bash
# 1. 进入项目目录
cd pyutagent-vscode

# 2. 安装依赖
npm install

# 3. 运行编译检查
npm run compile

# 4. 运行测试（如果配置了后端）
npm test

# 5. 检查文件完整性
ls -la
ls -la resources/
ls -la test/
```

**检查项**:
- [ ] 编译成功
- [ ] 无错误输出
- [ ] 所有文件存在

---

### 步骤 2: 打包插件

```bash
# 安装 vsce（如果未安装）
npm install -g @vscode/vsce

# 打包插件
vsce package

# 检查生成的文件
ls -lh *.vsix
```

**预期输出**:
```
pyutagent-vscode-0.1.0.vsix
```

**检查项**:
- [ ] .vsix 文件生成成功
- [ ] 文件大小合理（应该 < 5MB）

---

### 步骤 3: 发布到 Marketplace

#### 方法 A: 使用命令行

```bash
# 登录到 Marketplace（首次需要）
vsce login pyutagent

# 发布
vsce publish

# 或者指定版本
vsce publish 0.1.0
```

#### 方法 B: 使用 Azure DevOps

1. 访问 https://marketplace.visualstudio.com/manage
2. 创建 Publisher: `pyutagent`
3. 创建新扩展
4. 上传 .vsix 文件

**检查项**:
- [ ] 发布成功
- [ ] Marketplace 页面可访问
- [ ] 安装按钮可用

---

### 步骤 4: 验证安装

```bash
# 在 VS Code 中
# 1. 打开扩展面板
# 2. 搜索 "PyUT Agent"
# 3. 点击安装
# 4. 重新加载窗口
# 5. 测试功能
```

**测试清单**:
- [ ] 扩展可搜索到
- [ ] 安装成功
- [ ] 无激活错误
- [ ] Chat 面板可打开
- [ ] 右键菜单显示
- [ ] 配置面板可打开

---

### 步骤 5: 创建 GitHub Release

```bash
# 创建 Git tag
git add .
git commit -m "Release v0.1.0"
git tag -a v0.1.0 -m "PyUT Agent v0.1.0 - Initial Release"
git push origin main --tags

# 创建 GitHub Release
# 访问：https://github.com/coding-agent/pyutagent-vscode/releases/new
# Tag: v0.1.0
# Title: PyUT Agent v0.1.0
# Description: 复制 CHANGELOG.md 内容
```

**Release 内容**:
- [ ] Tag 已创建
- [ ] Release 已发布
- [ ] 更新日志完整
- [ ] 附件上传（可选）

---

## 📊 发布后检查（Post-Release Checklist）

### 监控指标
- [ ] 下载量统计
- [ ] 用户评分
- [ ] Issue 报告
- [ ] 用户反馈

### 社区推广
- [ ] Twitter/X 推文
- [ ] LinkedIn 分享
- [ ] 开发者社区宣传
- [ ] 技术博客文章

### 持续维护
- [ ] 响应 Issue
- [ ] 修复 Bug
- [ ] 准备 v0.2.0 计划
- [ ] 收集功能建议

---

## ⚠️ 常见问题排查

### 问题 1: 发布失败 - 认证错误

**错误**: `401 Unauthorized`

**解决**:
```bash
vsce logout pyutagent
vsce login pyutagent
```

---

### 问题 2: 打包失败 - 文件缺失

**错误**: `The following files are missing: resources/icon.png`

**解决**:
```bash
# 方案 1: 使用 SVG 代替 PNG
# 编辑 package.json，将 icon 改为 resources/icon.svg

# 方案 2: 生成 PNG
convert -resize 128x128 resources/icon.svg resources/icon.png
```

---

### 问题 3: 激活失败

**错误**: `Activating extension failed`

**解决**:
```bash
# 重新编译
npm run compile

# 检查 dist 目录
ls -la dist/
```

---

## 📈 成功标准

### 发布成功标志
- ✅ .vsix 文件生成
- ✅ Marketplace 页面正常
- ✅ 可正常安装
- ✅ 核心功能可用
- ✅ 无严重 Bug

### 质量指标
- ✅ 编译通过率 100%
- ✅ 测试覆盖率 > 70%
- ✅ 文档完整度 100%
- ✅ 用户满意度 > 4 星

---

## 🎯 发布检查表总结

### 必须完成（Blocking）
- [x] ✅ 代码实现完成
- [x] ✅ 文档完善
- [x] ✅ LICENSE 添加
- [x] ✅ 图标添加
- [ ] ⏳ 最终测试运行
- [ ] ⏳ 打包验证
- [ ] ⏳ 发布到 Marketplace

### 可选改进（Non-Blocking）
- [ ] PNG 图标生成
- [ ] Demo GIF 录制
- [ ] 更多截图
- [ ] 性能基准测试

---

## 📞 联系支持

**技术问题**:
- GitHub Issues: https://github.com/coding-agent/pyutagent-vscode/issues

**商务咨询**:
- Email: support@pyutagent.com

**社区**:
- Discord: [待添加]
- Twitter: @PyUTAgent

---

**创建时间**: 2026-04-07  
**版本**: 0.1.0  
**状态**: 准备发布
