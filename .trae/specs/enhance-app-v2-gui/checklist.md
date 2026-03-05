# Checklist - App V2 GUI 功能增强

## Phase 1: 核心交互能力（P0）

### Task 1: AI 流式响应与 Markdown 渲染
- [x] `pyutagent/ui/components/markdown_renderer.py` 文件创建完成
- [x] MarkdownRenderer 类实现完整
- [x] 支持代码块、列表、表格、引用渲染
- [x] 代码块语法高亮功能正常
- [x] `pyutagent/ui/components/streaming_handler.py` 文件创建完成
- [x] StreamingHandler 类实现完整
- [x] 打字机效果流畅（延迟 < 100ms/字）
- [ ] ChatMessageWidget 集成 Markdown 渲染
- [x] 代码块复制按钮功能正常
- [x] 代码块插入到编辑器按钮功能正常
- [x] `pyutagent/ui/components/thinking_expander.py` 文件创建完成
- [x] ThinkingExpander 组件支持折叠/展开
- [x] 思考耗时显示准确
- [ ] LLM 流式 API 连接正常
- [ ] 流式响应错误处理完善
- [ ] 单元测试覆盖率 > 90%

### Task 2: 行内代码编辑（Inline Edit）
- [ ] 技术方案确定（Monaco/QTextEdit）
- [ ] `pyutagent/ui/editor/ghost_text.py` 文件创建完成
- [ ] GhostText 层正确叠加在编辑器上
- [ ] 幽灵文本半透明灰色显示
- [ ] 支持多行幽灵文本
- [ ] `pyutagent/ui/editor/inline_diff.py` 文件创建完成
- [ ] 添加行显示绿色背景 (#1E4620)
- [ ] 删除行显示红色背景 (#4A1E1E)
- [ ] 字符级 Diff 高亮
- [ ] Tab 键接受建议功能正常
- [ ] Esc 键拒绝建议功能正常
- [ ] 视觉提示（如幽灵文本边框）清晰
- [ ] `pyutagent/ui/editor/ai_suggestion_provider.py` 文件创建完成
- [ ] AI 建议生成接口正常
- [ ] 建议解析为编辑操作正确
- [ ] 多行修改一次性应用
- [ ] 撤销操作支持
- [ ] 单元测试覆盖率 > 90%

### Task 3: Agent 执行可视化
- [x] `pyutagent/ui/agent_panel/agent_worker.py` 文件创建完成
- [x] AgentWorker 信号定义完整
- [x] 状态广播机制正常
- [ ] AutonomousLoop 连接成功
- [ ] ToolCallWidget 实时状态更新
- [ ] 工具参数显示正确
- [ ] 工具执行结果展示完整
- [ ] 工具调用可展开/折叠
- [x] `pyutagent/ui/components/progress_tracker.py` 文件创建完成
- [x] 当前步骤/总步骤显示正确
- [x] 进度百分比计算准确
- [x] 步骤详情可展开查看
- [x] `pyutagent/ui/components/error_display.py` 文件创建完成
- [x] 错误红色高亮显示
- [x] 重试按钮功能正常
- [x] 跳过按钮功能正常
- [ ] AgentMode 实时更新 ThinkingChain
- [ ] AgentMode 实时更新 ToolCalls
- [ ] 单元测试覆盖率 > 90%

---

## Phase 2: 上下文与搜索增强（P1）

### Task 4: @符号引用增强
- [ ] `pyutagent/ui/services/symbol_indexer.py` 文件创建完成
- [ ] SymbolIndexer 类实现完整
- [ ] 类/方法/函数解析正确
- [ ] 符号索引构建成功
- [ ] 增量更新功能正常
- [ ] MentionSystem 扩展 @symbol 类型
- [ ] 符号搜索响应 < 200ms
- [ ] MentionPopup 类型分组显示（类/方法/变量）
- [ ] 模糊匹配功能正常
- [ ] 最近使用优先排序
- [ ] ContextManager 支持符号类型上下文
- [ ] 符号定义预览显示正确
- [ ] 单元测试覆盖率 > 90%

### Task 5: 代码库语义搜索
- [ ] `pyutagent/ui/dialogs/semantic_search_dialog.py` 文件创建完成
- [ ] 搜索界面设计完成
- [ ] 搜索结果列表展示正常
- [ ] 预览面板功能完整
- [ ] `pyutagent/ui/services/semantic_search.py` 文件创建完成
- [ ] 自然语言查询支持
- [ ] 相似度计算准确
- [ ] 搜索结果相关度 > 80%
- [ ] Ctrl+Shift+F 快捷键绑定
- [ ] Command Palette 集成
- [ ] 单元测试覆盖率 > 90%

### Task 6: 终端 AI 集成
- [ ] 终端输出错误检测功能正常
- [ ] 常见错误模式识别（Python/Java/编译错误）
- [ ] 错误行高亮显示
- [ ] "Ask AI" 按钮添加成功
- [ ] 点击发送错误到 AI 功能正常
- [ ] AI 修复建议显示正确
- [ ] Diff 预览功能正常
- [ ] 一键应用修复功能正常

---

## Phase 3: 体验优化（P2）

### Task 7: 文件树增强
- [ ] Git 状态检测功能正常
- [ ] 修改文件绿色标识
- [ ] 新增文件黄色标识
- [ ] 删除文件红色标识
- [ ] 文件树搜索框添加成功
- [ ] 模糊匹配搜索功能正常
- [ ] 搜索结果高亮显示
- [ ] 文件拖拽到上下文功能正常
- [ ] 文件拖拽到编辑器功能正常
- [ ] 拖拽视觉反馈清晰

### Task 8: 命令面板完善
- [ ] Command Palette 完整命令列表
- [ ] 模糊搜索功能正常
- [ ] 快捷键显示正确
- [ ] 快捷键配置对话框创建
- [ ] 自定义绑定功能正常
- [ ] 快捷键冲突检测
- [ ] 配置保存成功

---

## 整体验收

### 功能验收
- [ ] 所有 P0 任务完成
- [ ] 所有 P1 任务完成
- [ ] 所有 P2 任务完成
- [ ] AI 流式响应可用，用户体验流畅
- [ ] Markdown 渲染正确率 > 95%
- [ ] 行内编辑 Tab/Esc 交互可用
- [ ] Agent 执行过程实时可见
- [ ] @symbol 引用可用
- [ ] 语义搜索返回结果相关
- [ ] 终端 AI 修复可用
- [ ] 文件树 Git 状态显示正确
- [ ] 命令面板模糊搜索可用

### 性能验收
- [ ] 首 token 响应时间 < 2s
- [ ] 流式渲染帧率 > 30fps
- [ ] @symbol 自动完成响应 < 200ms
- [ ] 内存占用 < 500MB

### 质量验收
- [ ] 所有单元测试通过率 > 90%
- [ ] 所有集成测试通过
- [ ] 代码覆盖率 > 80%
- [ ] 代码审查通过
- [ ] 文档完整

### 用户体验验收
- [ ] 用户感觉 "Chat 真正可用"
- [ ] 与 Cursor 相比交互体验覆盖 > 70%
- [ ] 无明显卡顿或闪烁
- [ ] 错误提示清晰友好
