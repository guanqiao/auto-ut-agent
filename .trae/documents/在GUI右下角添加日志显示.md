## 实现方案

修改 `ProgressWidget` 类，使用 `QSplitter` 将右侧区域垂直分割为上下两部分：

### 修改内容

1. **修改 `ProgressWidget.setup_ui()` 方法**:
   - 使用 `QSplitter(Qt.Orientation.Vertical)` 分割区域
   - 上半部分：现有的进度信息（状态、进度条、覆盖率等）
   - 下半部分：增强的日志显示区域

2. **增强日志显示功能**:
   - 增大日志区域默认高度
   - 添加日志级别颜色区分（INFO-蓝色、WARNING-橙色、ERROR-红色）
   - 添加"清空日志"按钮
   - 添加日志自动滚动功能

3. **修改 `add_log()` 方法**:
   - 支持带级别的日志显示
   - 自动滚动到底部

### 代码变更位置
- 文件: `pyutagent/ui/main_window.py`
- 类: `ProgressWidget`
- 方法: `setup_ui()`, `add_log()`

### 预期效果
- 右下角显示一个可调整大小的日志面板
- 日志与进度信息分离，更清晰
- 支持颜色区分的日志级别显示