## 计划：为所有 GUI 文件添加日志记录

### 涉及的文件
1. `pyutagent/ui/dialogs/llm_config_dialog.py`
2. `pyutagent/ui/dialogs/aider_config_dialog.py`
3. `pyutagent/ui/main_window.py`
4. `pyutagent/ui/chat_widget.py`

### 修改内容

#### 1. llm_config_dialog.py
- 添加 `import logging` 和 `logger = logging.getLogger(__name__)`
- **LLMTestThread.run()** (第31-40行): 现有 except 块中添加 `logger.exception()`
- **load_current_config()**: 添加 try-except 包裹方法体
- **save_current_config()**: 添加 try-except 包裹方法体
- **on_config_selected()**: 添加 try-except
- **on_add_config()**: 添加 try-except
- **on_delete_config()**: 添加 try-except
- **on_set_default()**: 添加 try-except
- **test_connection()**: 添加 try-except
- **on_aider_config()**: 添加 try-except
- **closeEvent()**: 添加 try-except

#### 2. aider_config_dialog.py
- 添加 `import logging` 和 `logger = logging.getLogger(__name__)`
- **setup_ui()**: 添加 try-except
- **_create_general_tab()**: 添加 try-except
- **_create_architect_tab()**: 添加 try-except
- **_populate_model_combo()**: 添加 try-except
- **_create_multifile_tab()**: 添加 try-except
- **_create_format_tab()**: 添加 try-except
- **_on_architect_toggled()**: 添加 try-except
- **_on_multifile_toggled()**: 添加 try-except
- **load_config()**: 添加 try-except
- **get_config()**: 添加 try-except
- **accept()**: 添加 try-except

#### 3. main_window.py
- 添加 `import logging` 和 `logger = logging.getLogger(__name__)`
- **AgentWorker.run()** (第58-96行): 现有 except 块中添加 `logger.exception()`
- **ProjectTreeWidget.load_project()**: 添加 try-except
- **ProjectTreeWidget._add_directory()**: 现有 except 块中添加日志
- **setup_llm_client()**: 现有 except 块中添加 `logger.exception()`
- **on_open_project()**: 添加 try-except
- **on_llm_config()**: 添加 try-except
- **on_aider_config()**: 添加 try-except
- **on_scan_project()**: 添加 try-except
- **on_generate_tests()**: 添加 try-except
- **on_pause_generation()**: 添加 try-except
- **start_generation()**: 添加 try-except
- **on_agent_progress()**: 添加 try-except
- **on_agent_state_changed()**: 添加 try-except
- **on_agent_log()**: 添加 try-except
- **on_agent_completed()**: 添加 try-except
- **on_agent_error()**: 添加 try-except
- **on_file_selected()**: 添加 try-except
- **on_message_sent()**: 添加 try-except
- **on_about()**: 添加 try-except
- **closeEvent()**: 添加 try-except

#### 4. chat_widget.py
- 添加 `import logging` 和 `logger = logging.getLogger(__name__)`
- **ChatMessageWidget.__init__()**: 添加 try-except
- **ChatMessageWidget.update_content()**: 添加 try-except
- **ChatWidget.setup_ui()**: 添加 try-except
- **ChatWidget.add_message()**: 添加 try-except
- **ChatWidget.update_last_message()**: 添加 try-except
- **ChatWidget.send_message()**: 添加 try-except
- **ChatWidget.scroll_to_bottom()**: 添加 try-except
- **ChatWidget.on_generate()**: 添加 try-except
- **ChatWidget.on_pause()**: 添加 try-except
- **ChatWidget.on_status()**: 添加 try-except
- **ChatWidget.on_clear()**: 添加 try-except
- **ChatWidget.clear_chat()**: 添加 try-except
- **ChatWidget.add_agent_message()**: 添加 try-except
- **ChatWidget.set_input_enabled()**: 添加 try-except

### 日志使用规范
- 使用 `logger = logging.getLogger(__name__)` 获取 logger
- 异常捕获使用 `logger.exception("错误描述")` 自动记录堆栈
- 错误信息使用 `logger.error("错误描述")`
- 警告信息使用 `logger.warning("警告描述")`
- 关键操作使用 `logger.info("操作描述")`

### 修改原则
- 保持原有代码逻辑不变
- 仅在必要时添加 try-except 块
- 确保所有异常都被记录到日志
- 使用项目统一的日志风格（参考 error_recovery.py 和 client.py）