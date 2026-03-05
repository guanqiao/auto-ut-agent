"""Main window v2 with new three-panel layout.

This is the refactored main window using the new layout system.
It maintains backward compatibility with the existing agent and config systems.
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QMessageBox, QFileDialog, QApplication, QDialog
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QAction, QKeySequence

from .layout import MainLayout
from .panels import SidebarPanel, ContentPanel
from .agent_panel import AgentPanel
from .session import SessionManager
from .commands import SlashCommandHandler, MentionSystem
from .styles import get_style_manager
from .components import get_notification_manager, ThinkingExpander, ThinkingStep, ThinkingStatus
from .editor.approval_diff_viewer import ApprovalDialog
from .terminal.embedded_terminal import TerminalWidget
from .services.semantic_search import SemanticSearchService

# Import existing components (backward compatibility)
from ..core.config import (
    LLMConfigCollection, AiderConfig, AppState,
    load_llm_config, save_llm_config,
    load_aider_config, save_aider_config,
    load_app_state, save_app_state, get_settings
)
from ..llm.client import LLMClient

logger = logging.getLogger(__name__)


class MainWindowV2(QMainWindow):
    """Refactored main window with three-panel layout.
    
    Features:
    - Three-panel layout (sidebar, content, agent panel)
    - Collapsible panels
    - Multiple layout modes
    - Multi-language project support
    - Session management
    - Slash commands and @mentions
    - Streaming AI responses with Markdown rendering
    """
    
    project_opened = pyqtSignal(str)
    generate_requested = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyUT Agent - AI Coding Assistant")
        self.setGeometry(100, 100, 1600, 900)
        
        # Initialize state
        self.current_project: str = ""
        self.config_collection: LLMConfigCollection = load_llm_config()
        self.aider_config: AiderConfig = load_aider_config()
        self.app_state: AppState = load_app_state()
        self.llm_client: Optional[LLMClient] = None
        
        # Streaming state
        self._current_streaming_task: Optional[asyncio.Task] = None
        self._is_streaming = False
        
        # Initialize managers
        self._session_manager = SessionManager()
        self._slash_handler = SlashCommandHandler()
        self._mention_system = MentionSystem()
        self._semantic_search_service: Optional[SemanticSearchService] = None

        # Initialize UI components
        self._style_manager = get_style_manager()
        self._notification_manager = get_notification_manager()
        
        self.setup_ui()
        self.setup_menu()
        self.setup_shortcuts()
        self.apply_styles()
        self.setup_llm_client()
        
        # Connect chat mode signals
        self._connect_chat_signals()
        
        # Create initial session
        self._create_new_session()
        
    def _connect_chat_signals(self):
        """Connect chat mode signals."""
        chat_mode = self._agent_panel.get_chat_mode()
        chat_mode.code_copy_requested.connect(self._on_code_copy)
        chat_mode.code_insert_requested.connect(self._on_code_insert)
        chat_mode.streaming_started.connect(self._on_streaming_started)
        chat_mode.streaming_finished.connect(self._on_streaming_finished)
        
    def _on_code_copy(self, code: str):
        """Handle code copy request."""
        clipboard = QApplication.clipboard()
        clipboard.setText(code)
        self._notification_manager.show_success("Code copied to clipboard", duration=2000)
        
    def _on_code_insert(self, code: str):
        """Handle code insert request."""
        current_file = self._content_panel.get_current_file()
        if current_file:
            self._content_panel.insert_text(code)
            self._notification_manager.show_success("Code inserted into editor", duration=2000)
        else:
            self._notification_manager.show_warning("No file open in editor", duration=3000)
            
    def _on_streaming_started(self):
        """Handle streaming started."""
        self._is_streaming = True
        logger.info("AI response streaming started")
        
    def _on_streaming_finished(self):
        """Handle streaming finished."""
        self._is_streaming = False
        self._current_streaming_task = None
        logger.info("AI response streaming finished")
        
    def setup_ui(self):
        """Setup the main UI."""
        # Create central widget with new layout
        self._main_layout = MainLayout(self)
        self.setCentralWidget(self._main_layout)
        
        # Setup sidebar (file tree)
        self._sidebar_panel = SidebarPanel()
        self._sidebar_panel.file_selected.connect(self.on_file_selected)
        self._sidebar_panel.file_activated.connect(self.on_file_activated)
        self._main_layout.set_panel_widget(MainLayout.SIDEBAR_PANEL, self._sidebar_panel)
        
        # Setup content panel (editor)
        self._content_panel = ContentPanel()
        self._content_panel.file_opened.connect(self.on_file_opened)
        self._main_layout.set_panel_widget(MainLayout.CONTENT_PANEL, self._content_panel)
        
        # Setup agent panel (chat)
        self._agent_panel = AgentPanel()
        self._agent_panel.message_sent.connect(self.on_message_sent)
        self._agent_panel.generate_requested.connect(self.on_generate_tests)
        self._agent_panel.context_changed.connect(self.on_context_changed)
        self._main_layout.set_panel_widget(MainLayout.AGENT_PANEL, self._agent_panel)
        
        # Setup slash commands and mentions
        self._setup_chat_input_handlers()
        
        # Connect layout signals
        self._main_layout.panel_collapsed.connect(self.on_panel_collapsed)
        
    def _setup_chat_input_handlers(self):
        """Setup slash command and mention handlers for chat input."""
        chat_mode = self._agent_panel.get_chat_mode()
        input_widget = chat_mode._input
        
        # Attach slash command handler
        self._slash_handler.attach_to_input(input_widget, self)
        self._slash_handler.command_triggered.connect(self._on_slash_command)
        
        # Attach mention system
        self._mention_system.attach_to_input(input_widget, self)
        self._mention_system.mention_triggered.connect(self._on_mention)
        self._mention_system.set_project_path(self.current_project)
        
        # Override key press to handle popup navigation
        original_key_press = input_widget.keyPressEvent
        
        def custom_key_press(event):
            # Check if handlers want to handle the event
            if self._slash_handler.handle_key_press(event):
                return
            if self._mention_system.handle_key_press(event):
                return
            # Otherwise use original handler
            original_key_press(event)
        
        input_widget.keyPressEvent = custom_key_press
        
    def setup_menu(self):
        """Setup menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        open_action = QAction("&Open Project...", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self.on_open_project)
        file_menu.addAction(open_action)
        
        file_menu.addSeparator()
        
        # Session submenu
        session_menu = file_menu.addMenu("&Session")
        
        new_session_action = QAction("&New Session", self)
        new_session_action.setShortcut("Ctrl+N")
        new_session_action.triggered.connect(self._create_new_session)
        session_menu.addAction(new_session_action)
        
        history_action = QAction("&History...", self)
        history_action.setShortcut("Ctrl+H")
        history_action.triggered.connect(self._show_session_history)
        session_menu.addAction(history_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Search menu
        search_menu = menubar.addMenu("&Search")

        semantic_search_action = QAction("&Semantic Search...", self)
        semantic_search_action.setShortcut("Ctrl+Shift+F")
        semantic_search_action.triggered.connect(self.on_semantic_search)
        search_menu.addAction(semantic_search_action)

        search_menu.addSeparator()

        find_in_files_action = QAction("Find in &Files...", self)
        find_in_files_action.setShortcut("Ctrl+Shift+H")
        find_in_files_action.triggered.connect(self.on_find_in_files)
        search_menu.addAction(find_in_files_action)

        # View menu
        view_menu = menubar.addMenu("&View")

        toggle_sidebar_action = QAction("Toggle &Sidebar", self)
        toggle_sidebar_action.setShortcut("Ctrl+B")
        toggle_sidebar_action.triggered.connect(
            lambda: self._main_layout.toggle_panel(MainLayout.SIDEBAR_PANEL)
        )
        view_menu.addAction(toggle_sidebar_action)
        
        toggle_agent_action = QAction("Toggle &Agent Panel", self)
        toggle_agent_action.setShortcut("Ctrl+J")
        toggle_agent_action.triggered.connect(
            lambda: self._main_layout.toggle_panel(MainLayout.AGENT_PANEL)
        )
        view_menu.addAction(toggle_agent_action)
        
        view_menu.addSeparator()
        
        # Layout modes
        default_layout_action = QAction("&Default Layout", self)
        default_layout_action.triggered.connect(
            lambda: self._main_layout.set_layout_mode(MainLayout.MODE_DEFAULT)
        )
        view_menu.addAction(default_layout_action)
        
        focus_editor_action = QAction("Focus &Editor", self)
        focus_editor_action.triggered.connect(
            lambda: self._main_layout.set_layout_mode(MainLayout.MODE_FOCUS_EDITOR)
        )
        view_menu.addAction(focus_editor_action)
        
        focus_agent_action = QAction("Focus A&gent", self)
        focus_agent_action.triggered.connect(
            lambda: self._main_layout.set_layout_mode(MainLayout.MODE_FOCUS_AGENT)
        )
        view_menu.addAction(focus_agent_action)

        view_menu.addSeparator()

        terminal_action = QAction("&Terminal", self)
        terminal_action.setShortcut("Ctrl+`")
        terminal_action.triggered.connect(self.on_show_terminal)
        view_menu.addAction(terminal_action)

        # Settings menu
        settings_menu = menubar.addMenu("&Settings")
        
        llm_config_action = QAction("&LLM Configuration...", self)
        llm_config_action.triggered.connect(self.on_llm_config)
        settings_menu.addAction(llm_config_action)
        
        # Help menu
        help_menu = menubar.addMenu("&Help")
        
        commands_action = QAction("&Slash Commands", self)
        commands_action.setShortcut("Ctrl+/")
        commands_action.triggered.connect(self._show_slash_commands_help)
        help_menu.addAction(commands_action)

        help_menu.addSeparator()

        review_action = QAction("&Review Changes...", self)
        review_action.setShortcut("Ctrl+Shift+R")
        review_action.triggered.connect(self.on_review_changes)
        help_menu.addAction(review_action)

        about_action = QAction("&About", self)
        about_action.triggered.connect(self.on_about)
        help_menu.addAction(about_action)
        
    def setup_shortcuts(self):
        """Setup keyboard shortcuts."""
        # Command palette shortcut
        command_palette_shortcut = QAction(self)
        command_palette_shortcut.setShortcut("Ctrl+Shift+P")
        command_palette_shortcut.triggered.connect(self.on_command_palette)
        self.addAction(command_palette_shortcut)

        # Semantic search shortcut (Ctrl+Shift+F)
        semantic_search_shortcut = QAction(self)
        semantic_search_shortcut.setShortcut("Ctrl+Shift+F")
        semantic_search_shortcut.triggered.connect(self.on_semantic_search)
        self.addAction(semantic_search_shortcut)

        # Cancel streaming shortcut
        cancel_streaming_shortcut = QAction(self)
        cancel_streaming_shortcut.setShortcut("Escape")
        cancel_streaming_shortcut.triggered.connect(self._cancel_streaming)
        self.addAction(cancel_streaming_shortcut)
        
    def _cancel_streaming(self):
        """Cancel current streaming response."""
        if self._is_streaming and self.llm_client:
            logger.info("Cancelling streaming response")
            self.llm_client.cancel()
            self._notification_manager.show_info("Response cancelled", duration=2000)
        
    def apply_styles(self):
        """Apply theme styles."""
        self._style_manager.apply_stylesheet(self, "main_window")
        
    def setup_llm_client(self):
        """Setup LLM client."""
        try:
            default_config = self.config_collection.get_default_config()
            if default_config:
                self.llm_client = LLMClient.from_config(default_config)
                self._main_layout.set_status_llm(
                    f"🟢 LLM: {default_config.get_display_name()}",
                    connected=True
                )
                self._notification_manager.show_success(
                    f"LLM client initialized: {default_config.get_display_name()}",
                    duration=3000
                )
            else:
                self._main_layout.set_status_llm("🔴 LLM: Not configured")
        except Exception as e:
            logger.exception("Failed to initialize LLM client")
            self._main_layout.set_status_llm("🔴 LLM: Error")
            
    def _create_new_session(self):
        """Create a new chat session."""
        session = self._session_manager.create_session()
        self._session_manager.set_current_session(session.id)
        
        # Clear chat
        chat_mode = self._agent_panel.get_chat_mode()
        chat_mode.clear_messages()
        
        # Add welcome message
        chat_mode.add_agent_message(
            "New session started! I'm your AI coding assistant.\n\n"
            "Quick tips:\n"
            "• Type / to see available commands\n"
            "• Use @ to mention files\n"
            "• Select a file and click 'Generate Tests' to get started"
        )
        
        logger.info(f"Created new session: {session.id}")
        
    def _show_session_history(self):
        """Show session history dialog."""
        from .dialogs.session_history_dialog import SessionHistoryDialog
        
        dialog = SessionHistoryDialog(self._session_manager, self)
        dialog.session_selected.connect(self._on_session_selected_from_history)
        dialog.exec()
        
    def _on_session_selected_from_history(self, session_id: str):
        """Handle session selection from history."""
        session = self._session_manager.get_session(session_id)
        if not session:
            return
        
        self._session_manager.set_current_session(session_id)
        
        # Load messages into chat
        chat_mode = self._agent_panel.get_chat_mode()
        chat_mode.clear_messages()
        
        for msg in session.messages:
            if msg.role == 'user':
                chat_mode.add_user_message(msg.content)
            elif msg.role == 'agent':
                chat_mode.add_agent_message(msg.content)
                
        logger.info(f"Loaded session: {session_id}")
        
    def _show_slash_commands_help(self):
        """Show slash commands help."""
        commands = self._slash_handler.get_all_commands()
        
        help_text = "<h2>Slash Commands</h2><ul>"
        for cmd in sorted(commands, key=lambda c: c.name):
            help_text += f"<li><b>{cmd.display_name}</b> - {cmd.description}"
            if cmd.example:
                help_text += f"<br><small>Example: {cmd.example}</small>"
            help_text += "</li>"
        help_text += "</ul>"
        
        QMessageBox.information(self, "Slash Commands", help_text)
        
    def _on_slash_command(self, command_name: str, args: str):
        """Handle slash command execution."""
        logger.info(f"Slash command: {command_name} with args: {args}")
        
        # Handle different commands
        if command_name == 'test':
            self.on_generate_tests()
        elif command_name == 'explain':
            self._handle_explain_command(args)
        elif command_name == 'refactor':
            self._handle_refactor_command(args)
        elif command_name == 'doc':
            self._handle_doc_command(args)
        elif command_name == 'review':
            self._handle_review_command(args)
        else:
            # Generic command handling
            self._agent_panel.add_agent_message(
                f"Executing command: /{command_name} {args}\n\n"
                "(This is a placeholder - actual implementation would process the command)"
            )
            
    def _on_mention(self, mention_type: str, mention_id: str):
        """Handle @mention."""
        logger.info(f"Mention: {mention_type} - {mention_id}")
        
        # Add to context
        if mention_type == 'file':
            self._agent_panel.add_file_to_context(mention_id)
        elif mention_type == 'current':
            current_file = self._content_panel.get_current_file()
            if current_file:
                self._agent_panel.add_file_to_context(current_file)
                
    def _handle_explain_command(self, args: str):
        """Handle /explain command."""
        current_file = self._content_panel.get_current_file()
        if current_file:
            self._agent_panel.add_agent_message(
                f"I'll explain the code in {Path(current_file).name}..."
            )
            # TODO: Send to AI for explanation
        else:
            self._agent_panel.add_agent_message(
                "Please open a file first or specify what you'd like me to explain."
            )
            
    def _handle_refactor_command(self, args: str):
        """Handle /refactor command."""
        current_file = self._content_panel.get_current_file()
        if current_file:
            self._agent_panel.add_agent_message(
                f"I'll refactor the code in {Path(current_file).name}..."
            )
            # TODO: Send to AI for refactoring
        else:
            self._agent_panel.add_agent_message(
                "Please open a file first."
            )
            
    def _handle_doc_command(self, args: str):
        """Handle /doc command."""
        current_file = self._content_panel.get_current_file()
        if current_file:
            self._agent_panel.add_agent_message(
                f"I'll generate documentation for {Path(current_file).name}..."
            )
            # TODO: Send to AI for documentation generation
        else:
            self._agent_panel.add_agent_message(
                "Please open a file first."
            )
            
    def _handle_review_command(self, args: str):
        """Handle /review command."""
        current_file = self._content_panel.get_current_file()
        if current_file:
            self._agent_panel.add_agent_message(
                f"I'll review the code in {Path(current_file).name}..."
            )
            # TODO: Send to AI for code review
        else:
            self._agent_panel.add_agent_message(
                "Please open a file first."
            )
            
    def on_open_project(self):
        """Handle open project action."""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Project",
            "",
            QFileDialog.Option.ShowDirsOnly
        )
        
        if dir_path:
            self._open_project(dir_path)
            
    def _open_project(self, dir_path: str) -> bool:
        """Open a project by path."""
        try:
            self.current_project = dir_path
            self._sidebar_panel.load_project(dir_path)
            self.project_opened.emit(dir_path)
            
            # Update mention system
            self._mention_system.set_project_path(dir_path)
            
            # Update status
            project_name = Path(dir_path).name
            self._main_layout.set_status_project(f"📁 {project_name}", connected=True)
            self._main_layout.set_context_label(f"Project: {project_name}")
            
            # Save to app state
            self.app_state.add_project(dir_path)
            save_app_state(self.app_state)
            
            # Show notification
            self._notification_manager.show_success(
                f"Project '{project_name}' opened",
                duration=3000
            )
            
            self._agent_panel.add_agent_message(
                f"Project opened: {project_name}\n"
                "Select a file to start working with the AI assistant."
            )
            
            return True
        except Exception as e:
            logger.exception(f"Failed to open project: {dir_path}")
            self._notification_manager.show_error(f"Failed to open project: {e}")
            return False
            
    def on_file_selected(self, file_path: str):
        """Handle file selection."""
        logger.info(f"File selected: {file_path}")
        # Optionally preview file without opening tab
        
    def on_file_activated(self, file_path: str):
        """Handle file activation (double-click)."""
        logger.info(f"File activated: {file_path}")
        self._content_panel.open_file(file_path)
        
    def on_file_opened(self, file_path: str):
        """Handle file opened in editor."""
        logger.info(f"File opened: {file_path}")
        
    def on_message_sent(self, message: str):
        """Handle user message with streaming AI response."""
        logger.info(f"User message: {message}")
        
        # Save to session
        self._session_manager.add_message_to_current('user', message)
        
        # Check if it's a slash command
        if self._slash_handler.is_command(message):
            return
        
        # Check if LLM client is available
        if not self.llm_client:
            self._agent_panel.add_agent_message(
                "⚠️ LLM not configured. Please configure LLM settings first."
            )
            return
        
        # Start streaming response
        chat_mode = self._agent_panel.get_chat_mode()
        message_id = chat_mode.start_streaming_response()
        
        # Build context from mentioned files
        context = self._build_message_context(message)
        
        # Create async task for streaming
        try:
            import qasync
            loop = qasync.QEventLoop.instance()
            self._current_streaming_task = loop.create_task(
                self._stream_ai_response(message, context, chat_mode, message_id)
            )
        except Exception as e:
            logger.exception("Failed to start streaming task")
            chat_mode.finish_streaming()
            chat_mode.update_message(
                message_id,
                f"❌ Error: Failed to start AI response - {str(e)}"
            )
            
    async def _stream_ai_response(
        self, 
        message: str, 
        context: str, 
        chat_mode, 
        message_id: str
    ):
        """Stream AI response asynchronously.
        
        Args:
            message: User message
            context: Additional context (file contents, etc.)
            chat_mode: Chat mode widget
            message_id: Message ID for updates
        """
        try:
            # Build prompt
            if context:
                full_prompt = f"Context:\n{context}\n\nUser: {message}"
            else:
                full_prompt = message
            
            # System prompt
            system_prompt = (
                "You are an AI coding assistant. Help the user with their programming tasks.\n"
                "Use Markdown formatting for your responses.\n"
                "When providing code, use code blocks with the appropriate language."
            )
            
            logger.info(f"Starting AI stream for message: {message[:50]}...")
            
            # Reset cancellation state
            self.llm_client.reset_cancel()
            
            # Stream response
            full_response = ""
            async for chunk in self.llm_client.astream(full_prompt, system_prompt):
                full_response += chunk
                chat_mode.append_to_streaming(chunk)
                
            # Save to session
            self._session_manager.add_message_to_current('agent', full_response)
            
            logger.info(f"AI response completed: {len(full_response)} chars")
            
        except asyncio.CancelledError:
            logger.info("AI response streaming cancelled")
            chat_mode.append_to_streaming("\n\n[Response cancelled by user]")
        except Exception as e:
            logger.exception("AI response streaming failed")
            error_msg = f"\n\n❌ Error: {str(e)}"
            chat_mode.append_to_streaming(error_msg)
        finally:
            chat_mode.finish_streaming()
            
    def _build_message_context(self, message: str) -> str:
        """Build context from mentioned files and current file.
        
        Args:
            message: User message to check for mentions
            
        Returns:
            Context string with file contents
        """
        context_parts = []
        
        # Add current file if open
        current_file = self._content_panel.get_current_file()
        if current_file:
            try:
                with open(current_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                context_parts.append(f"Current file ({Path(current_file).name}):\n```\n{content}\n```")
            except Exception as e:
                logger.warning(f"Could not read current file: {e}")
        
        # Add context from context manager
        try:
            context_manager = self._agent_panel.get_context_manager()
            for item in context_manager.get_items():
                if item.file_path and item.file_path != current_file:
                    try:
                        with open(item.file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        context_parts.append(
                            f"File ({Path(item.file_path).name}):\n```\n{content[:2000]}\n```"
                        )
                    except Exception as e:
                        logger.warning(f"Could not read context file: {e}")
        except Exception as e:
            logger.warning(f"Could not get context: {e}")
        
        return "\n\n".join(context_parts)
        
    def on_generate_tests(self):
        """Handle generate tests request with streaming."""
        selected_file = self._sidebar_panel.get_file_tree().get_selected_path()
        current_file = self._content_panel.get_current_file()
        
        target_file = selected_file or current_file
        
        if not target_file:
            self._agent_panel.add_agent_message(
                "Please select a file first."
            )
            return
            
        logger.info(f"Generate tests for: {target_file}")
        
        # Switch to agent mode
        self._agent_panel.start_agent_task("Generating Tests")
        
        # Add thinking steps
        from .agent_panel.thinking_chain import ThinkingStep, StepStatus
        
        step1 = ThinkingStep(
            id="1",
            title="Analyzing code structure",
            description=f"Reading {Path(target_file).name}",
            status=StepStatus.COMPLETED
        )
        self._agent_panel.get_agent_mode().add_thinking_step(step1)
        self._agent_panel.get_agent_mode().complete_step("1")
        
        step2 = ThinkingStep(
            id="2",
            title="Identifying test cases",
            description="Finding methods and edge cases to test",
            status=StepStatus.RUNNING
        )
        self._agent_panel.get_agent_mode().add_thinking_step(step2)
        
        # Check if LLM client is available
        if not self.llm_client:
            self._agent_panel.add_agent_message(
                "⚠️ LLM not configured. Please configure LLM settings first."
            )
            return
        
        # Start streaming response
        chat_mode = self._agent_panel.get_chat_mode()
        message_id = chat_mode.start_streaming_response()
        
        # Create async task for streaming
        try:
            import qasync
            loop = qasync.QEventLoop.instance()
            self._current_streaming_task = loop.create_task(
                self._stream_test_generation(target_file, chat_mode, message_id)
            )
        except Exception as e:
            logger.exception("Failed to start test generation task")
            chat_mode.finish_streaming()
            chat_mode.update_message(
                message_id,
                f"❌ Error: Failed to start test generation - {str(e)}"
            )
            
    async def _stream_test_generation(
        self, 
        target_file: str, 
        chat_mode, 
        message_id: str
    ):
        """Stream test generation response.
        
        Args:
            target_file: Path to file to generate tests for
            chat_mode: Chat mode widget
            message_id: Message ID for updates
        """
        try:
            # Read file content
            with open(target_file, 'r', encoding='utf-8') as f:
                file_content = f.read()
            
            # Build prompt
            prompt = (
                f"Generate comprehensive unit tests for the following code:\n\n"
                f"File: {Path(target_file).name}\n"
                f"```java\n{file_content}\n```\n\n"
                f"Please provide:\n"
                f"1. Test cases for all public methods\n"
                f"2. Edge case coverage\n"
                f"3. Mock setup if needed\n"
                f"4. Use JUnit 5 and Mockito\n\n"
                f"Output the complete test class code."
            )
            
            system_prompt = (
                "You are an expert Java developer specializing in unit testing.\n"
                "Generate high-quality, well-documented test code.\n"
                "Use Markdown code blocks for the test code."
            )
            
            logger.info(f"Starting test generation stream for: {target_file}")
            
            # Reset cancellation state
            self.llm_client.reset_cancel()
            
            # Stream response
            full_response = ""
            async for chunk in self.llm_client.astream(prompt, system_prompt):
                full_response += chunk
                chat_mode.append_to_streaming(chunk)
                
            # Save to session
            self._session_manager.add_message_to_current('agent', full_response)
            
            logger.info(f"Test generation completed: {len(full_response)} chars")
            
        except asyncio.CancelledError:
            logger.info("Test generation cancelled")
            chat_mode.append_to_streaming("\n\n[Test generation cancelled by user]")
        except Exception as e:
            logger.exception("Test generation failed")
            error_msg = f"\n\n❌ Error: {str(e)}"
            chat_mode.append_to_streaming(error_msg)
        finally:
            chat_mode.finish_streaming()
        
    def on_context_changed(self):
        """Handle context changes."""
        context_manager = self._agent_panel.get_context_manager()
        token_count = context_manager.get_total_tokens()
        item_count = context_manager.get_item_count()
        
        self._main_layout.set_status_progress(
            f"Context: {item_count} files, ~{token_count} tokens"
        )
        
    def on_panel_collapsed(self, panel_name: str, is_collapsed: bool):
        """Handle panel collapse/expand."""
        logger.debug(f"Panel {panel_name} collapsed: {is_collapsed}")
        
    def on_llm_config(self):
        """Handle LLM config action."""
        from .dialogs.llm_config_dialog import LLMConfigDialog
        from PyQt6.QtWidgets import QDialog
        
        try:
            dialog = LLMConfigDialog(self.config_collection, self.aider_config, self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                self.config_collection = dialog.get_config_collection()
                self.aider_config = dialog.aider_config
                save_llm_config(self.config_collection)
                self.setup_llm_client()
        except Exception as e:
            logger.exception("Failed to open LLM config dialog")
            
    def on_command_palette(self):
        """Show command palette."""
        logger.info("Command palette requested")
        from .command_palette import show_command_palette

        # Add semantic search command if not exists
        palette = self._create_command_palette_with_search()
        palette.exec()

    def _create_command_palette_with_search(self):
        """Create command palette with semantic search command."""
        from .command_palette import CommandPalette

        palette = CommandPalette(self)

        # Add semantic search command
        from PyQt6.QtGui import QAction
        from PyQt6.QtWidgets import QDialog

        # Connect to execute semantic search
        original_execute = palette._execute

        def enhanced_execute(action: str):
            if action == "semantic_search":
                self.on_semantic_search()
            elif action == "find_in_files":
                self.on_find_in_files()
            else:
                original_execute(action)

        palette._execute = enhanced_execute

        return palette

    def on_semantic_search(self):
        """Show semantic search dialog."""
        logger.info("Semantic search requested")

        if not self.current_project:
            self._notification_manager.show_warning(
                "Please open a project first",
                duration=3000
            )
            return

        try:
            # Initialize search service if needed
            if self._semantic_search_service is None:
                self._semantic_search_service = SemanticSearchService(
                    project_path=self.current_project
                )

            # Show dialog
            from .dialogs.semantic_search_dialog import SemanticSearchDialog

            dialog = SemanticSearchDialog(
                project_path=self.current_project,
                search_service=self._semantic_search_service,
                parent=self
            )

            # Connect result selection to open file
            dialog.result_selected.connect(self._on_search_result_selected)
            dialog.result_activated.connect(self._on_search_result_activated)

            dialog.exec()

        except Exception as e:
            logger.exception("Failed to open semantic search")
            self._notification_manager.show_error(
                f"Failed to open semantic search: {e}",
                duration=5000
            )

    def _on_search_result_selected(self, file_path: str, line_number: int):
        """Handle search result selection."""
        logger.info(f"Search result selected: {file_path}:{line_number}")
        # File path is already copied to clipboard by the dialog

    def _on_search_result_activated(self, file_path: str, line_number: int):
        """Handle search result activation (double-click)."""
        logger.info(f"Search result activated: {file_path}:{line_number}")

        # Open the file in editor
        if Path(file_path).exists():
            self._content_panel.open_file(file_path)
            # TODO: Navigate to specific line number in editor
            self._notification_manager.show_success(
                f"Opened {Path(file_path).name}:{line_number}",
                duration=2000
            )
        else:
            self._notification_manager.show_warning(
                f"File not found: {file_path}",
                duration=3000
            )

    def on_find_in_files(self):
        """Show find in files dialog."""
        logger.info("Find in files requested")
        # TODO: Implement find in files
        self._notification_manager.show_info(
            "Find in files - Coming soon",
            duration=2000
        )

    def on_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About PyUT Agent",
            "<h2>PyUT Agent</h2>"
            "<p>AI-powered Coding Assistant</p>"
            "<p>Version: 2.0.0</p>"
            "<p>Features:</p>"
            "<ul>"
            "<li>Multi-language support</li>"
            "<li>Intelligent code generation</li>"
            "<li>Test generation</li>"
            "<li>Code explanation</li>"
            "<li>Refactoring assistance</li>"
            "<li>Streaming AI responses</li>"
            "<li>Markdown rendering</li>"
            "</ul>"
        )

    def on_review_changes(self):
        """Show review changes dialog."""
        logger.info("Review changes requested")
        dialog = ApprovalDialog(self, "Review Changes")
        if dialog.exec():
            approved = dialog.get_approved_files()
            rejected = dialog.get_rejected_files()
            logger.info(f"Review completed: {len(approved)} approved, {len(rejected)} rejected")

    def on_show_terminal(self):
        """Show integrated terminal panel."""
        logger.info("Terminal requested")
        if not hasattr(self, '_terminal_dialog') or self._terminal_dialog is None:
            self._terminal_dialog = QDialog(self)
            self._terminal_dialog.setWindowTitle("Terminal")
            self._terminal_dialog.setMinimumSize(800, 500)

            layout = QVBoxLayout(self._terminal_dialog)
            layout.setContentsMargins(0, 0, 0, 0)

            self._terminal_widget = TerminalWidget(cwd=self.current_project or None)
            layout.addWidget(self._terminal_widget)

        self._terminal_dialog.show()
        self._terminal_dialog.raise_()
        self._terminal_dialog.activateWindow()
        
    def closeEvent(self, event):
        """Handle window close."""
        # Cancel any ongoing streaming
        if self._is_streaming and self.llm_client:
            self.llm_client.cancel()
            
        # Save all sessions
        self._session_manager.save_all_sessions()
        
        # Save layout state
        # TODO: Save to app state
        
        event.accept()
