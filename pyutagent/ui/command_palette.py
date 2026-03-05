"""Command palette for quick access to all features."""

import logging
import re
from typing import List, Callable, Optional, Tuple
from dataclasses import dataclass
from difflib import SequenceMatcher

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QAbstractItemView, QWidget,
    QStyledItemDelegate, QStyleOptionViewItem, QStyle
)
from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtGui import QKeyEvent, QColor, QFont, QPalette, QFontMetrics

from .styles import get_style_manager

logger = logging.getLogger(__name__)


@dataclass
class Command:
    """A command in the command palette."""
    id: str
    name: str
    description: str
    shortcut: str
    category: str
    callback: Callable
    icon: str = ""
    keywords: List[str] = None

    def __post_init__(self):
        if self.keywords is None:
            self.keywords = []


class CommandItemDelegate(QStyledItemDelegate):
    """Custom delegate for rendering command items with highlighting."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._highlight_text = ""
        self._is_dark = True

    def set_highlight_text(self, text: str):
        """Set the text to highlight."""
        self._highlight_text = text.lower()

    def set_dark_mode(self, is_dark: bool):
        """Set dark mode."""
        self._is_dark = is_dark

    def paint(self, painter, option, index):
        """Paint the item with custom styling."""
        # Get the command data
        cmd = index.data(Qt.ItemDataRole.UserRole)
        if not cmd:
            super().paint(painter, option, index)
            return

        # Save painter state
        painter.save()

        # Get colors based on theme and selection state
        is_selected = option.state & QStyle.StateFlag.State_Selected
        is_hover = option.state & QStyle.StateFlag.State_MouseOver

        if self._is_dark:
            bg_color = QColor("#094771") if is_selected else (QColor("#3C3C3C") if is_hover else QColor("#2D2D2D"))
            text_color = QColor("#E0E0E0")
            shortcut_color = QColor("#808080")
            desc_color = QColor("#A0A0A0")
            highlight_color = QColor("#FFD700")
        else:
            bg_color = QColor("#E3F2FD") if is_selected else (QColor("#F5F5F5") if is_hover else QColor("#FFFFFF"))
            text_color = QColor("#212121")
            shortcut_color = QColor("#757575")
            desc_color = QColor("#616161")
            highlight_color = QColor("#1976D2")

        # Fill background
        painter.fillRect(option.rect, bg_color)

        # Calculate layout
        padding = 8
        text_x = option.rect.left() + padding
        text_y = option.rect.top() + padding
        available_width = option.rect.width() - 2 * padding

        # Draw category badge
        category_font = QFont(painter.font())
        category_font.setPointSize(8)
        painter.setFont(category_font)
        painter.setPen(desc_color)

        category_text = cmd.category
        category_width = painter.fontMetrics().horizontalAdvance(category_text)
        painter.drawText(text_x, text_y + 12, category_text)

        # Draw shortcut on the right
        if cmd.shortcut:
            painter.setPen(shortcut_color)
            shortcut_text = cmd.shortcut
            shortcut_width = painter.fontMetrics().horizontalAdvance(shortcut_text)
            painter.drawText(
                option.rect.right() - padding - shortcut_width,
                text_y + 12,
                shortcut_text
            )

        # Draw command name with highlighting
        name_font = QFont(painter.font())
        name_font.setPointSize(11)
        name_font.setBold(True)
        painter.setFont(name_font)

        name_y = text_y + 28
        self._draw_highlighted_text(
            painter, text_x, name_y, cmd.name,
            text_color, highlight_color,
            option.rect.right() - padding - (shortcut_width if cmd.shortcut else 0) - text_x - 10
        )

        # Draw description
        desc_font = QFont(painter.font())
        desc_font.setPointSize(9)
        desc_font.setBold(False)
        painter.setFont(desc_font)
        painter.setPen(desc_color)

        desc_y = text_y + 46
        desc_metrics = QFontMetrics(desc_font)
        desc_elided = desc_metrics.elidedText(
            cmd.description, Qt.TextElideMode.ElideRight,
            available_width
        )
        painter.drawText(text_x, desc_y, desc_elided)

        # Restore painter state
        painter.restore()

    def _draw_highlighted_text(self, painter, x, y, text, text_color, highlight_color, max_width):
        """Draw text with highlight for matching characters."""
        if not self._highlight_text:
            painter.setPen(text_color)
            painter.drawText(x, y, text)
            return

        # Find matching positions
        text_lower = text.lower()
        highlight_positions = set()

        # Simple character matching for highlight
        search_chars = list(self._highlight_text)
        text_chars = list(text_lower)

        search_idx = 0
        for i, char in enumerate(text_chars):
            if search_idx < len(search_chars) and char == search_chars[search_idx]:
                highlight_positions.add(i)
                search_idx += 1

        # Draw each character
        current_x = x
        font_metrics = painter.fontMetrics()

        for i, char in enumerate(text):
            char_width = font_metrics.horizontalAdvance(char)

            if current_x + char_width > x + max_width:
                # Draw ellipsis
                painter.setPen(text_color)
                painter.drawText(current_x, y, "...")
                break

            if i in highlight_positions:
                painter.setPen(highlight_color)
                # Draw underline for highlighted chars
                painter.drawLine(
                    current_x, y + 2,
                    current_x + char_width, y + 2
                )
            else:
                painter.setPen(text_color)

            painter.drawText(current_x, y, char)
            current_x += char_width

    def sizeHint(self, option, index):
        """Return the size hint for the item."""
        return QSize(option.rect.width(), 70)


class FuzzyMatcher:
    """Fuzzy matching algorithm for command search."""

    @staticmethod
    def calculate_score(query: str, command: Command) -> Tuple[float, List[int]]:
        """
        Calculate fuzzy match score for a command.

        Returns:
            Tuple of (score, match_positions) where higher score is better.
        """
        query = query.lower().strip()
        if not query:
            return (1.0, [])

        scores = []
        match_positions = []

        # Check name match
        name_score, name_positions = FuzzyMatcher._match_text(query, command.name)
        scores.append((name_score * 2.0, name_positions))  # Name has higher weight

        # Check description match
        desc_score, desc_positions = FuzzyMatcher._match_text(query, command.description)
        scores.append((desc_score * 1.0, desc_positions))

        # Check category match
        cat_score, cat_positions = FuzzyMatcher._match_text(query, command.category)
        scores.append((cat_score * 1.5, cat_positions))  # Category has medium weight

        # Check keywords
        for keyword in command.keywords:
            kw_score, kw_positions = FuzzyMatcher._match_text(query, keyword)
            scores.append((kw_score * 1.3, kw_positions))

        # Check shortcut match
        if command.shortcut:
            shortcut_lower = command.shortcut.lower()
            if query in shortcut_lower:
                scores.append((1.5, []))

        # Return best score
        if scores:
            best = max(scores, key=lambda x: x[0])
            return best

        return (0.0, [])

    @staticmethod
    def _match_text(query: str, text: str) -> Tuple[float, List[int]]:
        """
        Match query against text using fuzzy algorithm.

        Returns:
            Tuple of (score, match_positions)
        """
        text_lower = text.lower()

        # Exact match
        if query == text_lower:
            return (3.0, list(range(len(text))))

        # Starts with
        if text_lower.startswith(query):
            return (2.5, list(range(len(query))))

        # Contains as substring
        if query in text_lower:
            start = text_lower.index(query)
            return (2.0, list(range(start, start + len(query))))

        # Fuzzy match - find characters in order
        query_chars = list(query)
        text_chars = list(text_lower)

        positions = []
        query_idx = 0
        last_match_pos = -1
        consecutive_bonus = 0

        for text_idx, char in enumerate(text_chars):
            if query_idx < len(query_chars) and char == query_chars[query_idx]:
                positions.append(text_idx)

                # Bonus for consecutive matches
                if text_idx == last_match_pos + 1:
                    consecutive_bonus += 0.1

                last_match_pos = text_idx
                query_idx += 1

        if query_idx == len(query_chars):
            # All characters matched
            coverage = len(positions) / len(text) if text else 0
            score = 1.0 + consecutive_bonus - coverage * 0.5
            return (max(0.1, score), positions)

        # Partial match
        if query_idx > 0:
            ratio = query_idx / len(query_chars)
            return (ratio * 0.5, positions)

        return (0.0, [])


class CommandPalette(QDialog):
    """Command palette dialog for quick access to features."""

    command_executed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Command Palette")
        self.setMinimumSize(700, 500)

        self._style_manager = get_style_manager()
        self._commands: List[Command] = []
        self._filtered_commands: List[Tuple[Command, float]] = []  # (command, score)
        self._last_command: Optional[str] = None
        self._fuzzy_matcher = FuzzyMatcher()

        self.setup_ui()
        self.setup_commands()
        self.apply_styles()

    def setup_ui(self):
        """Setup the UI."""
        self.setWindowFlags(
            Qt.WindowType.Dialog |
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)

        # Search box
        search_container = QWidget()
        search_layout = QHBoxLayout(search_container)
        search_layout.setContentsMargins(0, 0, 0, 0)

        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Type a command... (e.g., 'open', 'search', 'generate')")
        self.search_box.textChanged.connect(self.on_search_changed)
        search_layout.addWidget(self.search_box)

        # Shortcut hint
        shortcut_hint = QLabel("⌘K")
        shortcut_hint.setStyleSheet("color: #808080; font-size: 12px;")
        search_layout.addWidget(shortcut_hint)

        layout.addWidget(search_container)

        # Results count label
        self.results_label = QLabel("All commands")
        self.results_label.setStyleSheet("color: #808080; font-size: 11px;")
        layout.addWidget(self.results_label)

        # Results list
        self.results_list = QListWidget()
        self.results_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.results_list.itemActivated.connect(self.on_item_activated)
        self.results_list.itemClicked.connect(self.on_item_activated)

        # Set custom delegate
        self._item_delegate = CommandItemDelegate(self)
        self._item_delegate.set_dark_mode(self._style_manager.current_theme == "dark")
        self.results_list.setItemDelegate(self._item_delegate)

        layout.addWidget(self.results_list)

        # Hint label
        hint = QLabel("↑↓ to navigate, ↵ to execute, Esc to close")
        hint.setStyleSheet("color: #808080; font-size: 11px;")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(hint)

        # Set focus to search box
        self.search_box.setFocus()

    def setup_commands(self):
        """Setup available commands."""
        self._commands = [
            # File commands
            Command(
                "file.open",
                "Open Project",
                "Open a project directory",
                "Ctrl+O",
                "File",
                lambda: self._execute("open_project"),
                keywords=["open", "project", "folder", "directory", "load"]
            ),
            Command(
                "file.new_session",
                "New Session",
                "Create a new chat session",
                "Ctrl+N",
                "File",
                lambda: self._execute("new_session"),
                keywords=["new", "session", "chat", "clear", "start"]
            ),
            Command(
                "file.session_history",
                "Session History",
                "View and restore previous sessions",
                "Ctrl+H",
                "File",
                lambda: self._execute("session_history"),
                keywords=["history", "session", "previous", "restore", "past"]
            ),
            Command(
                "file.exit",
                "Exit Application",
                "Close the application",
                "Alt+F4",
                "File",
                lambda: self._execute("exit"),
                keywords=["exit", "quit", "close", "shutdown"]
            ),

            # Edit commands
            Command(
                "edit.undo",
                "Undo",
                "Undo last action",
                "Ctrl+Z",
                "Edit",
                lambda: self._execute("undo"),
                keywords=["undo", "revert", "back"]
            ),
            Command(
                "edit.redo",
                "Redo",
                "Redo last undone action",
                "Ctrl+Y",
                "Edit",
                lambda: self._execute("redo"),
                keywords=["redo", "restore", "forward"]
            ),
            Command(
                "edit.copy",
                "Copy",
                "Copy selected text",
                "Ctrl+C",
                "Edit",
                lambda: self._execute("copy"),
                keywords=["copy", "clipboard", "duplicate"]
            ),
            Command(
                "edit.paste",
                "Paste",
                "Paste from clipboard",
                "Ctrl+V",
                "Edit",
                lambda: self._execute("paste"),
                keywords=["paste", "clipboard", "insert"]
            ),

            # View commands
            Command(
                "view.toggle_sidebar",
                "Toggle Sidebar",
                "Show or hide the sidebar panel",
                "Ctrl+B",
                "View",
                lambda: self._execute("toggle_sidebar"),
                keywords=["sidebar", "panel", "tree", "files", "toggle", "show", "hide"]
            ),
            Command(
                "view.toggle_agent",
                "Toggle Agent Panel",
                "Show or hide the AI agent panel",
                "Ctrl+J",
                "View",
                lambda: self._execute("toggle_agent"),
                keywords=["agent", "panel", "chat", "ai", "toggle", "show", "hide"]
            ),
            Command(
                "view.terminal",
                "Show Terminal",
                "Open integrated terminal",
                "Ctrl+`",
                "View",
                lambda: self._execute("terminal"),
                keywords=["terminal", "console", "shell", "command line", "cmd"]
            ),
            Command(
                "view.layout_default",
                "Default Layout",
                "Restore default panel layout",
                "",
                "View",
                lambda: self._execute("layout_default"),
                keywords=["layout", "default", "reset", "panels"]
            ),
            Command(
                "view.layout_focus_editor",
                "Focus Editor",
                "Maximize editor panel",
                "",
                "View",
                lambda: self._execute("layout_focus_editor"),
                keywords=["layout", "focus", "editor", "code", "maximize"]
            ),
            Command(
                "view.layout_focus_agent",
                "Focus Agent",
                "Maximize agent panel",
                "",
                "View",
                lambda: self._execute("layout_focus_agent"),
                keywords=["layout", "focus", "agent", "chat", "maximize"]
            ),

            # Search commands
            Command(
                "search.semantic",
                "Semantic Search",
                "Search code using natural language",
                "Ctrl+Shift+F",
                "Search",
                lambda: self._execute("semantic_search"),
                keywords=["search", "semantic", "natural language", "find", "ai search"]
            ),
            Command(
                "search.files",
                "Find in Files",
                "Search text across all project files",
                "Ctrl+Shift+H",
                "Search",
                lambda: self._execute("find_in_files"),
                keywords=["search", "find", "files", "text", "grep", "global"]
            ),
            Command(
                "search.command_palette",
                "Command Palette",
                "Quick access to all commands",
                "Ctrl+Shift+P",
                "Search",
                lambda: self._execute("command_palette"),
                keywords=["command", "palette", "commands", "quick", "access"]
            ),

            # AI / Generate commands
            Command(
                "ai.generate_tests",
                "Generate Tests",
                "Generate unit tests for selected file",
                "Ctrl+G",
                "AI",
                lambda: self._execute("generate_tests"),
                keywords=["generate", "test", "unit test", "testing", "ai"]
            ),
            Command(
                "ai.explain",
                "Explain Code",
                "Get AI explanation of selected code",
                "",
                "AI",
                lambda: self._execute("explain_code"),
                keywords=["explain", "code", "understand", "documentation", "ai"]
            ),
            Command(
                "ai.refactor",
                "Refactor Code",
                "Get AI refactoring suggestions",
                "",
                "AI",
                lambda: self._execute("refactor_code"),
                keywords=["refactor", "improve", "optimize", "clean", "ai"]
            ),
            Command(
                "ai.document",
                "Generate Documentation",
                "Generate documentation for code",
                "",
                "AI",
                lambda: self._execute("generate_docs"),
                keywords=["document", "docs", "javadoc", "comment", "ai"]
            ),
            Command(
                "ai.review",
                "Review Code",
                "Get AI code review",
                "Ctrl+Shift+R",
                "AI",
                lambda: self._execute("review_code"),
                keywords=["review", "code review", "check", "quality", "ai"]
            ),
            Command(
                "ai.fix_terminal",
                "Fix Terminal Error",
                "Ask AI to fix terminal errors",
                "",
                "AI",
                lambda: self._execute("fix_terminal"),
                keywords=["fix", "error", "terminal", "debug", "ai"]
            ),

            # Settings commands
            Command(
                "settings.llm",
                "LLM Settings",
                "Configure LLM provider and API keys",
                "Ctrl+,",
                "Settings",
                lambda: self._execute("llm_config"),
                keywords=["settings", "llm", "ai", "api", "key", "config"]
            ),
            Command(
                "settings.shortcuts",
                "Keyboard Shortcuts",
                "View and customize keyboard shortcuts",
                "Ctrl+/",
                "Settings",
                lambda: self._execute("shortcuts"),
                keywords=["shortcuts", "keyboard", "keybinding", "hotkey", "settings"]
            ),
            Command(
                "settings.coverage",
                "Coverage Settings",
                "Configure code coverage settings",
                "",
                "Settings",
                lambda: self._execute("coverage_config"),
                keywords=["coverage", "jacoco", "test coverage", "settings"]
            ),
            Command(
                "settings.jdk",
                "JDK Settings",
                "Configure Java JDK path",
                "",
                "Settings",
                lambda: self._execute("jdk_config"),
                keywords=["jdk", "java", "sdk", "path", "settings"]
            ),
            Command(
                "settings.maven",
                "Maven Settings",
                "Configure Maven path",
                "",
                "Settings",
                lambda: self._execute("maven_config"),
                keywords=["maven", "build", "path", "settings"]
            ),

            # Theme commands
            Command(
                "theme.light",
                "Light Theme",
                "Switch to light color theme",
                "",
                "Theme",
                lambda: self._execute("theme_light"),
                keywords=["theme", "light", "white", "color", "appearance"]
            ),
            Command(
                "theme.dark",
                "Dark Theme",
                "Switch to dark color theme",
                "",
                "Theme",
                lambda: self._execute("theme_dark"),
                keywords=["theme", "dark", "black", "color", "appearance"]
            ),

            # Help commands
            Command(
                "help.slash_commands",
                "Slash Commands Help",
                "Show available slash commands",
                "",
                "Help",
                lambda: self._execute("slash_commands_help"),
                keywords=["help", "slash", "commands", "/", "chat commands"]
            ),
            Command(
                "help.about",
                "About",
                "Show application information",
                "",
                "Help",
                lambda: self._execute("about"),
                keywords=["about", "info", "version", "credits"]
            ),
        ]

        self._filtered_commands = [(cmd, 1.0) for cmd in self._commands]
        self.update_results()

    def apply_styles(self):
        """Apply theme styles."""
        is_dark = self._style_manager.current_theme == "dark"
        self._item_delegate.set_dark_mode(is_dark)

        bg_color = "#2D2D2D" if is_dark else "#FFFFFF"
        text_color = "#E0E0E0" if is_dark else "#212121"
        border_color = "#3C3C3C" if is_dark else "#E0E0E0"
        input_bg = "#1E1E1E" if is_dark else "#F5F5F5"

        self.setStyleSheet(f"""
            CommandPalette {{
                background-color: {bg_color};
                border: 1px solid {border_color};
                border-radius: 8px;
            }}
            QLineEdit {{
                background-color: {input_bg};
                color: {text_color};
                border: 1px solid {border_color};
                border-radius: 4px;
                padding: 10px;
                font-size: 14px;
            }}
            QListWidget {{
                background-color: {bg_color};
                color: {text_color};
                border: none;
                outline: none;
            }}
            QListWidget::item {{
                padding: 0px;
                border-radius: 4px;
            }}
        """)

    def on_search_changed(self, text: str):
        """Handle search text change with fuzzy matching."""
        text = text.strip()

        if not text:
            # Show all commands sorted by category
            self._filtered_commands = [(cmd, 1.0) for cmd in self._commands]
            self._filtered_commands.sort(key=lambda x: (x[0].category, x[0].name))
            self.results_label.setText(f"All {len(self._commands)} commands")
        else:
            # Calculate fuzzy match scores
            scored_commands = []
            for cmd in self._commands:
                score, positions = self._fuzzy_matcher.calculate_score(text, cmd)
                if score > 0.1:  # Minimum threshold
                    scored_commands.append((cmd, score))

            # Sort by score (descending)
            scored_commands.sort(key=lambda x: x[1], reverse=True)
            self._filtered_commands = scored_commands

            match_count = len(self._filtered_commands)
            self.results_label.setText(
                f"{match_count} result{'s' if match_count != 1 else ''} for '{text}'"
            )

        # Update delegate highlight text
        self._item_delegate.set_highlight_text(text)

        self.update_results()

    def update_results(self):
        """Update the results list."""
        self.results_list.clear()

        # Limit to top 20 results
        for cmd, score in self._filtered_commands[:20]:
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, cmd)
            item.setSizeHint(QSize(self.results_list.width(), 70))
            self.results_list.addItem(item)

        # Select first item
        if self.results_list.count() > 0:
            self.results_list.setCurrentRow(0)

    def on_item_activated(self, item: QListWidgetItem):
        """Handle item activation."""
        cmd = item.data(Qt.ItemDataRole.UserRole)
        if cmd:
            self.execute_command(cmd)

    def execute_command(self, cmd: Command):
        """Execute a command."""
        try:
            self._last_command = cmd.id
            cmd.callback()
            self.command_executed.emit(cmd.id)
            self.accept()
        except Exception as e:
            logger.error(f"Failed to execute command {cmd.id}: {e}")

    def _execute(self, action: str):
        """Execute an action on the parent window."""
        parent = self.parent()
        if not parent:
            return

        try:
            # File actions
            if action == "open_project":
                parent.on_open_project()
            elif action == "new_session":
                parent._create_new_session()
            elif action == "session_history":
                parent._show_session_history()
            elif action == "exit":
                parent.close()

            # Edit actions
            elif action == "undo":
                parent._content_panel.undo() if hasattr(parent, '_content_panel') else None
            elif action == "redo":
                parent._content_panel.redo() if hasattr(parent, '_content_panel') else None
            elif action == "copy":
                parent._content_panel.copy() if hasattr(parent, '_content_panel') else None
            elif action == "paste":
                parent._content_panel.paste() if hasattr(parent, '_content_panel') else None

            # View actions
            elif action == "toggle_sidebar":
                from .layout import MainLayout
                parent._main_layout.toggle_panel(MainLayout.SIDEBAR_PANEL)
            elif action == "toggle_agent":
                from .layout import MainLayout
                parent._main_layout.toggle_panel(MainLayout.AGENT_PANEL)
            elif action == "terminal":
                parent.on_show_terminal()
            elif action == "layout_default":
                from .layout import MainLayout
                parent._main_layout.set_layout_mode(MainLayout.MODE_DEFAULT)
            elif action == "layout_focus_editor":
                from .layout import MainLayout
                parent._main_layout.set_layout_mode(MainLayout.MODE_FOCUS_EDITOR)
            elif action == "layout_focus_agent":
                from .layout import MainLayout
                parent._main_layout.set_layout_mode(MainLayout.MODE_FOCUS_AGENT)

            # Search actions
            elif action == "semantic_search":
                parent.on_semantic_search()
            elif action == "find_in_files":
                parent.on_find_in_files()
            elif action == "command_palette":
                pass  # Already in command palette

            # AI actions
            elif action == "generate_tests":
                parent.on_generate_tests()
            elif action == "explain_code":
                parent._handle_explain_command("")
            elif action == "refactor_code":
                parent._handle_refactor_command("")
            elif action == "generate_docs":
                parent._handle_doc_command("")
            elif action == "review_code":
                parent.on_review_changes()
            elif action == "fix_terminal":
                parent.on_show_terminal()  # Open terminal first

            # Settings actions
            elif action == "llm_config":
                parent.on_llm_config()
            elif action == "shortcuts":
                parent._show_shortcuts_dialog()
            elif action == "coverage_config":
                parent.on_coverage_config() if hasattr(parent, 'on_coverage_config') else None
            elif action == "jdk_config":
                parent.on_jdk_config() if hasattr(parent, 'on_jdk_config') else None
            elif action == "maven_config":
                parent.on_maven_config() if hasattr(parent, 'on_maven_config') else None

            # Theme actions
            elif action == "theme_light":
                parent.on_theme_changed("light") if hasattr(parent, 'on_theme_changed') else None
            elif action == "theme_dark":
                parent.on_theme_changed("dark") if hasattr(parent, 'on_theme_changed') else None

            # Help actions
            elif action == "slash_commands_help":
                parent._show_slash_commands_help()
            elif action == "about":
                parent.on_about()

        except Exception as e:
            logger.error(f"Failed to execute action {action}: {e}")

    def keyPressEvent(self, event: QKeyEvent):
        """Handle key press events."""
        if event.key() == Qt.Key.Key_Escape:
            self.reject()
        elif event.key() == Qt.Key.Key_Return or event.key() == Qt.Key.Key_Enter:
            current_item = self.results_list.currentItem()
            if current_item:
                self.on_item_activated(current_item)
        elif event.key() == Qt.Key.Key_Down:
            current_row = self.results_list.currentRow()
            if current_row < self.results_list.count() - 1:
                self.results_list.setCurrentRow(current_row + 1)
        elif event.key() == Qt.Key.Key_Up:
            current_row = self.results_list.currentRow()
            if current_row > 0:
                self.results_list.setCurrentRow(current_row - 1)
        elif event.key() == Qt.Key.Key_PageDown:
            current_row = self.results_list.currentRow()
            new_row = min(current_row + 5, self.results_list.count() - 1)
            self.results_list.setCurrentRow(new_row)
        elif event.key() == Qt.Key.Key_PageUp:
            current_row = self.results_list.currentRow()
            new_row = max(current_row - 5, 0)
            self.results_list.setCurrentRow(new_row)
        elif event.key() == Qt.Key.Key_Home:
            self.results_list.setCurrentRow(0)
        elif event.key() == Qt.Key.Key_End:
            self.results_list.setCurrentRow(self.results_list.count() - 1)
        else:
            super().keyPressEvent(event)


def show_command_palette(parent=None) -> Optional[str]:
    """Show the command palette.

    Args:
        parent: Parent widget

    Returns:
        The executed command ID or None if cancelled
    """
    palette = CommandPalette(parent)

    # Center on parent
    if parent:
        rect = parent.geometry()
        palette.move(
            rect.center().x() - palette.width() // 2,
            rect.center().y() - palette.height() // 2
        )

    result = palette.exec()

    if result == QDialog.DialogCode.Accepted:
        # Get the last executed command
        return getattr(palette, '_last_command', None)

    return None
