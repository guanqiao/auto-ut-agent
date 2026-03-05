"""Semantic search dialog for natural language code search.

This dialog provides:
- Natural language search input
- Real-time search results display
- Code preview panel
- Result filtering and sorting
"""

import logging
from typing import Optional, List
from pathlib import Path

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QListWidget, QListWidgetItem, QTextEdit,
    QSplitter, QWidget, QFrame, QProgressBar, QComboBox,
    QAbstractItemView, QMessageBox, QMenu, QApplication
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QAction, QFont, QKeyEvent

from pyutagent.ui.services.semantic_search import SemanticSearchService, SearchResult
from pyutagent.ui.styles import get_style_manager

logger = logging.getLogger(__name__)


class SearchResultItem(QListWidgetItem):
    """Custom list item for search results."""

    def __init__(self, result: SearchResult):
        super().__init__()
        self.result = result

        # Format display text
        relevance = result.relevance_percentage
        symbol_info = f" [{result.symbol_type}]" if result.symbol_type else ""
        display_text = f"{result.file_name}{symbol_info}\n{result.location} • {relevance}% match"

        self.setText(display_text)
        self.setToolTip(f"{result.file_path}\n{result.content[:200]}...")

        # Set icon based on file type
        self._set_icon()

    def _set_icon(self):
        """Set icon based on file extension."""
        ext = Path(self.result.file_path).suffix.lower()

        icon_map = {
            '.py': '🐍',
            '.java': '☕',
            '.js': '📜',
            '.ts': '🔷',
            '.go': '🐹',
            '.rs': '🦀',
            '.cpp': '⚙️',
            '.c': '⚙️',
            '.h': '📋',
        }

        icon = icon_map.get(ext, '📄')
        self.setText(f"{icon} {self.text()}")


class CodePreviewWidget(QTextEdit):
    """Widget for displaying code preview with syntax highlighting."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self.setFont(QFont("Consolas", 10))
        self.setStyleSheet("""
            QTextEdit {
                background-color: #1E1E1E;
                color: #D4D4D4;
                border: 1px solid #3C3C3C;
                padding: 8px;
            }
        """)

    def show_result(self, result: Optional[SearchResult]):
        """Display a search result."""
        if result is None:
            self.clear()
            self.setPlaceholderText("Select a result to preview")
            return

        # Build preview content
        lines = []
        lines.append(f"📁 {result.file_path}")
        lines.append(f"📍 Lines {result.start_line}-{result.end_line} • {result.relevance_percentage}% relevance")

        if result.symbol_name:
            lines.append(f"🏷️ Symbol: {result.symbol_name} ({result.symbol_type})")

        lines.append("")
        lines.append("─" * 60)
        lines.append("")

        if result.context:
            lines.append(result.context)
        else:
            lines.append(result.content)

        self.setPlainText('\n'.join(lines))

        # Highlight the result lines
        self._highlight_result()

    def _highlight_result(self):
        """Highlight the main result lines in the preview."""
        # Simple highlighting - in a real implementation,
        # you'd use a proper syntax highlighter
        pass


class SemanticSearchDialog(QDialog):
    """Dialog for semantic code search.

    Features:
    - Natural language search input
    - Real-time search with debouncing
    - Results list with relevance scores
    - Code preview panel
    - Keyboard navigation
    """

    result_selected = pyqtSignal(str, int)  # file_path, line_number
    result_activated = pyqtSignal(str, int)  # file_path, line_number (double-click)

    def __init__(
        self,
        project_path: str,
        search_service: Optional[SemanticSearchService] = None,
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)
        self.project_path = project_path
        self.search_service = search_service
        self._current_results: List[SearchResult] = []
        self._selected_result: Optional[SearchResult] = None
        self._search_timer: Optional[QTimer] = None

        self.setWindowTitle("Semantic Search")
        self.setMinimumSize(1000, 700)

        self._style_manager = get_style_manager()

        self.setup_ui()
        self.apply_styles()
        self.setup_connections()

        # Initialize search service if not provided
        if self.search_service is None:
            self._init_search_service()

    def setup_ui(self):
        """Setup the dialog UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        # Header
        header_layout = QHBoxLayout()

        title = QLabel("🔍 Semantic Search")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        header_layout.addWidget(title)

        header_layout.addStretch()

        # Index status
        self._index_status = QLabel("📊 Not indexed")
        self._index_status.setStyleSheet("color: #858585;")
        header_layout.addWidget(self._index_status)

        layout.addLayout(header_layout)

        # Search input area
        search_layout = QHBoxLayout()

        self._search_input = QLineEdit()
        self._search_input.setPlaceholderText(
            "Search code using natural language (e.g., 'find authentication functions')..."
        )
        self._search_input.setMinimumHeight(32)
        search_layout.addWidget(self._search_input, stretch=1)

        # Filter dropdown
        self._filter_combo = QComboBox()
        self._filter_combo.addItems([
            "All Types",
            "Functions/Methods",
            "Classes",
            "Variables/Fields"
        ])
        self._filter_combo.setMaximumWidth(150)
        search_layout.addWidget(self._filter_combo)

        # Search button
        self._search_btn = QPushButton("🔍 Search")
        self._search_btn.setMaximumWidth(100)
        self._search_btn.clicked.connect(self._on_search_clicked)
        search_layout.addWidget(self._search_btn)

        layout.addLayout(search_layout)

        # Progress bar (hidden by default)
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setFormat("Searching... %p%")
        self._progress_bar.hide()
        layout.addWidget(self._progress_bar)

        # Main content splitter
        self._splitter = QSplitter(Qt.Orientation.Horizontal)

        # Results panel
        results_widget = QWidget()
        results_layout = QVBoxLayout(results_widget)
        results_layout.setContentsMargins(0, 0, 0, 0)

        # Results header
        results_header = QHBoxLayout()
        self._results_label = QLabel("Results: 0")
        results_header.addWidget(self._results_label)

        results_header.addStretch()

        # Sort dropdown
        self._sort_combo = QComboBox()
        self._sort_combo.addItems(["Relevance", "File Name", "Location"])
        self._sort_combo.currentTextChanged.connect(self._on_sort_changed)
        results_header.addWidget(QLabel("Sort by:"))
        results_header.addWidget(self._sort_combo)

        results_layout.addLayout(results_header)

        # Results list
        self._results_list = QListWidget()
        self._results_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._results_list.itemSelectionChanged.connect(self._on_selection_changed)
        self._results_list.itemDoubleClicked.connect(self._on_item_double_clicked)
        self._results_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._results_list.customContextMenuRequested.connect(self._on_context_menu)
        self._results_list.setStyleSheet("""
            QListWidget {
                background-color: #1E1E1E;
                border: 1px solid #3C3C3C;
                outline: none;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #2D2D2D;
            }
            QListWidget::item:selected {
                background-color: #094771;
            }
            QListWidget::item:hover {
                background-color: #2D2D2D;
            }
        """)
        results_layout.addWidget(self._results_list)

        self._splitter.addWidget(results_widget)

        # Preview panel
        preview_widget = QWidget()
        preview_layout = QVBoxLayout(preview_widget)
        preview_layout.setContentsMargins(0, 0, 0, 0)

        preview_header = QLabel("👁️ Preview")
        preview_header.setStyleSheet("font-weight: bold; padding: 4px;")
        preview_layout.addWidget(preview_header)

        self._preview = CodePreviewWidget()
        preview_layout.addWidget(self._preview)

        # Action buttons for preview
        preview_actions = QHBoxLayout()

        self._btn_open_file = QPushButton("📂 Open File")
        self._btn_open_file.clicked.connect(self._on_open_file)
        self._btn_open_file.setEnabled(False)
        preview_actions.addWidget(self._btn_open_file)

        self._btn_copy_path = QPushButton("📋 Copy Path")
        self._btn_copy_path.clicked.connect(self._on_copy_path)
        self._btn_copy_path.setEnabled(False)
        preview_actions.addWidget(self._btn_copy_path)

        preview_actions.addStretch()

        preview_layout.addLayout(preview_actions)

        self._splitter.addWidget(preview_widget)

        # Set splitter sizes (40% results, 60% preview)
        self._splitter.setSizes([400, 600])

        layout.addWidget(self._splitter, stretch=1)

        # Bottom buttons
        button_layout = QHBoxLayout()

        self._btn_index = QPushButton("🔄 Index Project")
        self._btn_index.clicked.connect(self._on_index_project)
        button_layout.addWidget(self._btn_index)

        button_layout.addStretch()

        self._btn_close = QPushButton("Close")
        self._btn_close.clicked.connect(self.close)
        button_layout.addWidget(self._btn_close)

        layout.addLayout(button_layout)

        # Help text
        help_label = QLabel(
            "💡 Tips: Use natural language like 'find authentication functions' or 'how to handle errors'. "
            "Press Enter to search, ↑↓ to navigate, ↵ to open."
        )
        help_label.setStyleSheet("color: #858585; font-size: 11px;")
        help_label.setWordWrap(True)
        layout.addWidget(help_label)

    def apply_styles(self):
        """Apply theme styles."""
        is_dark = self._style_manager.current_theme == "dark"

        bg_color = "#2D2D2D" if is_dark else "#FFFFFF"
        text_color = "#E0E0E0" if is_dark else "#212121"
        border_color = "#3C3C3C" if is_dark else "#E0E0E0"

        self.setStyleSheet(f"""
            QDialog {{
                background-color: {bg_color};
            }}
            QLineEdit {{
                background-color: {"#1E1E1E" if is_dark else "#F5F5F5"};
                color: {text_color};
                border: 1px solid {border_color};
                border-radius: 4px;
                padding: 8px;
                font-size: 13px;
            }}
            QPushButton {{
                background-color: {"#0E639C" if is_dark else "#2196F3"};
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
            }}
            QPushButton:hover {{
                background-color: {"#1177BB" if is_dark else "#42A5F5"};
            }}
            QPushButton:disabled {{
                background-color: {"#3C3C3C" if is_dark else "#E0E0E0"};
                color: {"#858585" if is_dark else "#9E9E9E"};
            }}
            QComboBox {{
                background-color: {"#1E1E1E" if is_dark else "#F5F5F5"};
                color: {text_color};
                border: 1px solid {border_color};
                padding: 4px;
            }}
            QLabel {{
                color: {text_color};
            }}
        """)

    def setup_connections(self):
        """Setup signal connections."""
        # Debounced search on text change
        self._search_timer = QTimer(self)
        self._search_timer.setSingleShot(True)
        self._search_timer.timeout.connect(self._perform_search)

        self._search_input.textChanged.connect(self._on_search_text_changed)

        # Filter change triggers re-sort
        self._filter_combo.currentTextChanged.connect(self._on_filter_changed)

    def _init_search_service(self):
        """Initialize the search service."""
        try:
            self.search_service = SemanticSearchService(
                project_path=self.project_path
            )

            # Connect signals
            self.search_service.search_started.connect(self._on_search_started)
            self.search_service.search_completed.connect(self._on_search_completed)
            self.search_service.search_error.connect(self._on_search_error)
            self.search_service.progress_updated.connect(self._on_progress_updated)

            # Update index status
            self._update_index_status()

        except Exception as e:
            logger.exception("Failed to initialize search service")
            self._index_status.setText(f"❌ Error: {e}")

    def _update_index_status(self):
        """Update the index status display."""
        if self.search_service:
            try:
                stats = self.search_service.get_index_stats()
                if stats.get('total_files', 0) > 0:
                    self._index_status.setText(
                        f"📊 {stats['total_files']} files, {stats['total_symbols']} symbols"
                    )
                else:
                    self._index_status.setText("📊 Not indexed")
            except Exception:
                self._index_status.setText("📊 Status unknown")

    def _on_search_text_changed(self, text: str):
        """Handle search text change with debouncing."""
        if len(text) >= 3:
            self._search_timer.stop()
            self._search_timer.start(300)  # 300ms debounce
        elif len(text) == 0:
            self._clear_results()

    def _on_search_clicked(self):
        """Handle search button click."""
        self._search_timer.stop()
        self._perform_search()

    def _perform_search(self):
        """Execute the search."""
        query = self._search_input.text().strip()
        if not query:
            return

        if self.search_service:
            self.search_service.search_async(query, max_results=20)

    def _on_search_started(self):
        """Handle search start."""
        self._progress_bar.show()
        self._progress_bar.setValue(0)
        self._search_btn.setEnabled(False)

    def _on_search_completed(self, results: List[SearchResult]):
        """Handle search completion."""
        self._current_results = results
        self._update_results_list()

        self._progress_bar.hide()
        self._search_btn.setEnabled(True)

    def _on_search_error(self, error: str):
        """Handle search error."""
        self._progress_bar.hide()
        self._search_btn.setEnabled(True)

        QMessageBox.warning(self, "Search Error", f"Search failed: {error}")

    def _on_progress_updated(self, current: int, total: int):
        """Handle progress update."""
        if total > 0:
            progress = int((current / total) * 100)
            self._progress_bar.setValue(progress)

    def _update_results_list(self):
        """Update the results list widget."""
        self._results_list.clear()

        # Apply filter
        filtered_results = self._apply_filter(self._current_results)

        # Sort results
        sorted_results = self._sort_results(filtered_results)

        for result in sorted_results:
            item = SearchResultItem(result)
            self._results_list.addItem(item)

        self._results_label.setText(f"Results: {len(sorted_results)}")

        # Select first result
        if self._results_list.count() > 0:
            self._results_list.setCurrentRow(0)

    def _apply_filter(self, results: List[SearchResult]) -> List[SearchResult]:
        """Apply type filter to results."""
        filter_text = self._filter_combo.currentText()

        if filter_text == "All Types":
            return results
        elif filter_text == "Functions/Methods":
            return [r for r in results if r.symbol_type in ('method', 'function')]
        elif filter_text == "Classes":
            return [r for r in results if r.symbol_type in ('class', 'interface', 'struct')]
        elif filter_text == "Variables/Fields":
            return [r for r in results if r.symbol_type in ('field', 'variable', 'property')]

        return results

    def _sort_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Sort results based on selected criteria."""
        sort_by = self._sort_combo.currentText()

        if sort_by == "Relevance":
            return sorted(results, key=lambda r: r.score, reverse=True)
        elif sort_by == "File Name":
            return sorted(results, key=lambda r: r.file_name.lower())
        elif sort_by == "Location":
            return sorted(results, key=lambda r: (r.file_path, r.start_line))

        return results

    def _on_selection_changed(self):
        """Handle result selection change."""
        selected_items = self._results_list.selectedItems()

        if selected_items:
            item = selected_items[0]
            if isinstance(item, SearchResultItem):
                self._selected_result = item.result
                self._preview.show_result(item.result)
                self._btn_open_file.setEnabled(True)
                self._btn_copy_path.setEnabled(True)
        else:
            self._selected_result = None
            self._preview.show_result(None)
            self._btn_open_file.setEnabled(False)
            self._btn_copy_path.setEnabled(False)

    def _on_item_double_clicked(self, item: QListWidgetItem):
        """Handle double-click on result."""
        if isinstance(item, SearchResultItem):
            self.result_activated.emit(item.result.file_path, item.result.start_line)
            self._open_result(item.result)

    def _on_context_menu(self, position):
        """Show context menu for result."""
        item = self._results_list.itemAt(position)
        if not item or not isinstance(item, SearchResultItem):
            return

        menu = QMenu(self)

        open_action = QAction("📂 Open File", self)
        open_action.triggered.connect(lambda: self._open_result(item.result))
        menu.addAction(open_action)

        copy_path_action = QAction("📋 Copy Path", self)
        copy_path_action.triggered.connect(lambda: self._copy_result_path(item.result))
        menu.addAction(copy_path_action)

        menu.addSeparator()

        copy_content_action = QAction("📄 Copy Content", self)
        copy_content_action.triggered.connect(lambda: self._copy_result_content(item.result))
        menu.addAction(copy_content_action)

        menu.exec(self._results_list.viewport().mapToGlobal(position))

    def _on_sort_changed(self):
        """Handle sort criteria change."""
        self._update_results_list()

    def _on_filter_changed(self):
        """Handle filter change."""
        self._update_results_list()

    def _on_open_file(self):
        """Handle open file button."""
        if self._selected_result:
            self._open_result(self._selected_result)

    def _on_copy_path(self):
        """Handle copy path button."""
        if self._selected_result:
            self._copy_result_path(self._selected_result)

    def _open_result(self, result: SearchResult):
        """Open a result in the editor."""
        self.result_selected.emit(result.file_path, result.start_line)

        # Copy path to clipboard for convenience
        self._copy_result_path(result)

    def _copy_result_path(self, result: SearchResult):
        """Copy result file path to clipboard."""
        clipboard = QApplication.clipboard()
        clipboard.setText(f"{result.file_path}:{result.start_line}")

    def _copy_result_content(self, result: SearchResult):
        """Copy result content to clipboard."""
        clipboard = QApplication.clipboard()
        clipboard.setText(result.content)

    def _on_index_project(self):
        """Handle index project button."""
        if not self.search_service:
            return

        self._btn_index.setEnabled(False)
        self._index_status.setText("🔄 Indexing...")

        # Show progress
        self._progress_bar.show()
        self._progress_bar.setRange(0, 0)  # Indeterminate

        # Start indexing in background
        import threading

        def do_index():
            try:
                result = self.search_service.index_project()

                # Update UI on main thread
                from PyQt6.QtCore import QMetaObject, Qt, Q_ARG

                if result.get('success'):
                    QMetaObject.invokeMethod(
                        self, "_on_index_complete",
                        Qt.ConnectionType.QueuedConnection,
                        Q_ARG(str, f"✅ Indexed {result.get('indexed', 0)} files")
                    )
                else:
                    QMetaObject.invokeMethod(
                        self, "_on_index_error",
                        Qt.ConnectionType.QueuedConnection,
                        Q_ARG(str, result.get('error', 'Unknown error'))
                    )
            except Exception as e:
                from PyQt6.QtCore import QMetaObject, Qt, Q_ARG
                QMetaObject.invokeMethod(
                    self, "_on_index_error",
                    Qt.ConnectionType.QueuedConnection,
                    Q_ARG(str, str(e))
                )

        thread = threading.Thread(target=do_index, daemon=True)
        thread.start()

    def _on_index_complete(self, message: str):
        """Handle index completion (called from background thread)."""
        self._progress_bar.hide()
        self._btn_index.setEnabled(True)
        self._index_status.setText(message)
        self._update_index_status()

    def _on_index_error(self, error: str):
        """Handle index error (called from background thread)."""
        self._progress_bar.hide()
        self._btn_index.setEnabled(True)
        self._index_status.setText(f"❌ Error: {error}")
        QMessageBox.warning(self, "Indexing Error", f"Failed to index project: {error}")

    def _clear_results(self):
        """Clear all search results."""
        self._current_results = []
        self._results_list.clear()
        self._results_label.setText("Results: 0")
        self._preview.show_result(None)
        self._selected_result = None

    def keyPressEvent(self, event: QKeyEvent):
        """Handle key press events."""
        if event.key() == Qt.Key.Key_Escape:
            if self._search_input.hasFocus() and self._search_input.text():
                self._search_input.clear()
            else:
                self.close()
        elif event.key() == Qt.Key.Key_Return or event.key() == Qt.Key.Key_Enter:
            if self._results_list.hasFocus() and self._selected_result:
                self._open_result(self._selected_result)
            else:
                self._perform_search()
        elif event.key() == Qt.Key.Key_Down:
            if not self._results_list.hasFocus():
                self._results_list.setFocus()
                if self._results_list.count() > 0:
                    self._results_list.setCurrentRow(0)
        else:
            super().keyPressEvent(event)

    def set_query(self, query: str):
        """Set the search query programmatically."""
        self._search_input.setText(query)
        self._search_input.setFocus()
        if len(query) >= 3:
            self._perform_search()

    def get_selected_result(self) -> Optional[SearchResult]:
        """Get the currently selected result."""
        return self._selected_result


def show_semantic_search(
    project_path: str,
    search_service: Optional[SemanticSearchService] = None,
    initial_query: str = "",
    parent: Optional[QWidget] = None
) -> Optional[SearchResult]:
    """Show the semantic search dialog.

    Args:
        project_path: Path to the project
        search_service: Optional existing search service
        initial_query: Optional initial search query
        parent: Parent widget

    Returns:
        Selected SearchResult or None if cancelled
    """
    dialog = SemanticSearchDialog(
        project_path=project_path,
        search_service=search_service,
        parent=parent
    )

    if initial_query:
        dialog.set_query(initial_query)

    # Center on parent
    if parent:
        rect = parent.geometry()
        dialog.move(
            rect.center().x() - dialog.width() // 2,
            rect.center().y() - dialog.height() // 2
        )

    result = dialog.exec()

    if result == QDialog.DialogCode.Accepted:
        return dialog.get_selected_result()

    return None
