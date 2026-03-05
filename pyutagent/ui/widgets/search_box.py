"""Enhanced search box with suggestions and history."""

from typing import Optional, List, Callable

from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QLineEdit, QPushButton, 
    QCompleter, QListView
)
from PyQt6.QtCore import Qt, pyqtSignal, QStringListModel
from PyQt6.QtGui import QIcon


class SearchBox(QWidget):
    """Enhanced search box with autocomplete and history.
    
    Features:
    - Autocomplete suggestions
    - Search history
    - Clear button
    - Search button
    """
    
    # Signals
    search_requested = pyqtSignal(str)  # search_text
    text_changed = pyqtSignal(str)  # current_text
    
    def __init__(self, parent: Optional[QWidget] = None, placeholder: str = "Search..."):
        super().__init__(parent)
        self._placeholder = placeholder
        self._history: List[str] = []
        self._max_history = 20
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the search box UI."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        
        # Search input
        self._input = QLineEdit()
        self._input.setPlaceholderText(self._placeholder)
        self._input.returnPressed.connect(self._on_search)
        self._input.textChanged.connect(self.text_changed.emit)
        layout.addWidget(self._input)
        
        # Setup autocomplete
        self._completer = QCompleter()
        self._completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self._completer.setFilterMode(Qt.MatchFlag.MatchContains)
        self._input.setCompleter(self._completer)
        
        self._suggestion_model = QStringListModel()
        self._completer.setModel(self._suggestion_model)
        
        # Clear button
        self._clear_btn = QPushButton("✕")
        self._clear_btn.setFixedSize(24, 24)
        self._clear_btn.setToolTip("Clear search")
        self._clear_btn.clicked.connect(self.clear)
        layout.addWidget(self._clear_btn)
        
        # Search button
        self._search_btn = QPushButton("🔍")
        self._search_btn.setFixedSize(24, 24)
        self._search_btn.setToolTip("Search")
        self._search_btn.clicked.connect(self._on_search)
        layout.addWidget(self._search_btn)
        
    def _on_search(self):
        """Handle search request."""
        text = self._input.text().strip()
        if text:
            self._add_to_history(text)
            self.search_requested.emit(text)
            
    def _add_to_history(self, text: str):
        """Add text to search history."""
        if text in self._history:
            self._history.remove(text)
        self._history.insert(0, text)
        
        # Trim history
        if len(self._history) > self._max_history:
            self._history = self._history[:self._max_history]
            
        # Update completer
        self._suggestion_model.setStringList(self._history)
        
    def set_suggestions(self, suggestions: List[str]):
        """Set autocomplete suggestions.
        
        Args:
            suggestions: List of suggestion strings
        """
        self._suggestion_model.setStringList(suggestions)
        
    def clear(self):
        """Clear the search box."""
        self._input.clear()
        
    def text(self) -> str:
        """Get current text."""
        return self._input.text()
        
    def set_text(self, text: str):
        """Set text."""
        self._input.setText(text)
        
    def focus(self):
        """Set focus to the search box."""
        self._input.setFocus()
        self._input.selectAll()
