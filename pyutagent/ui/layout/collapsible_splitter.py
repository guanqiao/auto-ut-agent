"""Collapsible splitter for resizable panels with collapse/expand functionality."""

import logging
from typing import Optional, Callable

from PyQt6.QtWidgets import (
    QSplitter, QSplitterHandle, QWidget, QVBoxLayout,
    QPushButton, QApplication
)
from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtGui import QCursor, QIcon

logger = logging.getLogger(__name__)


class CollapsibleSplitterHandle(QSplitterHandle):
    """Custom splitter handle with collapse/expand button."""
    
    def __init__(self, orientation: Qt.Orientation, parent: QSplitter, panel_index: int):
        super().__init__(orientation, parent)
        self._panel_index = panel_index
        self._is_collapsed = False
        self._collapsed_size = 0
        self._normal_size = 200
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the handle UI."""
        self._collapse_btn = QPushButton(self)
        self._collapse_btn.setFixedSize(16, 32)
        self._collapse_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self._collapse_btn.clicked.connect(self._on_collapse_clicked)
        self._update_button_style()
        
    def _update_button_style(self):
        """Update button style based on orientation and state."""
        if self.orientation() == Qt.Orientation.Horizontal:
            if self._is_collapsed:
                self._collapse_btn.setText("◀")
            else:
                self._collapse_btn.setText("◀")
        else:
            if self._is_collapsed:
                self._collapse_btn.setText("▲")
            else:
                self._collapse_btn.setText("▼")
                
    def _on_collapse_clicked(self):
        """Handle collapse/expand button click."""
        splitter = self.splitter()
        if not splitter:
            return
            
        if self._is_collapsed:
            # Expand
            splitter.setSizes([self._normal_size, splitter.width() - self._normal_size - 10])
            self._is_collapsed = False
        else:
            # Collapse
            sizes = splitter.sizes()
            if self._panel_index < len(sizes):
                self._normal_size = sizes[self._panel_index]
                self._collapsed_size = 30  # Minimum visible size
                new_sizes = sizes.copy()
                new_sizes[self._panel_index] = self._collapsed_size
                splitter.setSizes(new_sizes)
                self._is_collapsed = True
                
        self._update_button_style()
        
    def resizeEvent(self, event):
        """Center the button on the handle."""
        super().resizeEvent(event)
        if hasattr(self, '_collapse_btn'):
            if self.orientation() == Qt.Orientation.Horizontal:
                self._collapse_btn.move(2, (self.height() - 32) // 2)
            else:
                self._collapse_btn.move((self.width() - 32) // 2, 2)


class CollapsibleSplitter(QSplitter):
    """Splitter with collapsible panels.
    
    Features:
    - Drag to resize panels
    - Double-click handle to collapse/expand
    - Collapse/expand buttons on handles
    - Smooth animations (optional)
    """
    
    panel_collapsed = pyqtSignal(int, bool)  # panel_index, is_collapsed
    panel_resized = pyqtSignal(int, int)  # panel_index, new_size
    
    def __init__(self, orientation: Qt.Orientation = Qt.Orientation.Horizontal, 
                 parent: Optional[QWidget] = None):
        super().__init__(orientation, parent)
        self._collapsible = []
        self._min_sizes = []
        self._collapsed_sizes = []
        self._normal_sizes = []
        self._animation_enabled = False
        
        self.setHandleWidth(8)
        self.setChildrenCollapsible(True)
        
    def addWidget(self, widget: QWidget, collapsible: bool = True, 
                  min_size: int = 50, collapsed_size: int = 30):
        """Add a widget to the splitter.
        
        Args:
            widget: The widget to add
            collapsible: Whether this panel can be collapsed
            min_size: Minimum size when expanded
            collapsed_size: Size when collapsed
        """
        super().addWidget(widget)
        index = self.count() - 1
        
        self._collapsible.append(collapsible)
        self._min_sizes.append(min_size)
        self._collapsed_sizes.append(collapsed_size)
        self._normal_sizes.append(widget.width() if self.orientation() == Qt.Orientation.Horizontal else widget.height())
        
        # Set minimum size
        if self.orientation() == Qt.Orientation.Horizontal:
            widget.setMinimumWidth(collapsed_size if collapsible else min_size)
        else:
            widget.setMinimumHeight(collapsed_size if collapsible else min_size)
            
    def createHandle(self) -> QSplitterHandle:
        """Create custom handle with collapse button."""
        index = len(self._collapsible) - 1
        if index >= 0 and self._collapsible[index]:
            return CollapsibleSplitterHandle(self.orientation(), self, index)
        return super().createHandle()
        
    def collapse_panel(self, index: int):
        """Collapse a panel by index."""
        if index < 0 or index >= self.count():
            logger.warning(f"Invalid panel index: {index}")
            return
            
        if not self._collapsible[index]:
            logger.debug(f"Panel {index} is not collapsible")
            return
            
        sizes = self.sizes()
        self._normal_sizes[index] = sizes[index]
        sizes[index] = self._collapsed_sizes[index]
        self.setSizes(sizes)
        self.panel_collapsed.emit(index, True)
        
    def expand_panel(self, index: int):
        """Expand a panel by index."""
        if index < 0 or index >= self.count():
            logger.warning(f"Invalid panel index: {index}")
            return
            
        sizes = self.sizes()
        sizes[index] = self._normal_sizes[index]
        self.setSizes(sizes)
        self.panel_collapsed.emit(index, False)
        
    def toggle_panel(self, index: int):
        """Toggle collapse/expand state of a panel."""
        if index < 0 or index >= self.count():
            return
            
        sizes = self.sizes()
        if sizes[index] <= self._collapsed_sizes[index] + 5:
            self.expand_panel(index)
        else:
            self.collapse_panel(index)
            
    def is_panel_collapsed(self, index: int) -> bool:
        """Check if a panel is collapsed."""
        if index < 0 or index >= self.count():
            return False
        sizes = self.sizes()
        return sizes[index] <= self._collapsed_sizes[index] + 5
        
    def set_panel_sizes(self, sizes: list):
        """Set sizes for all panels."""
        if len(sizes) == self.count():
            self.setSizes(sizes)
            self._normal_sizes = sizes.copy()
            
    def get_panel_sizes(self) -> list:
        """Get current sizes of all panels."""
        return self.sizes()
        
    def save_state(self) -> dict:
        """Save splitter state."""
        return {
            'sizes': self.sizes(),
            'normal_sizes': self._normal_sizes,
            'collapsible': self._collapsible.copy()
        }
        
    def restore_state(self, state: dict):
        """Restore splitter state."""
        if 'sizes' in state and len(state['sizes']) == self.count():
            self.setSizes(state['sizes'])
        if 'normal_sizes' in state:
            self._normal_sizes = state['normal_sizes']
            
    def mouseDoubleClickEvent(self, event):
        """Handle double-click on splitter handle."""
        handle = self.handleAt(event.pos())
        if handle:
            # Find which panel this handle controls
            for i in range(self.count() - 1):
                if self.handle(i) == handle:
                    self.toggle_panel(i)
                    break
        super().mouseDoubleClickEvent(event)
