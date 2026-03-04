"""Notification manager for displaying toast notifications."""

import logging
from enum import Enum
from typing import Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGraphicsDropShadowEffect, QApplication
)
from PyQt6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, QPoint, pyqtSignal
from PyQt6.QtGui import QColor, QIcon

from ..styles import get_style_manager

logger = logging.getLogger(__name__)


class NotificationType(Enum):
    """Notification types with associated colors and icons."""
    SUCCESS = ("success", "✓", "#4CAF50")
    WARNING = ("warning", "⚠", "#FF9800")
    ERROR = ("error", "✕", "#F44336")
    INFO = ("info", "ℹ", "#2196F3")
    
    def __init__(self, key: str, icon: str, color: str):
        self.key = key
        self.icon = icon
        self.color = color


class ToastNotification(QWidget):
    """A single toast notification widget."""
    
    closed = pyqtSignal()
    
    def __init__(
        self,
        message: str,
        notification_type: NotificationType = NotificationType.INFO,
        title: Optional[str] = None,
        duration: int = 5000,
        parent=None
    ):
        super().__init__(parent)
        self._duration = duration
        self._notification_type = notification_type
        self._style_manager = get_style_manager()
        
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        self.setup_ui(message, title)
        self.apply_styles()
        
        # Auto-close timer
        if duration > 0:
            self._timer = QTimer(self)
            self._timer.timeout.connect(self.close_notification)
            self._timer.start(duration)
        
        # Animation
        self._animation = QPropertyAnimation(self, b"windowOpacity")
        self._animation.setDuration(300)
        self._animation.setEasingCurve(QEasingCurve.Type.InOutQuad)
    
    def setup_ui(self, message: str, title: Optional[str]):
        """Setup the notification UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(8)
        
        # Header with icon and title
        header_layout = QHBoxLayout()
        header_layout.setSpacing(8)
        
        # Icon
        self.icon_label = QLabel(self._notification_type.icon)
        self.icon_label.setStyleSheet(f"font-size: 18px; color: {self._notification_type.color};")
        header_layout.addWidget(self.icon_label)
        
        # Title
        if title:
            self.title_label = QLabel(title)
            self.title_label.setStyleSheet("font-weight: bold; font-size: 14px;")
            header_layout.addWidget(self.title_label)
        
        header_layout.addStretch()
        
        # Close button
        self.close_btn = QPushButton("✕")
        self.close_btn.setFixedSize(20, 20)
        self.close_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                border: none;
                color: #999;
                font-size: 12px;
            }
            QPushButton:hover {
                color: #333;
            }
        """)
        self.close_btn.clicked.connect(self.close_notification)
        header_layout.addWidget(self.close_btn)
        
        layout.addLayout(header_layout)
        
        # Message
        self.message_label = QLabel(message)
        self.message_label.setWordWrap(True)
        self.message_label.setStyleSheet("font-size: 13px; line-height: 1.4;")
        layout.addWidget(self.message_label)
        
        # Set fixed width
        self.setFixedWidth(320)
        self.adjustSize()
    
    def apply_styles(self):
        """Apply theme styles to the notification."""
        is_dark = self._style_manager.current_theme == "dark"
        
        bg_color = "#2D2D2D" if is_dark else "#FFFFFF"
        text_color = "#E0E0E0" if is_dark else "#212121"
        border_color = self._notification_type.color
        
        self.setStyleSheet(f"""
            ToastNotification {{
                background-color: {bg_color};
                color: {text_color};
                border-left: 4px solid {border_color};
                border-radius: 8px;
            }}
            QLabel {{
                color: {text_color};
                background: transparent;
            }}
        """)
        
        # Add shadow effect
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(0, 0, 0, 60))
        shadow.setOffset(0, 4)
        self.setGraphicsEffect(shadow)
    
    def showEvent(self, event):
        """Handle show event with fade-in animation."""
        super().showEvent(event)
        self._animation.setStartValue(0.0)
        self._animation.setEndValue(1.0)
        self._animation.start()
    
    def close_notification(self):
        """Close the notification with fade-out animation."""
        self._animation.setStartValue(1.0)
        self._animation.setEndValue(0.0)
        self._animation.finished.connect(self._do_close)
        self._animation.start()
    
    def _do_close(self):
        """Actually close the widget."""
        self.closed.emit()
        self.close()
    
    def enterEvent(self, event):
        """Pause auto-close when mouse enters."""
        if hasattr(self, '_timer') and self._timer.isActive():
            self._timer.stop()
        super().enterEvent(event)
    
    def leaveEvent(self, event):
        """Resume auto-close when mouse leaves."""
        if hasattr(self, '_timer') and self._duration > 0:
            self._timer.start(self._duration)
        super().leaveEvent(event)


class NotificationManager:
    """Manages toast notifications display."""
    
    _instance: Optional['NotificationManager'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._notifications: list = []
        self._max_notifications = 5
        self._spacing = 10
        self._margin = 20
    
    def show(
        self,
        message: str,
        notification_type: NotificationType = NotificationType.INFO,
        title: Optional[str] = None,
        duration: int = 5000
    ):
        """Show a toast notification."""
        notification = ToastNotification(
            message=message,
            notification_type=notification_type,
            title=title,
            duration=duration
        )
        
        notification.closed.connect(lambda: self._remove_notification(notification))
        
        self._notifications.append(notification)
        
        # Limit max notifications
        if len(self._notifications) > self._max_notifications:
            old_notification = self._notifications.pop(0)
            old_notification.close()
        
        self._position_notifications()
        notification.show()
        
        logger.debug(f"Showed {notification_type.key} notification: {message[:50]}...")
    
    def show_success(self, message: str, title: Optional[str] = "成功", duration: int = 3000):
        """Show a success notification."""
        self.show(message, NotificationType.SUCCESS, title, duration)
    
    def show_warning(self, message: str, title: Optional[str] = "警告", duration: int = 5000):
        """Show a warning notification."""
        self.show(message, NotificationType.WARNING, title, duration)
    
    def show_error(self, message: str, title: Optional[str] = "错误", duration: int = 8000):
        """Show an error notification."""
        self.show(message, NotificationType.ERROR, title, duration)
    
    def show_info(self, message: str, title: Optional[str] = "信息", duration: int = 5000):
        """Show an info notification."""
        self.show(message, NotificationType.INFO, title, duration)
    
    def _remove_notification(self, notification: ToastNotification):
        """Remove a notification from the list."""
        if notification in self._notifications:
            self._notifications.remove(notification)
            self._position_notifications()
    
    def _position_notifications(self):
        """Position all notifications on screen."""
        if not self._notifications:
            return
        
        # Get primary screen geometry
        screen = QApplication.primaryScreen()
        if not screen:
            return
        
        screen_geometry = screen.availableGeometry()
        
        # Position from bottom-right
        x = screen_geometry.right() - self._margin - 320  # 320 is notification width
        y = screen_geometry.bottom() - self._margin
        
        # Stack notifications from bottom to top
        for notification in reversed(self._notifications):
            notification_height = notification.height()
            y -= notification_height
            notification.move(x, y)
            y -= self._spacing
    
    def clear_all(self):
        """Close all notifications."""
        for notification in self._notifications[:]:
            notification.close()
        self._notifications.clear()


def get_notification_manager() -> NotificationManager:
    """Get the singleton notification manager instance."""
    return NotificationManager()
