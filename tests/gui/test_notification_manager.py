"""Tests for the notification manager."""

import pytest
from PyQt6.QtWidgets import QApplication

from pyutagent.ui.components import NotificationManager, ToastNotification, NotificationType, get_notification_manager


@pytest.mark.gui
class TestNotificationManager:
    """Test suite for NotificationManager."""
    
    def test_singleton_pattern(self):
        """Test that NotificationManager is a singleton."""
        manager1 = NotificationManager()
        manager2 = NotificationManager()
        assert manager1 is manager2
    
    def test_get_notification_manager(self):
        """Test get_notification_manager function."""
        manager = get_notification_manager()
        assert isinstance(manager, NotificationManager)
        assert manager is get_notification_manager()
    
    def test_show_success(self, qtbot):
        """Test showing success notification."""
        manager = NotificationManager()
        
        # Should not raise
        manager.show_success("Test success message", duration=100)
        
        # Clean up
        manager.clear_all()
    
    def test_show_warning(self, qtbot):
        """Test showing warning notification."""
        manager = NotificationManager()
        
        # Should not raise
        manager.show_warning("Test warning message", duration=100)
        
        # Clean up
        manager.clear_all()
    
    def test_show_error(self, qtbot):
        """Test showing error notification."""
        manager = NotificationManager()
        
        # Should not raise
        manager.show_error("Test error message", duration=100)
        
        # Clean up
        manager.clear_all()
    
    def test_show_info(self, qtbot):
        """Test showing info notification."""
        manager = NotificationManager()
        
        # Should not raise
        manager.show_info("Test info message", duration=100)
        
        # Clean up
        manager.clear_all()
    
    def test_clear_all(self, qtbot):
        """Test clearing all notifications."""
        manager = NotificationManager()
        
        # Add some notifications
        manager.show_info("Message 1", duration=5000)
        manager.show_info("Message 2", duration=5000)
        
        # Clear all
        manager.clear_all()
        
        # Should have no notifications
        assert len(manager._notifications) == 0
    
    def test_max_notifications(self, qtbot):
        """Test maximum notification limit."""
        manager = NotificationManager()
        manager._max_notifications = 3
        
        # Add more than max notifications
        for i in range(5):
            manager.show_info(f"Message {i}", duration=5000)
        
        # Should only have max notifications
        assert len(manager._notifications) <= manager._max_notifications
        
        # Clean up
        manager.clear_all()


@pytest.mark.gui
class TestToastNotification:
    """Test suite for ToastNotification."""
    
    def test_notification_creation(self, qtbot):
        """Test creating a toast notification."""
        notification = ToastNotification(
            message="Test message",
            notification_type=NotificationType.INFO,
            title="Test Title",
            duration=100
        )
        qtbot.addWidget(notification)
        
        assert notification._content == "Test message"
        assert notification._notification_type == NotificationType.INFO
        assert notification._duration == 100
    
    def test_notification_types(self, qtbot):
        """Test different notification types."""
        for notif_type in [NotificationType.SUCCESS, NotificationType.WARNING, 
                          NotificationType.ERROR, NotificationType.INFO]:
            notification = ToastNotification(
                message="Test message",
                notification_type=notif_type,
                duration=100
            )
            qtbot.addWidget(notification)
            
            assert notification._notification_type == notif_type
    
    def test_append_chunk(self, qtbot):
        """Test appending content chunks."""
        notification = ToastNotification(
            message="Initial",
            notification_type=NotificationType.INFO,
            duration=5000
        )
        qtbot.addWidget(notification)
        
        notification.append_chunk(" chunk1")
        assert notification._content == "Initial chunk1"
        
        notification.append_chunk(" chunk2")
        assert notification._content == "Initial chunk1 chunk2"
    
    def test_update_content(self, qtbot):
        """Test updating content."""
        notification = ToastNotification(
            message="Initial content",
            notification_type=NotificationType.INFO,
            duration=5000
        )
        qtbot.addWidget(notification)
        
        notification.update_content("Updated content")
        assert notification._content == "Updated content"
    
    def test_get_content(self, qtbot):
        """Test getting raw content."""
        notification = ToastNotification(
            message="Test content",
            notification_type=NotificationType.INFO,
            duration=5000
        )
        qtbot.addWidget(notification)
        
        assert notification.get_content() == "Test content"
    
    def test_close_notification(self, qtbot):
        """Test closing notification."""
        notification = ToastNotification(
            message="Test message",
            notification_type=NotificationType.INFO,
            duration=5000
        )
        qtbot.addWidget(notification)
        
        # Close notification
        notification.close_notification()
        
        # Signal should be emitted
        # Note: In real test, we'd wait for the signal
    
    def test_notification_without_duration(self, qtbot):
        """Test notification without auto-close duration."""
        notification = ToastNotification(
            message="Persistent message",
            notification_type=NotificationType.INFO,
            duration=0  # No auto-close
        )
        qtbot.addWidget(notification)
        
        assert notification._duration == 0
        assert not hasattr(notification, '_timer') or notification._timer is None
