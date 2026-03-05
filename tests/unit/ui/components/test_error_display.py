"""Tests for error_display module."""

import pytest
from unittest.mock import Mock, patch

from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt

from pyutagent.ui.components.error_display import (
    ErrorDisplayWidget, ErrorListWidget,
    create_error_display, create_error_list
)


class TestErrorDisplayWidget:
    """Test ErrorDisplayWidget class."""

    def test_error_widget_creation(self):
        """Test ErrorDisplayWidget can be created."""
        widget = ErrorDisplayWidget()
        assert widget is not None
        assert widget._error_id is None
        widget.deleteLater()

    def test_set_error(self):
        """Test set_error method."""
        widget = ErrorDisplayWidget()
        widget.set_error(
            error_id="err_1",
            error_message="Something went wrong",
            error_type="RuntimeError",
            context={"key": "value"},
            retryable=True
        )

        assert widget._error_id == "err_1"
        assert "RuntimeError" in widget._title_label.text()
        assert widget._retry_btn.isEnabled() is True
        widget.deleteLater()

    def test_set_error_not_retryable(self):
        """Test set_error with non-retryable error."""
        widget = ErrorDisplayWidget()
        widget.set_error(
            error_id="err_1",
            error_message="Fatal error",
            retryable=False
        )

        assert widget._retry_btn.isEnabled() is False
        widget.deleteLater()

    def test_set_error_with_context(self):
        """Test set_error with context."""
        widget = ErrorDisplayWidget()
        context = {"step": "5", "tool": "test_tool"}
        widget.set_error(
            error_id="err_1",
            error_message="Error occurred",
            context=context
        )

        assert widget._context_label.isVisible() is True
        widget.deleteLater()

    def test_clear(self):
        """Test clear method."""
        widget = ErrorDisplayWidget()
        widget.set_error(
            error_id="err_1",
            error_message="Test error"
        )

        widget.clear()

        assert widget._error_id is None
        assert widget._retry_btn.isEnabled() is True
        widget.deleteLater()

    def test_get_error_id(self):
        """Test get_error_id method."""
        widget = ErrorDisplayWidget()
        widget.set_error(error_id="err_123", error_message="Test")
        assert widget.get_error_id() == "err_123"
        widget.deleteLater()

    def test_is_retryable(self):
        """Test is_retryable method."""
        widget = ErrorDisplayWidget()
        widget.set_error(
            error_id="err_1",
            error_message="Test",
            retryable=True
        )
        assert widget.is_retryable() is True
        widget.deleteLater()

    def test_toggle_details(self):
        """Test _toggle_details method."""
        widget = ErrorDisplayWidget()
        widget.set_error("err_1", "Test error")

        # Initially hidden
        assert widget._details_container.isVisible() is False

        # Toggle to show
        widget._toggle_details()
        assert widget._details_container.isVisible() is True
        assert widget._expanded is True

        # Toggle to hide
        widget._toggle_details()
        assert widget._details_container.isVisible() is False
        assert widget._expanded is False
        widget.deleteLater()

    def test_on_retry_clicked(self):
        """Test _on_retry_clicked method."""
        widget = ErrorDisplayWidget()
        widget.set_error(error_id="err_1", error_message="Test")

        with patch.object(widget, 'retry_requested') as mock_signal:
            widget._on_retry_clicked()
            mock_signal.emit.assert_called_once_with("err_1")
        widget.deleteLater()

    def test_on_skip_clicked(self):
        """Test _on_skip_clicked method."""
        widget = ErrorDisplayWidget()
        widget.set_error(error_id="err_1", error_message="Test")

        with patch.object(widget, 'skip_requested') as mock_signal:
            widget._on_skip_clicked()
            mock_signal.emit.assert_called_once_with("err_1")
        widget.deleteLater()

    def test_set_callbacks(self):
        """Test set_callbacks method."""
        widget = ErrorDisplayWidget()
        on_retry = Mock()
        on_skip = Mock()

        widget.set_callbacks(on_retry=on_retry, on_skip=on_skip)

        assert widget._on_retry == on_retry
        assert widget._on_skip == on_skip
        widget.deleteLater()


class TestErrorListWidget:
    """Test ErrorListWidget class."""

    def test_error_list_creation(self):
        """Test ErrorListWidget can be created."""
        widget = ErrorListWidget()
        assert widget is not None
        assert widget.get_error_count() == 0
        widget.deleteLater()

    def test_add_error(self):
        """Test add_error method."""
        widget = ErrorListWidget()
        error_widget = widget.add_error(
            error_id="err_1",
            error_message="Test error",
            error_type="RuntimeError"
        )

        assert widget.get_error_count() == 1
        assert error_widget is not None
        assert "err_1" in widget._errors
        widget.deleteLater()

    def test_add_error_update_existing(self):
        """Test add_error updates existing error."""
        widget = ErrorListWidget()
        widget.add_error(error_id="err_1", error_message="Old error")
        error_widget = widget.add_error(error_id="err_1", error_message="New error")

        assert widget.get_error_count() == 1
        assert error_widget._summary_label.text() == "New error"
        widget.deleteLater()

    def test_remove_error(self):
        """Test remove_error method."""
        widget = ErrorListWidget()
        widget.add_error(error_id="err_1", error_message="Test")
        widget.remove_error("err_1")

        assert widget.get_error_count() == 0
        assert "err_1" not in widget._errors
        widget.deleteLater()

    def test_clear_errors(self):
        """Test clear_errors method."""
        widget = ErrorListWidget()
        widget.add_error(error_id="err_1", error_message="Test 1")
        widget.add_error(error_id="err_2", error_message="Test 2")

        widget.clear_errors()

        assert widget.get_error_count() == 0
        assert len(widget._errors) == 0
        widget.deleteLater()

    def test_has_errors(self):
        """Test has_errors method."""
        widget = ErrorListWidget()
        assert widget.has_errors() is False

        widget.add_error(error_id="err_1", error_message="Test")
        assert widget.has_errors() is True
        widget.deleteLater()

    def test_get_error(self):
        """Test get_error method."""
        widget = ErrorListWidget()
        widget.add_error(error_id="err_1", error_message="Test")

        error_widget = widget.get_error("err_1")
        assert error_widget is not None
        assert error_widget.get_error_id() == "err_1"
        widget.deleteLater()

    def test_get_error_not_found(self):
        """Test get_error returns None for non-existent error."""
        widget = ErrorListWidget()
        assert widget.get_error("non_existent") is None
        widget.deleteLater()

    def test_retry_signal_propagation(self):
        """Test retry signal is propagated."""
        widget = ErrorListWidget()
        with patch.object(widget, 'retry_requested') as mock_signal:
            widget.add_error(error_id="err_1", error_message="Test")
            widget._on_retry("err_1")
            mock_signal.emit.assert_called_once_with("err_1")
        widget.deleteLater()

    def test_skip_signal_propagation(self):
        """Test skip signal is propagated."""
        widget = ErrorListWidget()
        with patch.object(widget, 'skip_requested') as mock_signal:
            widget.add_error(error_id="err_1", error_message="Test")
            widget._on_skip("err_1")
            mock_signal.emit.assert_called_once_with("err_1")
        widget.deleteLater()


class TestCreateErrorDisplay:
    """Test create_error_display function."""
