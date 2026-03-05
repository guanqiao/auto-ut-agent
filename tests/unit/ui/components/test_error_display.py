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

    @pytest.fixture
    def error_widget(self, qtbot):
        """Create ErrorDisplayWidget instance."""
        widget = ErrorDisplayWidget()
        qtbot.addWidget(widget)
        return widget

    def test_error_widget_creation(self, error_widget):
        """Test ErrorDisplayWidget can be created."""
        assert error_widget is not None
        assert error_widget._error_id is None

    def test_set_error(self, error_widget):
        """Test set_error method."""
        error_widget.set_error(
            error_id="err_1",
            error_message="Something went wrong",
            error_type="RuntimeError",
            context={"key": "value"},
            retryable=True
        )

        assert error_widget._error_id == "err_1"
        assert "RuntimeError" in error_widget._title_label.text()
        assert error_widget._retry_btn.isEnabled() is True

    def test_set_error_not_retryable(self, error_widget):
        """Test set_error with non-retryable error."""
        error_widget.set_error(
            error_id="err_1",
            error_message="Fatal error",
            retryable=False
        )

        assert error_widget._retry_btn.isEnabled() is False

    def test_set_error_with_context(self, error_widget):
        """Test set_error with context."""
        context = {"step": "5", "tool": "test_tool"}
        error_widget.set_error(
            error_id="err_1",
            error_message="Error occurred",
            context=context
        )

        assert error_widget._context_label.isVisible() is True

    def test_clear(self, error_widget):
        """Test clear method."""
        error_widget.set_error(
            error_id="err_1",
            error_message="Test error"
        )

        error_widget.clear()

        assert error_widget._error_id is None
        assert error_widget._retry_btn.isEnabled() is True

    def test_get_error_id(self, error_widget):
        """Test get_error_id method."""
        error_widget.set_error(error_id="err_123", error_message="Test")
        assert error_widget.get_error_id() == "err_123"

    def test_is_retryable(self, error_widget):
        """Test is_retryable method."""
        error_widget.set_error(
            error_id="err_1",
            error_message="Test",
            retryable=True
        )
        assert error_widget.is_retryable() is True

    def test_toggle_details(self, error_widget):
        """Test _toggle_details method."""
        error_widget.set_error("err_1", "Test error")

        # Initially hidden
        assert error_widget._details_container.isVisible() is False

        # Toggle to show
        error_widget._toggle_details()
        assert error_widget._details_container.isVisible() is True
        assert error_widget._expanded is True

        # Toggle to hide
        error_widget._toggle_details()
        assert error_widget._details_container.isVisible() is False
        assert error_widget._expanded is False

    def test_on_retry_clicked(self, error_widget):
        """Test _on_retry_clicked method."""
        error_widget.set_error(error_id="err_1", error_message="Test")

        with patch.object(error_widget, 'retry_requested') as mock_signal:
            error_widget._on_retry_clicked()
            mock_signal.emit.assert_called_once_with("err_1")

    def test_on_skip_clicked(self, error_widget):
        """Test _on_skip_clicked method."""
        error_widget.set_error(error_id="err_1", error_message="Test")

        with patch.object(error_widget, 'skip_requested') as mock_signal:
            error_widget._on_skip_clicked()
            mock_signal.emit.assert_called_once_with("err_1")

    def test_set_callbacks(self, error_widget):
        """Test set_callbacks method."""
        on_retry = Mock()
        on_skip = Mock()

        error_widget.set_callbacks(on_retry=on_retry, on_skip=on_skip)

        assert error_widget._on_retry == on_retry
        assert error_widget._on_skip == on_skip


class TestErrorListWidget:
    """Test ErrorListWidget class."""

    @pytest.fixture
    def error_list(self, qtbot):
        """Create ErrorListWidget instance."""
        widget = ErrorListWidget()
        qtbot.addWidget(widget)
        return widget

    def test_error_list_creation(self, error_list):
        """Test ErrorListWidget can be created."""
        assert error_list is not None
        assert error_list.get_error_count() == 0

    def test_add_error(self, error_list):
        """Test add_error method."""
        error_widget = error_list.add_error(
            error_id="err_1",
            error_message="Test error",
            error_type="RuntimeError"
        )

        assert error_list.get_error_count() == 1
        assert error_widget is not None
        assert "err_1" in error_list._errors

    def test_add_error_update_existing(self, error_list):
        """Test add_error updates existing error."""
        error_list.add_error(error_id="err_1", error_message="Old error")
        error_widget = error_list.add_error(error_id="err_1", error_message="New error")

        assert error_list.get_error_count() == 1
        assert error_widget._summary_label.text() == "New error"

    def test_remove_error(self, error_list):
        """Test remove_error method."""
        error_list.add_error(error_id="err_1", error_message="Test")
        error_list.remove_error("err_1")

        assert error_list.get_error_count() == 0
        assert "err_1" not in error_list._errors

    def test_clear_errors(self, error_list):
        """Test clear_errors method."""
        error_list.add_error(error_id="err_1", error_message="Test 1")
        error_list.add_error(error_id="err_2", error_message="Test 2")

        error_list.clear_errors()

        assert error_list.get_error_count() == 0
        assert len(error_list._errors) == 0

    def test_has_errors(self, error_list):
        """Test has_errors method."""
        assert error_list.has_errors() is False

        error_list.add_error(error_id="err_1", error_message="Test")
        assert error_list.has_errors() is True

    def test_get_error(self, error_list):
        """Test get_error method."""
        error_list.add_error(error_id="err_1", error_message="Test")

        widget = error_list.get_error("err_1")
        assert widget is not None
        assert widget.get_error_id() == "err_1"

    def test_get_error_not_found(self, error_list):
        """Test get_error returns None for non-existent error."""
        assert error_list.get_error("non_existent") is None

    def test_retry_signal_propagation(self, error_list):
        """Test retry signal is propagated."""
        with patch.object(error_list, 'retry_requested') as mock_signal:
            error_list.add_error(error_id="err_1", error_message="Test")
            error_list._on_retry("err_1")
            mock_signal.emit.assert_called_once_with("err_1")

    def test_skip_signal_propagation(self, error_list):
        """Test skip signal is propagated."""
        with patch.object(error_list, 'skip_requested') as mock_signal:
            error_list.add_error(error_id="err_1", error_message="Test")
            error_list._on_skip("err_1")
            mock_signal.emit.assert_called_once_with("err_1")


class TestCreateErrorDisplay:
    """Test create_error_display function."""

    def test_create_error_display(self, qtbot):
        """Test create_error_display creates correct instance."""
        widget = create_error_display()
        qtbot.addWidget(widget)

        assert isinstance(widget, ErrorDisplayWidget)


class TestCreateErrorList:
    """Test create_error_list function."""

    def test_create_error_list(self, qtbot):
        """Test create_error_list creates correct instance."""
        widget = create_error_list()
        qtbot.addWidget(widget)

        assert isinstance(widget, ErrorListWidget)
