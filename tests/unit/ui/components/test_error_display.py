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
            error_message="Something