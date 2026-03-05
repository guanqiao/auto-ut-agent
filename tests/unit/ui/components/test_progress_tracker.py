"""Tests for progress_tracker module."""

import pytest
from unittest.mock import patch

from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt

from pyutagent.ui.components.progress_tracker import (
    ProgressTracker, CircularProgress, create_progress_tracker
)


class TestProgressTracker:
    """Test ProgressTracker class."""

    @pytest.fixture
    def progress_tracker(self, qtbot):
        """Create ProgressTracker instance."""
        tracker = ProgressTracker()
        qtbot.addWidget(tracker)
        return tracker

    def test_progress_tracker_creation(self, progress_tracker):
        """Test ProgressTracker can be created."""
        assert progress_tracker is not None
        assert progress_tracker._current_step == 0
        assert progress_tracker._total_steps == 0

    def test_set_progress(self, progress_tracker):
        """Test set_progress method."""
        progress_tracker.set_progress(5, 10, "Halfway done")

        assert progress_tracker._current_step == 5
        assert progress_tracker._total_steps == 10
        assert progress_tracker._progress_bar.value() == 50
        assert progress_tracker._step_label.text() == "5/10"

    def test_set_progress_completion(self, progress_tracker):
        """Test set_progress with completion."""
        progress_tracker.set_progress(10, 10, "Complete")

        assert progress_tracker._progress_bar.value() == 100
        assert "Completed" in progress_tracker._status_label.text()

    def test_set_status_info(self, progress_tracker):
        """Test set_status with info type."""
        progress_tracker.set_status("Processing", "info")
        assert progress_tracker._status_label.text() == "Processing"

    def test_set_status_success(self, progress_tracker):
        """Test set_status with success type."""
        progress_tracker.set_status("Done", "success")
        assert progress_tracker._status_label.text() == "Done"

    def test_set_status_warning(self, progress_tracker):
        """Test set_status with warning type."""
        progress_tracker.set_status("Warning", "warning")
        assert progress_tracker._status_label.text() == "Warning"

    def test_set_status_error(self, progress_tracker):
        """Test set_status with error type."""
        progress_tracker.set_status("Error", "error")
        assert progress_tracker._status_label.text() == "Error"

    def test_add_step_detail(self, progress_tracker):
        """Test add_step_detail method."""
        progress_tracker.add_step_detail(1, "First step", "completed")
        progress_tracker.add_step_detail(2, "Second step", "running")

        assert len(progress_tracker._step_details) == 2
        assert progress_tracker._step_details[0]["number"] == 1
        assert progress_tracker._step_details[0]["description"] == "First step"
        assert progress_tracker._step_details[0]["status"] == "completed"

    def test_update_step_status(self, progress_tracker):
        """Test update_step_status method."""
        progress_tracker.add_step_detail(1, "First step", "pending")
        progress_tracker.update_step_status(1, "completed")

        assert progress_tracker._step_details[0]["status"] == "completed"

    def test_clear(self, progress_tracker):
        """Test clear method."""
        progress_tracker.set_progress(5, 10, "Halfway")
        progress_tracker.add_step_detail(1, "Step 1", "completed")

        progress_tracker.clear()

        assert progress_tracker._current_step == 0
        assert progress_tracker._total_steps == 0
        assert progress_tracker._progress_bar.value() == 0
        assert len(progress_tracker._step_details) == 0

    def test_get_progress_percent(self, progress_tracker):
        """Test get_progress_percent method."""
        progress_tracker.set_progress(5, 10)
        assert progress_tracker.get_progress_percent() == 50.0

    def test_get_progress_percent_zero(self, progress_tracker):
        """Test get_progress_percent with zero total."""
        assert progress_tracker.get_progress_percent() == 0.0

    def test_is_completed(self, progress_tracker):
        """Test is_completed method."""
        progress_tracker.set_progress(10, 10)
        assert progress_tracker.is_completed() is True

    def test_is_completed_not_complete(self, progress_tracker):
        """Test is_completed when not complete."""
        progress_tracker.set_progress(5, 10)
        assert progress_tracker.is_completed() is False

    def test_toggle_details(self, progress_tracker):
        """Test _toggle_details method."""
        # Initially hidden
        assert progress_tracker._details_container.isVisible() is False

        # Toggle to show
        progress_tracker._toggle_details()
        assert progress_tracker._details_container.isVisible() is True
        assert progress_tracker._expanded is True

        # Toggle to hide
        progress_tracker._toggle_details()
        assert progress_tracker._details_container.isVisible() is False
        assert progress_tracker._expanded is False

    def test_update_details_display(self, progress_tracker):
        """Test _update_details_display method."""
        progress_tracker.add_step_detail(1, "Step 1", "completed")
        progress_tracker.add_step_detail(2, "Step 2", "running")

        progress_tracker._update_details_display()

        text = progress_tracker._steps_text.toPlainText()
        assert "Step 1" in text
        assert "Step 2" in text


class TestCircularProgress:
    """Test CircularProgress class."""

    @pytest.fixture
    def circular_progress(self, qtbot):
        """Create CircularProgress instance."""
        progress = CircularProgress(diameter=40)
        qtbot.addWidget(progress)
        return progress

    def test_circular_progress_creation(self, circular_progress):
        """Test CircularProgress can be created."""
        assert circular_progress is not None
        assert circular_progress._diameter == 40
        assert circular_progress._progress == 0

    def test_set_progress(self, circular_progress):
        """Test set_progress method."""
        circular_progress.set_progress(75)
        assert circular_progress._progress == 75

    def test_set_progress_clamping(self, circular_progress):
        """Test set_progress clamps values."""
        circular_progress.set_progress(150)
        assert circular_progress._progress == 100

        circular_progress.set_progress(-10)
        assert circular_progress._progress == 0

    def test_set_color(self, circular_progress):
        """Test set_color method."""
        circular_progress.set_color("#FF0000")
        assert circular_progress._color == "#FF0000"


class TestCreateProgressTracker:
    """Test create_progress_tracker function."""

    def test_create_progress_tracker(self):
        """Test create_progress_tracker creates correct instance."""
        tracker = create_progress_tracker()
        assert isinstance(tracker, ProgressTracker)
        tracker.deleteLater()
