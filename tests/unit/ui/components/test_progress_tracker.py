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

    def test_progress_tracker_creation(self):
        """Test ProgressTracker can be created."""
        tracker = ProgressTracker()
        assert tracker is not None
        assert tracker._current_step == 0
        assert tracker._total_steps == 0
        tracker.deleteLater()

    def test_set_progress(self):
        """Test set_progress method."""
        tracker = ProgressTracker()
        tracker.set_progress(5, 10, "Halfway done")

        assert tracker._current_step == 5
        assert tracker._total_steps == 10
        assert tracker._progress_bar.value() == 50
        assert tracker._step_label.text() == "5/10"
        tracker.deleteLater()

    def test_set_progress_completion(self):
        """Test set_progress with completion."""
        tracker = ProgressTracker()
        tracker.set_progress(10, 10, "Complete")

        assert tracker._progress_bar.value() == 100
        assert "Completed" in tracker._status_label.text()
        tracker.deleteLater()

    def test_set_status_info(self):
        """Test set_status with info type."""
        tracker = ProgressTracker()
        tracker.set_status("Processing", "info")
        assert tracker._status_label.text() == "Processing"
        tracker.deleteLater()

    def test_set_status_success(self):
        """Test set_status with success type."""
        tracker = ProgressTracker()
        tracker.set_status("Done", "success")
        assert tracker._status_label.text() == "Done"
        tracker.deleteLater()

    def test_set_status_warning(self):
        """Test set_status with warning type."""
        tracker = ProgressTracker()
        tracker.set_status("Warning", "warning")
        assert tracker._status_label.text() == "Warning"
        tracker.deleteLater()

    def test_set_status_error(self):
        """Test set_status with error type."""
        tracker = ProgressTracker()
        tracker.set_status("Error", "error")
        assert tracker._status_label.text() == "Error"
        tracker.deleteLater()

    def test_add_step_detail(self):
        """Test add_step_detail method."""
        tracker = ProgressTracker()
        tracker.add_step_detail(1, "First step", "completed")
        tracker.add_step_detail(2, "Second step", "running")

        assert len(tracker._step_details) == 2
        assert tracker._step_details[0]["number"] == 1
        assert tracker._step_details[0]["description"] == "First step"
        assert tracker._step_details[0]["status"] == "completed"
        tracker.deleteLater()

    def test_update_step_status(self):
        """Test update_step_status method."""
        tracker = ProgressTracker()
        tracker.add_step_detail(1, "First step", "pending")
        tracker.update_step_status(1, "completed")

        assert tracker._step_details[0]["status"] == "completed"
        tracker.deleteLater()

    def test_clear(self):
        """Test clear method."""
        tracker = ProgressTracker()
        tracker.set_progress(5, 10, "Halfway")
        tracker.add_step_detail(1, "Step 1", "completed")

        tracker.clear()

        assert tracker._current_step == 0
        assert tracker._total_steps == 0
        assert tracker._progress_bar.value() == 0
        assert len(tracker._step_details) == 0
        tracker.deleteLater()

    def test_get_progress_percent(self):
        """Test get_progress_percent method."""
        tracker = ProgressTracker()
        tracker.set_progress(5, 10)
        assert tracker.get_progress_percent() == 50.0
        tracker.deleteLater()

    def test_get_progress_percent_zero(self):
        """Test get_progress_percent with zero total."""
        tracker = ProgressTracker()
        assert tracker.get_progress_percent() == 0.0
        tracker.deleteLater()

    def test_is_completed(self):
        """Test is_completed method."""
        tracker = ProgressTracker()
        tracker.set_progress(10, 10)
        assert tracker.is_completed() is True
        tracker.deleteLater()

    def test_is_completed_not_complete(self):
        """Test is_completed when not complete."""
        tracker = ProgressTracker()
        tracker.set_progress(5, 10)
        assert tracker.is_completed() is False
        tracker.deleteLater()

    def test_toggle_details(self):
        """Test _toggle_details method."""
        tracker = ProgressTracker()
        # Initially hidden
        assert tracker._details_container.isVisible() is False

        # Toggle to show
        tracker._toggle_details()
        assert tracker._details_container.isVisible() is True
        assert tracker._expanded is True

        # Toggle to hide
        tracker._toggle_details()
        assert tracker._details_container.isVisible() is False
        assert tracker._expanded is False
        tracker.deleteLater()

    def test_update_details_display(self):
        """Test _update_details_display method."""
        tracker = ProgressTracker()
        tracker.add_step_detail(1, "Step 1", "completed")
        tracker.add_step_detail(2, "Step 2", "running")

        tracker._update_details_display()

        text = tracker._steps_text.toPlainText()
        assert "Step 1" in text
        assert "Step 2" in text
        tracker.deleteLater()


class TestCircularProgress:
    """Test CircularProgress class."""

    def test_circular_progress_creation(self):
        """Test CircularProgress can be created."""
        progress = CircularProgress(diameter=40)
        assert progress is not None
        assert progress._diameter == 40
        assert progress._progress == 0
        progress.deleteLater()

    def test_set_progress(self):
        """Test set_progress method."""
        progress = CircularProgress()
        progress.set_progress(75)
        assert progress._progress == 75
        progress.deleteLater()

    def test_set_progress_clamping(self):
        """Test set_progress clamps values."""
        progress = CircularProgress()
        progress.set_progress(150)
        assert progress._progress == 100

        progress.set_progress(-10)
        assert progress._progress == 0
        progress.deleteLater()

    def test_set_color(self):
        """Test set_color method."""
        progress = CircularProgress()
        progress.set_color("#FF0000")
        assert progress._color == "#FF0000"
        progress.deleteLater()


class TestCreateProgressTracker:
    """Test create_progress_tracker function."""

    def test_create_progress_tracker(self):
        """Test create_progress_tracker creates correct instance."""
        tracker = create_progress_tracker()
        assert isinstance(tracker, ProgressTracker)
        tracker.deleteLater()
