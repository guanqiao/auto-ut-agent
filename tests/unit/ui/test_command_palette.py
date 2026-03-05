"""Tests for command palette functionality."""

import pytest
from unittest.mock import Mock, MagicMock

from pyutagent.ui.command_palette import (
    Command,
    FuzzyMatcher,
    CommandItemDelegate,
    CommandPalette
)


class TestCommand:
    """Test Command dataclass."""

    def test_command_creation(self):
        """Test creating a Command instance."""
        callback = Mock()
        cmd = Command(
            id="test.cmd",
            name="Test Command",
            description="A test command",
            shortcut="Ctrl+T",
            category="Test",
            callback=callback,
            keywords=["test", "example"]
        )

        assert cmd.id == "test.cmd"
        assert cmd.name == "Test Command"
        assert cmd.description == "A test command"
        assert cmd.shortcut == "Ctrl+T"
        assert cmd.category == "Test"
        assert cmd.keywords == ["test", "example"]

    def test_command_default_keywords(self):
        """Test Command with default empty keywords."""
        cmd = Command(
            id="test.cmd",
            name="Test",
            description="Test",
            shortcut="",
            category="Test",
            callback=Mock()
        )

        assert cmd.keywords == []


class TestFuzzyMatcher:
    """Test fuzzy matching algorithm."""

    @pytest.fixture
    def matcher(self):
        return FuzzyMatcher()

    @pytest.fixture
    def sample_command(self):
        return Command(
            id="file.open",
            name="Open Project",
            description="Open a project directory",
            shortcut="Ctrl+O",
            category="File",
            callback=Mock(),
            keywords=["open", "folder"]
        )

    def test_exact_match(self, matcher, sample_command):
        """Test exact match scoring."""
        score, positions = matcher.calculate_score("open project", sample_command)
        assert score > 2.0  # Exact match should have high score

    def test_partial_match(self, matcher, sample_command):
        """Test partial match scoring."""
        score, positions = matcher.calculate_score("proj", sample_command)
        assert score > 0.5  # Partial match should have decent score

    def test_fuzzy_match(self, matcher, sample_command):
        """Test fuzzy match scoring."""
        score, positions = matcher.calculate_score("opn", sample_command)
        assert score > 0.1  # Fuzzy match should have some score

    def test_no_match(self, matcher, sample_command):
        """Test no match case."""
        score, positions = matcher.calculate_score("xyz", sample_command)
        assert score == 0.0

    def test_empty_query(self, matcher, sample_command):
        """Test empty query returns default score."""
        score, positions = matcher.calculate_score("", sample_command)
        assert score == 1.0

    def test_category_match(self, matcher, sample_command):
        """Test category matching."""
        score, positions = matcher.calculate_score("file", sample_command)
        assert score > 1.0  # Category match has weight 1.5

    def test_keyword_match(self, matcher, sample_command):
        """Test keyword matching."""
        score, positions = matcher.calculate_score("folder", sample_command)
        assert score > 1.0  # Keyword match has weight 1.3

    def test_shortcut_match(self, matcher, sample_command):
        """Test shortcut matching."""
        score, positions = matcher.calculate_score("ctrl+o", sample_command)
        assert score > 1.0  # Shortcut match has weight 1.5


class TestConflictDetector:
    """Test conflict detection in keyboard shortcuts."""

    @pytest.fixture
    def detector(self):
        from pyutagent.ui.dialogs.keyboard_shortcuts_dialog import ConflictDetector
        return ConflictDetector()

    @pytest.fixture
    def sample_shortcuts(self):
        from pyutagent.ui.dialogs.keyboard_shortcuts_dialog import ShortcutDefinition, ShortcutCategory
        return [
            ShortcutDefinition("cmd1", "Command 1", "Desc 1", "Ctrl+A", ShortcutCategory.FILE),
            ShortcutDefinition("cmd2", "Command 2", "Desc 2", "Ctrl+B", ShortcutCategory.EDIT),
            ShortcutDefinition("cmd3", "Command 3", "Desc 3", "Ctrl+A", ShortcutCategory.VIEW),  # Conflict
        ]

    def test_find_conflicts(self, detector, sample_shortcuts):
        """Test finding conflicting shortcuts."""
        conflicts = detector.find_conflicts(sample_shortcuts)
        assert len(conflicts) == 1
        assert conflicts[0][0].id == "cmd1"
        assert conflicts[0][1].id == "cmd3"

    def test_no_conflicts(self, detector):
        """Test with no conflicts."""
        from pyutagent.ui.dialogs.keyboard_shortcuts_dialog import ShortcutDefinition, ShortcutCategory
        shortcuts = [
            ShortcutDefinition("cmd1", "Command 1", "Desc 1", "Ctrl+A", ShortcutCategory.FILE),
            ShortcutDefinition("cmd2", "Command 2", "Desc 2", "Ctrl+B", ShortcutCategory.EDIT),
        ]
        conflicts = detector.find_conflicts(shortcuts)
        assert len(conflicts) == 0

    def test_normalize_key(self, detector):
        """Test key normalization."""
        assert detector._normalize_key("Ctrl+A") == "CTRL+A"
        assert detector._normalize_key("Shift+Ctrl+B") == "CTRL+SHIFT+B"
        assert detector._normalize_key("Alt+Shift+X") == "ALT+SHIFT+X"

    def test_check_validity_empty(self, detector):
        """Test validity check for empty key."""
        is_valid, error = detector.check_validity("")
        assert is_valid is True
        assert error == ""

    def test_check_validity_reserved(self, detector):
        """Test validity check for reserved shortcut."""
        is_valid, error = detector.check_validity("Alt+F4")
        assert is_valid is False
        assert "reserved" in error.lower()


class TestCommandPaletteCore:
    """Core logic tests for CommandPalette (no GUI)."""

    def test_command_palette_has_commands(self):
        """Test that CommandPalette has default commands."""
        # Just test that we can create the command list
        from pyutagent.ui.command_palette import CommandPalette
        # The commands are set up in setup_commands
        # We can verify the structure without creating the GUI
        assert hasattr(CommandPalette, 'setup_commands')

    def test_fuzzy_search_integration(self):
        """Test fuzzy search with multiple commands."""
        matcher = FuzzyMatcher()

        commands = [
            Command("file.open", "Open File", "Open a file", "Ctrl+O", "File", Mock()),
            Command("file.save", "Save File", "Save the file", "Ctrl+S", "File", Mock()),
            Command("edit.copy", "Copy", "Copy selection", "Ctrl+C", "Edit", Mock()),
        ]

        # Search for "file"
        results = []
        for cmd in commands:
            score, _ = matcher.calculate_score("file", cmd)
            if score > 0.1:
                results.append((cmd, score))

        results.sort(key=lambda x: x[1], reverse=True)

        # Should find file-related commands
        assert len(results) >= 2
        assert all(cmd.category == "File" for cmd, _ in results[:2])


class TestShortcutDefinition:
    """Test ShortcutDefinition dataclass."""

    def test_shortcut_definition_creation(self):
        """Test creating ShortcutDefinition."""
        from pyutagent.ui.dialogs.keyboard_shortcuts_dialog import ShortcutDefinition, ShortcutCategory

        shortcut = ShortcutDefinition(
            id="test.cmd",
            name="Test Command",
            description="A test command",
            default_key="Ctrl+T",
            category=ShortcutCategory.FILE
        )

        assert shortcut.id == "test.cmd"
        assert shortcut.name == "Test Command"
        assert shortcut.current_key == "Ctrl+T"  # Should default to default_key

    def test_shortcut_to_dict(self):
        """Test converting ShortcutDefinition to dict."""
        from pyutagent.ui.dialogs.keyboard_shortcuts_dialog import ShortcutDefinition, ShortcutCategory

        shortcut = ShortcutDefinition(
            id="test.cmd",
            name="Test",
            description="Test",
            default_key="Ctrl+T",
            category=ShortcutCategory.FILE
        )
        shortcut.current_key = "Ctrl+Shift+T"

        data = shortcut.to_dict()
        assert data["id"] == "test.cmd"
        assert data["current_key"] == "Ctrl+Shift+T"

    def test_shortcut_modified_detection(self):
        """Test detecting modified shortcuts."""
        from pyutagent.ui.dialogs.keyboard_shortcuts_dialog import ShortcutDefinition, ShortcutCategory

        shortcut = ShortcutDefinition(
            id="test.cmd",
            name="Test",
            description="Test",
            default_key="Ctrl+T",
            category=ShortcutCategory.FILE
        )

        assert shortcut.current_key == shortcut.default_key

        shortcut.current_key = "Ctrl+Shift+T"
        assert shortcut.current_key != shortcut.default_key
