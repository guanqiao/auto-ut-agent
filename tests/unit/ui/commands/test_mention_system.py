"""Tests for the mention system module."""

import tempfile
from pathlib import Path

import pytest

# Skip Qt tests if not available
pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication, QLineEdit, QTextEdit, QWidget
from PyQt6.QtCore import Qt

from pyutagent.ui.commands.mention_system import (
    MentionType,
    MentionItem,
    EnhancedMentionSystem,
)
from pyutagent.indexing.codebase_indexer import (
    IndexerConfig,
    CodebaseIndexer,
)


class TestMentionItem:
    """Tests for MentionItem dataclass."""

    def test_mention_item_creation(self):
        """Test creating a MentionItem."""
        item = MentionItem(
            id="file_001",
            name="Test.java",
            mention_type=MentionType.FILE,
            icon="📄",
            path="src/Test.java",
            description="Test file",
        )

        assert item.id == "file_001"
        assert item.name == "Test.java"
        assert item.mention_type == MentionType.FILE
        assert item.display_text == "@file:src/Test.java"

    def test_mention_item_symbol_display(self):
        """Test display text for symbol mentions."""
        item = MentionItem(
            id="sym_001",
            name="TestClass",
            mention_type=MentionType.CODE,
            icon="🔷",
            path="/test/Test.java",
            description="A test class",
        )

        assert item.display_text == "@code:TestClass"

    def test_mention_item_special_display(self):
        """Test display text for special mentions."""
        item = MentionItem(
            id="current",
            name="current",
            mention_type=MentionType.CURRENT,
            icon="📄",
        )

        assert item.display_text == "@current"


class TestMentionSystemParsing:
    """Tests for mention parsing functionality."""

    @pytest.fixture
    def mention_system(self):
        """Create a mention system instance."""
        return EnhancedMentionSystem()

    def test_parse_simple_mention(self, mention_system):
        """Test parsing simple @-mentions."""
        text = "Check @current file"
        mentions = mention_system.parse_mentions(text)

        assert len(mentions) == 1
        assert mentions[0]["type"] == "current"
        assert mentions[0]["text"] == "@current"

    def test_parse_file_mention(self, mention_system):
        """Test parsing @file mentions."""
        text = "Look at @file:src/main/java/Test.java"
        mentions = mention_system.parse_mentions(text)

        assert len(mentions) == 1
        assert mentions[0]["type"] == "file"
        assert mentions[0]["id"] == "src/main/java/Test.java"

    def test_parse_code_mention(self, mention_system):
        """Test parsing @code mentions."""
        text = "Check @code:UserService.getUserById method"
        mentions = mention_system.parse_mentions(text)

        assert len(mentions) == 1
        assert mentions[0]["type"] == "code"
        assert mentions[0]["id"] == "UserService.getUserById"

    def test_parse_multiple_mentions(self, mention_system):
        """Test parsing multiple mentions."""
        text = "Check @current and @file:Test.java also @workspace"
        mentions = mention_system.parse_mentions(text)

        assert len(mentions) == 3
        types = {m["type"] for m in mentions}
        assert types == {"current", "file", "workspace"}

    def test_parse_symbol_without_prefix(self, mention_system):
        """Test parsing symbol mentions without explicit prefix."""
        text = "Check @TestClass for details"
        mentions = mention_system.parse_mentions(text)

        assert len(mentions) == 1
        # Should be inferred as symbol type
        assert mentions[0]["type"] in ["symbol", "code"]

    def test_parse_no_mentions(self, mention_system):
        """Test parsing text with no mentions."""
        text = "This is just regular text"
        mentions = mention_system.parse_mentions(text)

        assert len(mentions) == 0


class TestMentionSystemWithIndexer:
    """Tests for mention system with codebase indexer."""

    @pytest.fixture
    def temp_project(self):
        """Create a temporary project with Java files."""
        tmpdir = tempfile.mkdtemp()
        try:
            project_path = Path(tmpdir)

            # Create source directory
            src_dir = project_path / "src" / "main" / "java" / "com" / "example"
            src_dir.mkdir(parents=True)

            # Create a Java file
            java_file = src_dir / "UserService.java"
            java_file.write_text('''
package com.example;

public class UserService {
    public User getUserById(Long id) {
        return new User();
    }
}
''')

            # Create indexer
            config = IndexerConfig(enable_semantic_search=False)
            indexer = CodebaseIndexer(str(project_path), config=config)
            indexer.index_project()

            yield project_path, indexer
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_resolve_file_reference(self, temp_project):
        """Test resolving file references."""
        project_path, indexer = temp_project
        mention_system = EnhancedMentionSystem()
        mention_system.set_codebase_indexer(indexer)
        mention_system.set_project_path(str(project_path))

        resolved = mention_system.resolve_mention("file:src/main/java/com/example/UserService.java")

        assert resolved is not None
        assert resolved["type"] == "file"

    def test_resolve_symbol_reference(self, temp_project):
        """Test resolving symbol references."""
        project_path, indexer = temp_project
        mention_system = EnhancedMentionSystem()
        mention_system.set_codebase_indexer(indexer)

        resolved = mention_system.resolve_mention("symbol:UserService")

        assert resolved is not None
        assert resolved["type"] == "symbol"

    def test_get_mention_context_for_symbol(self, temp_project):
        """Test getting context for a symbol mention."""
        project_path, indexer = temp_project
        mention_system = EnhancedMentionSystem()
        mention_system.set_codebase_indexer(indexer)

        context = mention_system.get_mention_context("symbol:UserService")

        assert "UserService" in context
        assert "Symbol:" in context


class TestMentionSystemFileDiscovery:
    """Tests for file and folder discovery."""

    @pytest.fixture
    def temp_project(self):
        """Create a temporary project."""
        tmpdir = tempfile.mkdtemp()
        try:
            project_path = Path(tmpdir)

            # Create some files
            (project_path / "src").mkdir()
            (project_path / "src" / "main.java").write_text("public class Main {}")
            (project_path / "src" / "utils.java").write_text("public class Utils {}")
            (project_path / "test").mkdir()
            (project_path / "test" / "test.java").write_text("public class Test {}")

            yield project_path
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_get_project_files(self, temp_project):
        """Test getting project files."""
        mention_system = EnhancedMentionSystem()
        mention_system.set_project_path(str(temp_project))

        items = mention_system._get_project_files("")

        assert len(items) >= 3  # At least 3 Java files
        assert any("main.java" in f for f in items)

    def test_get_project_folders(self, temp_project):
        """Test getting project folders."""
        mention_system = EnhancedMentionSystem()
        mention_system.set_project_path(str(temp_project))

        folders = mention_system._get_project_folders("")

        assert len(folders) >= 2  # src and test folders

    def test_file_icon_mapping(self, temp_project):
        """Test file icon mapping."""
        mention_system = EnhancedMentionSystem()

        assert mention_system._get_file_icon(".java") == "☕"
        assert mention_system._get_file_icon(".py") == "🐍"
        assert mention_system._get_file_icon(".js") == "📜"
        assert mention_system._get_file_icon(".unknown") == "📄"


class TestMentionSystemIntegration:
    """Integration tests for mention system."""

    @pytest.fixture(scope="module")
    def qapp(self):
        """Create QApplication for Qt tests."""
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        yield app

    @pytest.fixture
    def mention_system_with_ui(self, qapp):
        """Create mention system with UI components."""
        parent = QWidget()
        line_edit = QLineEdit()

        mention_system = EnhancedMentionSystem()
        mention_system.attach_to_input(line_edit, parent)

        yield mention_system, line_edit, parent

        parent.deleteLater()

    def test_attach_to_input(self, mention_system_with_ui):
        """Test attaching to input widget."""
        mention_system, line_edit, parent = mention_system_with_ui

        assert mention_system._current_input == line_edit
        assert mention_system._popup is not None

    def test_is_active_initially_false(self, mention_system_with_ui):
        """Test that mention system is not active initially."""
        mention_system, line_edit, parent = mention_system_with_ui

        assert mention_system.is_active() is False
