"""Unit tests for file tree widget."""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

from pyutagent.ui.widgets.file_tree import FileTree, FileTreeItem
from pyutagent.ui.services.git_status_service import GitStatusService, GitStatus


@pytest.fixture
def app():
    """Create QApplication for tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def temp_project():
    """Create a temporary project structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create some files
        (Path(tmpdir) / "main.py").write_text("print('hello')")
        (Path(tmpdir) / "README.md").write_text("# Test")
        
        # Create subdirectory
        subdir = Path(tmpdir) / "src"
        subdir.mkdir()
        (subdir / "utils.py").write_text("def helper(): pass")
        
        yield tmpdir


class TestFileTreeItem:
    """Tests for FileTreeItem class."""
    
    def test_init(self, app):
        """Test item initialization."""
        item = FileTreeItem(texts=["test.py"])
        assert item._original_text == "test.py"
        assert item._git_status is None
        assert item._is_highlighted is False
        
    def test_set_git_status(self, app):
        """Test setting Git status."""
        item = FileTreeItem(texts=["test.py"])
        
        # Test modified status
        item.set_git_status(GitStatus.MODIFIED)
        assert item.get_git_status() == GitStatus.MODIFIED
        assert "[M]" in item.text(0)
        
        # Test added status
        item.set_git_status(GitStatus.ADDED)
        assert item.get_git_status() == GitStatus.ADDED
        assert "[A]" in item.text(0)
        
    def test_set_highlighted(self, app):
        """Test setting highlight state."""
        item = FileTreeItem(texts=["test.py"])
        item.set_highlighted(True)
        assert item._is_highlighted is True
        
    def test_unmodified_status_no_indicator(self, app):
        """Test that unmodified status doesn't show indicator."""
        item = FileTreeItem(texts=["test.py"])
        item.set_git_status(GitStatus.UNMODIFIED)
        assert "[" not in item.text(0)  # No status indicator


class TestFileTree:
    """Tests for FileTree widget."""
    
    def test_init(self, app):
        """Test widget initialization."""
        tree = FileTree()
        assert tree._project_path == ""
        assert tree._all_items == []
        assert tree._search_text == ""
        
    def test_load_project(self, app, temp_project):
        """Test loading a project."""
        tree = FileTree()
        tree.load_project(temp_project)
        
        assert tree._project_path == temp_project
        # Python projects may load both root and src/ directories
        # We should have at least the 3 unique files
        file_paths = {path for _, _, path in tree._all_items}
        assert len(file_paths) == 3  # main.py, README.md, utils.py
        
    def test_fuzzy_match_substring(self, app):
        """Test fuzzy matching with substring."""
        tree = FileTree()
        
        # Substring match
        assert tree._fuzzy_match("main", "main.py") is True
        assert tree._fuzzy_match("utils", "src/utils.py") is True
        
    def test_fuzzy_match_character_order(self, app):
        """Test fuzzy matching with character order."""
        tree = FileTree()
        
        # Character order match
        assert tree._fuzzy_match("mp", "main.py") is True  # m...p
        assert tree._fuzzy_match("up", "utils.py") is True  # u...p
        
    def test_fuzzy_match_no_match(self, app):
        """Test fuzzy matching with no match."""
        tree = FileTree()
        
        # No match
        assert tree._fuzzy_match("xyz", "main.py") is False
        
    def test_search_filtering(self, app, temp_project):
        """Test search filtering."""
        tree = FileTree()
        tree.load_project(temp_project)
        
        # Apply search
        tree._search_text = "main"
        tree._apply_search()
        
        # Check that non-matching items are hidden
        for name, item, path in tree._all_items:
            if "main" not in name:
                assert item.isHidden() is True
            else:
                assert item.isHidden() is False
                
    def test_clear_search(self, app, temp_project):
        """Test clearing search."""
        tree = FileTree()
        tree.load_project(temp_project)
        
        # Apply and then clear search
        tree._search_text = "main"
        tree._apply_search()
        tree._search_text = ""
        tree._apply_search()
        
        # All items should be visible
        for _, item, _ in tree._all_items:
            assert item.isHidden() is False
            
    def test_get_selected_path(self, app, temp_project):
        """Test getting selected path."""
        tree = FileTree()
        tree.load_project(temp_project)
        
        # Initially nothing selected
        assert tree.get_selected_path() is None
        
    def test_select_path(self, app, temp_project):
        """Test selecting a path."""
        tree = FileTree()
        tree.load_project(temp_project)
        
        main_py = str(Path(temp_project) / "main.py")
        tree.select_path(main_py)
        
        assert tree.get_selected_path() == main_py
        
    def test_refresh(self, app, temp_project):
        """Test refreshing the tree."""
        tree = FileTree()
        tree.load_project(temp_project)
        
        # Get initial unique file count
        initial_paths = {path for _, _, path in tree._all_items}
        
        # Add a new file
        (Path(temp_project) / "new_file.py").write_text("# new")
        
        # Refresh
        tree.refresh()
        
        # Should now have 4 unique files
        file_paths = {path for _, _, path in tree._all_items}
        assert len(file_paths) == len(initial_paths) + 1


class TestGitStatusService:
    """Tests for GitStatusService."""
    
    def test_init(self):
        """Test service initialization."""
        service = GitStatusService()
        assert service._status_cache == {}
        assert service._repo_root is None
        
    def test_detect_repo_non_git(self):
        """Test detecting non-git directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            service = GitStatusService()
            result = service.detect_repo(tmpdir)
            assert result is False
            
    def test_get_file_status_no_repo(self):
        """Test getting status when not in repo."""
        service = GitStatusService()
        result = service.get_file_status("/some/path")
        assert result is None
        
    def test_refresh_all_status_no_repo(self):
        """Test refreshing status when not in repo."""
        service = GitStatusService()
        result = service.refresh_all_status()
        assert result == {}
        
    def test_clear_cache(self):
        """Test clearing cache."""
        service = GitStatusService()
        service._status_cache = {"/path": GitStatus.MODIFIED}
        service.clear_cache()
        assert service._status_cache == {}
        
    def test_git_status_properties(self):
        """Test GitStatus enum properties."""
        assert GitStatus.MODIFIED.color == "#E2C08D"
        assert GitStatus.ADDED.color == "#73C991"
        assert GitStatus.DELETED.color == "#F85149"
        
        assert GitStatus.MODIFIED.icon == "M"
        assert GitStatus.ADDED.icon == "A"
        assert GitStatus.DELETED.icon == "D"
        
        assert GitStatus.MODIFIED.display_name == "Modified"
        assert GitStatus.ADDED.display_name == "Added"


class TestFileTreeSignals:
    """Tests for FileTree signals."""
    
    def test_file_selected_signal(self, app, temp_project):
        """Test file selected signal."""
        tree = FileTree()
        tree.load_project(temp_project)
        
        mock_handler = Mock()
        tree.file_selected.connect(mock_handler)
        
        # Find and click a file item
        for name, item, path in tree._all_items:
            if "main.py" in path:
                tree._on_item_clicked(item, 0)
                break
                
        mock_handler.assert_called_once()
        
    def test_file_activated_signal(self, app, temp_project):
        """Test file activated signal."""
        tree = FileTree()
        tree.load_project(temp_project)
        
        mock_handler = Mock()
        tree.file_activated.connect(mock_handler)
        
        # Find and double-click a file item
        for name, item, path in tree._all_items:
            if "main.py" in path:
                tree._on_item_double_clicked(item, 0)
                break
                
        mock_handler.assert_called_once()
        
    def test_file_dragged_signal(self, app, temp_project):
        """Test file dragged signal."""
        tree = FileTree()
        tree.load_project(temp_project)
        
        mock_handler = Mock()
        tree.file_dragged.connect(mock_handler)
        
        # Select a file and start drag
        for name, item, path in tree._all_items:
            if "main.py" in path:
                tree._tree.setCurrentItem(item)
                # Note: Can't fully test drag without GUI interaction
                break


class TestFileTreeGitIntegration:
    """Tests for Git integration in FileTree."""
    
    @patch.object(GitStatusService, 'detect_repo')
    @patch.object(GitStatusService, 'refresh_all_status')
    def test_load_project_with_git(self, mock_refresh, mock_detect, app, temp_project):
        """Test loading project in git repo."""
        mock_detect.return_value = True
        mock_refresh.return_value = {}
        
        tree = FileTree()
        tree.load_project(temp_project)
        
        mock_detect.assert_called_once_with(temp_project)
        mock_refresh.assert_called_once()
        
    def test_git_status_display(self, app):
        """Test Git status display on items."""
        item = FileTreeItem(texts=["test.py"])
        
        item.set_git_status(GitStatus.MODIFIED)
        assert "[M]" in item.text(0)
        
        item.set_git_status(GitStatus.ADDED)
        assert "[A]" in item.text(0)
        
        item.set_git_status(GitStatus.DELETED)
        assert "[D]" in item.text(0)
