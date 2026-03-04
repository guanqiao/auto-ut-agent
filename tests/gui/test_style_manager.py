"""Tests for the style manager."""

import pytest
from PyQt6.QtWidgets import QApplication, QWidget

from pyutagent.ui.styles import StyleManager, get_style_manager


@pytest.mark.gui
class TestStyleManager:
    """Test suite for StyleManager."""
    
    def test_singleton_pattern(self):
        """Test that StyleManager is a singleton."""
        manager1 = StyleManager()
        manager2 = StyleManager()
        assert manager1 is manager2
    
    def test_get_style_manager(self):
        """Test get_style_manager function."""
        manager = get_style_manager()
        assert isinstance(manager, StyleManager)
        assert manager is get_style_manager()  # Should return same instance
    
    def test_default_theme(self):
        """Test default theme is set."""
        manager = StyleManager()
        assert manager.current_theme in ["light", "dark"]
    
    def test_get_available_themes(self):
        """Test getting available themes."""
        manager = StyleManager()
        themes = manager.get_available_themes()
        assert isinstance(themes, list)
        assert "light" in themes or "dark" in themes
    
    def test_set_theme(self, qtbot):
        """Test setting theme."""
        manager = StyleManager()
        available_themes = manager.get_available_themes()
        
        if len(available_themes) > 1:
            new_theme = available_themes[1] if available_themes[0] == manager.current_theme else available_themes[0]
            result = manager.set_theme(new_theme)
            assert result is True
            assert manager.current_theme == new_theme
    
    def test_set_invalid_theme(self):
        """Test setting invalid theme."""
        manager = StyleManager()
        result = manager.set_theme("nonexistent_theme")
        assert result is False
    
    def test_get_color(self):
        """Test getting color from theme."""
        manager = StyleManager()
        color = manager.get_color("primary")
        assert isinstance(color, str)
        assert color.startswith("#")
    
    def test_get_color_default(self):
        """Test getting color with default value."""
        manager = StyleManager()
        color = manager.get_color("nonexistent_color", "#000000")
        assert color == "#000000"
    
    def test_get_font_family(self):
        """Test getting font family."""
        manager = StyleManager()
        font = manager.get_font_family()
        assert isinstance(font, str)
        assert len(font) > 0
    
    def test_get_mono_font_family(self):
        """Test getting monospace font family."""
        manager = StyleManager()
        font = manager.get_mono_font_family()
        assert isinstance(font, str)
        assert len(font) > 0
    
    def test_get_stylesheet(self):
        """Test getting stylesheet."""
        manager = StyleManager()
        
        # Test known widget types
        for widget_type in ["main_window", "button", "input", "card"]:
            stylesheet = manager.get_stylesheet(widget_type)
            assert isinstance(stylesheet, str)
    
    def test_get_stylesheet_unknown(self):
        """Test getting stylesheet for unknown widget type."""
        manager = StyleManager()
        stylesheet = manager.get_stylesheet("unknown_widget")
        assert stylesheet == ""
    
    def test_apply_stylesheet(self, qtbot):
        """Test applying stylesheet to widget."""
        manager = StyleManager()
        widget = QWidget()
        qtbot.addWidget(widget)
        
        # Should not raise
        manager.apply_stylesheet(widget, "main_window")
    
    def test_get_status_color(self):
        """Test getting status color."""
        manager = StyleManager()
        
        for status in ["success", "warning", "error", "info", "pending"]:
            color = manager.get_status_color(status)
            assert isinstance(color, str)
            assert color.startswith("#")
    
    def test_get_status_color_unknown(self):
        """Test getting status color for unknown status."""
        manager = StyleManager()
        color = manager.get_status_color("unknown_status")
        assert isinstance(color, str)
        assert color.startswith("#")
    
    def test_theme_change_signal(self, qtbot):
        """Test theme change signal is emitted."""
        manager = StyleManager()
        available_themes = manager.get_available_themes()
        
        if len(available_themes) > 1:
            new_theme = available_themes[1] if available_themes[0] == manager.current_theme else available_themes[0]
            
            with qtbot.waitSignal(manager.theme_changed, timeout=1000):
                manager.set_theme(new_theme)
