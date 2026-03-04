"""Style manager for unified theming across the application."""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from PyQt6.QtWidgets import QApplication, QWidget
from PyQt6.QtCore import QObject, pyqtSignal

logger = logging.getLogger(__name__)


class StyleManager:
    """Manages application-wide styling and theming."""

    _instance: Optional['StyleManager'] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._current_theme = "light"
        self._themes: Dict[str, Dict[str, Any]] = {}
        self._callbacks = []
        self._load_themes()

    def _load_themes(self):
        """Load all available themes from the themes directory."""
        themes_dir = Path(__file__).parent / "themes"

        if not themes_dir.exists():
            logger.warning(f"Themes directory not found: {themes_dir}")
            return

        for theme_file in themes_dir.glob("*.json"):
            try:
                with open(theme_file, 'r', encoding='utf-8') as f:
                    theme_data = json.load(f)
                    theme_name = theme_file.stem.replace('_theme', '')
                    self._themes[theme_name] = theme_data
                    logger.debug(f"Loaded theme: {theme_name}")
            except Exception as e:
                logger.error(f"Failed to load theme {theme_file}: {e}")

        if not self._themes:
            logger.warning("No themes loaded, using defaults")
            self._themes["light"] = self._get_default_light_theme()
            self._themes["dark"] = self._get_default_dark_theme()

    def _get_default_light_theme(self) -> Dict[str, Any]:
        """Get default light theme."""
        return {
            "name": "Light",
            "colors": {
                "primary": "#2196F3",
                "success": "#4CAF50",
                "warning": "#FF9800",
                "error": "#F44336",
                "background": "#FAFAFA",
                "surface": "#FFFFFF",
                "card": "#FFFFFF",
                "border": "#E0E0E0",
                "text_primary": "#212121",
                "text_secondary": "#757575",
                "text_on_primary": "#FFFFFF",
                "hover": "#F5F5F5",
                "selected": "#E3F2FD",
            },
            "fonts": {
                "family": "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
                "mono_family": "'Consolas', 'Monaco', monospace",
            },
            "border_radius": {"small": "4px", "medium": "8px"},
        }

    def _get_default_dark_theme(self) -> Dict[str, Any]:
        """Get default dark theme."""
        return {
            "name": "Dark",
            "colors": {
                "primary": "#2196F3",
                "success": "#4CAF50",
                "warning": "#FF9800",
                "error": "#F44336",
                "background": "#1E1E1E",
                "surface": "#252526",
                "card": "#2D2D2D",
                "border": "#3C3C3C",
                "text_primary": "#E0E0E0",
                "text_secondary": "#A0A0A0",
                "text_on_primary": "#FFFFFF",
                "hover": "#3C3C3C",
                "selected": "#094771",
            },
            "fonts": {
                "family": "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
                "mono_family": "'Consolas', 'Monaco', monospace",
            },
            "border_radius": {"small": "4px", "medium": "8px"},
        }

    @property
    def current_theme(self) -> str:
        """Get current theme name."""
        return self._current_theme

    @property
    def theme_data(self) -> Dict[str, Any]:
        """Get current theme data."""
        return self._themes.get(self._current_theme, self._themes.get("light", {}))

    def set_theme(self, theme_name: str) -> bool:
        """Set the current theme."""
        if theme_name not in self._themes:
            logger.error(f"Theme not found: {theme_name}")
            return False

        self._current_theme = theme_name
        # Notify callbacks
        for callback in self._callbacks:
            try:
                callback(theme_name)
            except Exception as e:
                logger.error(f"Error notifying theme change callback: {e}")
        logger.info(f"Theme changed to: {theme_name}")
        return True

    def on_theme_changed(self, callback):
        """Register a callback for theme changes."""
        self._callbacks.append(callback)

    def get_available_themes(self) -> list:
        """Get list of available theme names."""
        return list(self._themes.keys())

    def get_color(self, color_name: str, default: str = "#000000") -> str:
        """Get a color from the current theme."""
        colors = self.theme_data.get("colors", {})
        return colors.get(color_name, default)

    def get_font_family(self) -> str:
        """Get the font family from current theme."""
        fonts = self.theme_data.get("fonts", {})
        return fonts.get("family", "sans-serif")

    def get_mono_font_family(self) -> str:
        """Get the monospace font family from current theme."""
        fonts = self.theme_data.get("fonts", {})
        return fonts.get("mono_family", "monospace")

    def get_border_radius(self, size: str = "medium") -> str:
        """Get border radius from current theme."""
        radii = self.theme_data.get("border_radius", {})
        return radii.get(size, "4px")

    def get_stylesheet(self, widget_type: str) -> str:
        """Get a complete stylesheet for a widget type."""
        c = self.theme_data.get("colors", {})

        stylesheets = {
            "main_window": f"""
                QMainWindow {{
                    background-color: {c.get('background', '#FAFAFA')};
                }}
            """,
            "widget": f"""
                QWidget {{
                    background-color: {c.get('background', '#FAFAFA')};
                    color: {c.get('text_primary', '#212121')};
                    font-family: {self.get_font_family()};
                }}
            """,
            "button": f"""
                QPushButton {{
                    background-color: {c.get('primary', '#2196F3')};
                    color: {c.get('text_on_primary', '#FFFFFF')};
                    border: none;
                    border-radius: {self.get_border_radius('small')};
                    padding: 8px 16px;
                    font-weight: 500;
                }}
                QPushButton:hover {{
                    background-color: {c.get('primary_dark', '#1976D2')};
                }}
                QPushButton:pressed {{
                    background-color: {c.get('primary_dark', '#1565C0')};
                }}
                QPushButton:disabled {{
                    background-color: {c.get('text_disabled', '#BDBDBD')};
                    color: {c.get('text_secondary', '#757575')};
                }}
            """,
            "button_secondary": f"""
                QPushButton {{
                    background-color: transparent;
                    color: {c.get('primary', '#2196F3')};
                    border: 1px solid {c.get('primary', '#2196F3')};
                    border-radius: {self.get_border_radius('small')};
                    padding: 8px 16px;
                }}
                QPushButton:hover {{
                    background-color: {c.get('selected', '#E3F2FD')};
                }}
            """,
            "card": f"""
                QFrame {{
                    background-color: {c.get('card', '#FFFFFF')};
                    border: 1px solid {c.get('border', '#E0E0E0')};
                    border-radius: {self.get_border_radius('medium')};
                }}
            """,
            "input": f"""
                QLineEdit, QTextEdit {{
                    background-color: {c.get('surface', '#FFFFFF')};
                    color: {c.get('text_primary', '#212121')};
                    border: 1px solid {c.get('border', '#E0E0E0')};
                    border-radius: {self.get_border_radius('small')};
                    padding: 8px;
                }}
                QLineEdit:focus, QTextEdit:focus {{
                    border: 2px solid {c.get('primary', '#2196F3')};
                }}
            """,
            "tree": f"""
                QTreeWidget {{
                    background-color: {c.get('surface', '#FFFFFF')};
                    color: {c.get('text_primary', '#212121')};
                    border: 1px solid {c.get('border', '#E0E0E0')};
                    border-radius: {self.get_border_radius('small')};
                }}
                QTreeWidget::item:hover {{
                    background-color: {c.get('hover', '#F5F5F5')};
                }}
                QTreeWidget::item:selected {{
                    background-color: {c.get('selected', '#E3F2FD')};
                    color: {c.get('text_primary', '#212121')};
                }}
            """,
            "menu": f"""
                QMenuBar {{
                    background-color: {c.get('surface', '#FFFFFF')};
                    color: {c.get('text_primary', '#212121')};
                    border-bottom: 1px solid {c.get('border', '#E0E0E0')};
                }}
                QMenu {{
                    background-color: {c.get('surface', '#FFFFFF')};
                    color: {c.get('text_primary', '#212121')};
                    border: 1px solid {c.get('border', '#E0E0E0')};
                }}
                QMenu::item:selected {{
                    background-color: {c.get('selected', '#E3F2FD')};
                }}
            """,
            "scrollbar": f"""
                QScrollBar:vertical {{
                    background-color: {c.get('background', '#FAFAFA')};
                    width: 12px;
                    border-radius: 6px;
                }}
                QScrollBar::handle:vertical {{
                    background-color: {c.get('text_disabled', '#BDBDBD')};
                    border-radius: 6px;
                    min-height: 30px;
                }}
                QScrollBar::handle:vertical:hover {{
                    background-color: {c.get('text_secondary', '#757575')};
                }}
            """,
        }

        return stylesheets.get(widget_type, "")

    def apply_stylesheet(self, widget: QWidget, widget_type: str):
        """Apply stylesheet to a widget."""
        stylesheet = self.get_stylesheet(widget_type)
        if stylesheet:
            widget.setStyleSheet(stylesheet)

    def get_status_color(self, status: str) -> str:
        """Get color for a status type."""
        status_colors = {
            "success": self.get_color("success", "#4CAF50"),
            "warning": self.get_color("warning", "#FF9800"),
            "error": self.get_color("error", "#F44336"),
            "info": self.get_color("info", "#2196F3"),
            "pending": self.get_color("text_secondary", "#757575"),
        }
        return status_colors.get(status, status_colors["info"])


def get_style_manager() -> StyleManager:
    """Get the singleton style manager instance."""
    return StyleManager()
