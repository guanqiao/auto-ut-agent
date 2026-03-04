"""GUI tests package.

This package contains tests for the PyQt6 GUI components.
Requires pytest-qt to be installed.
"""

import pytest

# Skip all GUI tests if pytest-qt is not available
try:
    import pytestqt
    PYTEST_QT_AVAILABLE = True
except ImportError:
    PYTEST_QT_AVAILABLE = False
    pytest.skip("pytest-qt not installed", allow_module_level=True)
