"""PyUT Agent Application V2 - Entry point with new UI.

This module provides the main entry point for running PyUT Agent with the new
refactored UI architecture.

Usage:
    python -m pyutagent.app_v2
    
Or:
    from pyutagent.app_v2 import main
    main()
"""

import sys
import logging
import argparse
from pathlib import Path

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QFontDatabase

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_application() -> QApplication:
    """Setup the Qt application.
    
    Returns:
        Configured QApplication instance
    """
    # Enable high DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    
    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("PyUT Agent")
    app.setApplicationVersion("2.0.0")
    app.setOrganizationName("PyUT")
    
    # Set application font
    font = QFont("Segoe UI", 10)
    if not QFontDatabase.hasFamily("Segoe UI"):
        # Fallback to system default
        font = QFont()
        font.setPointSize(10)
    app.setFont(font)
    
    return app


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(
        description="PyUT Agent - AI Coding Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # Start with GUI
  %(prog)s --project /path/to/proj   # Open project on startup
  %(prog)s --no-ui                   # Run in CLI mode
        """
    )
    
    parser.add_argument(
        "--project", "-p",
        type=str,
        help="Path to project directory to open on startup"
    )
    
    parser.add_argument(
        "--no-ui",
        action="store_true",
        help="Run in CLI mode (no GUI)"
    )
    
    parser.add_argument(
        "--legacy-ui",
        action="store_true",
        help="Use legacy UI instead of new UI"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # CLI mode
    if args.no_ui:
        logger.info("Running in CLI mode")
        from .cli import main as cli_main
        sys.argv = [sys.argv[0]]  # Remove our args
        cli_main()
        return
    
    # GUI mode
    logger.info("Starting PyUT Agent GUI v2")
    
    app = setup_application()
    
    # Import and create main window
    if args.legacy_ui:
        logger.info("Using legacy UI")
        from .ui.main_window import MainWindow
        window = MainWindow()
    else:
        logger.info("Using new UI (v2)")
        from .ui.main_window_v2 import MainWindowV2
        window = MainWindowV2()
    
    # Open project if specified
    if args.project:
        project_path = Path(args.project).resolve()
        if project_path.exists() and project_path.is_dir():
            logger.info(f"Opening project: {project_path}")
            window._open_project(str(project_path))
        else:
            logger.warning(f"Project path not found: {project_path}")
    
    window.show()
    
    # Run application
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
