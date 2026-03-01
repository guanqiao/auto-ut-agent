"""Main entry point for PyUT Agent.

This module provides the application entry point with proper async/Qt integration
using qasync for seamless asyncio event loop integration with Qt.
"""

import asyncio
import logging
import sys

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

try:
    import qasync
    QASYNC_AVAILABLE = True
except ImportError:
    QASYNC_AVAILABLE = False
    logging.warning("qasync not available, falling back to basic event loop handling")

from .ui.main_window import MainWindow
from .core.config import (
    get_settings,
    load_app_config,
    load_llm_config,
    load_aider_config,
    save_app_config,
)
from .core.container import configure_container

logger = logging.getLogger(__name__)


def setup_logging(settings):
    """Setup logging configuration."""
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger.info(f"Logging configured with level: {settings.log_level}")


def main():
    """Main entry point."""
    settings = load_app_config()
    setup_logging(settings)

    llm_config = load_llm_config()
    aider_config = load_aider_config()

    configure_container(
        settings=settings,
        llm_config_collection=llm_config,
        aider_config=aider_config
    )

    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("PyUT Agent")
    app.setApplicationVersion("0.1.0")
    app.setStyle("Fusion")

    if QASYNC_AVAILABLE:
        loop = qasync.QEventLoop(app)
        asyncio.set_event_loop(loop)
        logger.info("Using qasync for Qt/asyncio integration")

        window = MainWindow()
        window.show()

        # Auto-load last project after window is shown
        window.load_last_project()

        with loop:
            loop.run_forever()
    else:
        window = MainWindow()
        window.show()

        # Auto-load last project after window is shown
        window.load_last_project()

        sys.exit(app.exec())


if __name__ == "__main__":
    main()
