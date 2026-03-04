"""Qt log handler for bridging Python logging to Qt signals."""

import logging
from typing import Callable, Optional

from PyQt6.QtCore import QObject, pyqtSignal


class QtLogHandler(logging.Handler, QObject):
    """Custom logging handler that emits Qt signals.
    
    This handler bridges Python's logging module with Qt's signal/slot mechanism,
    allowing log messages to be displayed in GUI components.
    """
    
    log_signal = pyqtSignal(str, str)
    
    def __init__(self, parent: Optional[QObject] = None):
        """Initialize the handler.
        
        Args:
            parent: Optional parent QObject
        """
        logging.Handler.__init__(self)
        QObject.__init__(self, parent)
        self._callback: Optional[Callable[[str, str], None]] = None
        self._closed = False
    
    def set_callback(self, callback: Callable[[str, str], None]):
        """Set a callback function for log messages.
        
        Args:
            callback: Function that takes (message, level) arguments
        """
        self._callback = callback
    
    def emit(self, record: logging.LogRecord):
        """Emit a log record as a Qt signal.
        
        Args:
            record: The log record to emit
        """
        if self._closed:
            return
        try:
            msg = self.format(record)
            level = record.levelname
            
            if self._callback:
                self._callback(msg, level)
            else:
                self.log_signal.emit(msg, level)
        except Exception:
            self.handleError(record)
    
    def close(self):
        """Close the handler and prevent further emissions."""
        self._closed = True
        super().close()


class LogEmitter(QObject):
    """Log emitter that can be used in worker threads.
    
    This class provides a thread-safe way to emit log messages from
    worker threads to the main GUI thread.
    """
    
    log_message = pyqtSignal(str, str)
    
    def __init__(self, parent: Optional[QObject] = None):
        """Initialize the emitter.
        
        Args:
            parent: Optional parent QObject
        """
        super().__init__(parent)
        self._handler: Optional[QtLogHandler] = None
    
    def install_handler(self, logger_name: str = 'pyutagent') -> QtLogHandler:
        """Install the log handler on the specified logger.
        
        Args:
            logger_name: Name of the logger to install handler on
            
        Returns:
            The installed handler
        """
        self._handler = QtLogHandler()
        self._handler.log_signal.connect(self._on_log)
        
        logger = logging.getLogger(logger_name)
        logger.addHandler(self._handler)
        
        return self._handler
    
    def uninstall_handler(self, logger_name: str = 'pyutagent'):
        """Uninstall the log handler from the specified logger.
        
        Args:
            logger_name: Name of the logger to remove handler from
        """
        if self._handler:
            logger = logging.getLogger(logger_name)
            logger.removeHandler(self._handler)
            self._handler.close()
            self._handler = None
    
    def _on_log(self, message: str, level: str):
        """Handle log message from handler.
        
        Args:
            message: The log message
            level: The log level
        """
        self.log_message.emit(message, level)
