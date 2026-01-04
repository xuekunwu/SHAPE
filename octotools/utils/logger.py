"""
Unified logging system for OctoTools.
Replaces scattered print statements with proper logging.
"""

import logging
import sys
from typing import Optional


class OctoToolsLogger:
    """Centralized logger for OctoTools framework."""
    
    _instance: Optional['OctoToolsLogger'] = None
    _logger: Optional[logging.Logger] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize_logger()
        return cls._instance
    
    def _initialize_logger(self):
        """Initialize the logger with appropriate configuration."""
        self._logger = logging.getLogger('octotools')
        self._logger.setLevel(logging.INFO)
        
        # Avoid duplicate handlers
        if not self._logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(logging.INFO)
            
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)
    
    def set_level(self, level: int):
        """Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)."""
        if self._logger:
            self._logger.setLevel(level)
            for handler in self._logger.handlers:
                handler.setLevel(level)
    
    def debug(self, message: str, *args, **kwargs):
        """Log debug message."""
        if self._logger:
            self._logger.debug(message, *args, **kwargs)
    
    def info(self, message: str, *args, **kwargs):
        """Log info message."""
        if self._logger:
            self._logger.info(message, *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs):
        """Log warning message."""
        if self._logger:
            self._logger.warning(message, *args, **kwargs)
    
    def error(self, message: str, *args, **kwargs):
        """Log error message."""
        if self._logger:
            self._logger.error(message, *args, **kwargs)
    
    def critical(self, message: str, *args, **kwargs):
        """Log critical message."""
        if self._logger:
            self._logger.critical(message, *args, **kwargs)


# Global logger instance
logger = OctoToolsLogger()
