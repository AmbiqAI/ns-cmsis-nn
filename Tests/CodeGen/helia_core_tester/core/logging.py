"""
Logging configuration for CMSIS-NN Tools.
"""

import logging
import sys
from typing import Optional
from pathlib import Path


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[94m',      # Blue
        'WARNING': '\033[93m',   # Yellow
        'ERROR': '\033[91m',     # Red
        'CRITICAL': '\033[95m',  # Magenta
        'RESET': '\033[0m',      # Reset
        'BOLD': '\033[1m',       # Bold
    }
    
    def format(self, record):
        """Format log record with colors."""
        # Add color to level name
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"
        
        # Add bold to message for important levels
        if record.levelname in ['ERROR', 'CRITICAL']:
            record.msg = f"{self.COLORS['BOLD']}{record.msg}{self.COLORS['RESET']}"
        
        return super().format(record)


def setup_logger(
    name: str = "helia_core_tester",
    level: Optional[int] = None,
    log_file: Optional[Path] = None,
    verbosity: int = 0
) -> logging.Logger:
    """
    Set up logger for Helia Core Tester.
    
    Args:
        name: Logger name
        level: Logging level (overrides verbosity if provided)
        log_file: Optional log file path
        verbosity: Verbosity level (0=minimal, 1=progress, 2=commands, 3=debug)
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Map verbosity to logging level
    if level is None:
        if verbosity == 0:
            level = logging.WARNING  # Only errors/warnings
        elif verbosity == 1:
            level = logging.INFO  # Progress messages
        elif verbosity == 2:
            level = logging.INFO  # Commands and progress
        else:  # verbosity == 3
            level = logging.DEBUG  # All messages
    
    logger.setLevel(level)
    
    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logger.level)
    
    # Format for console
    console_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    console_formatter = ColoredFormatter(console_format)
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Format for file (no colors)
        file_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        file_formatter = logging.Formatter(file_format)
        file_handler.setFormatter(file_formatter)
        
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "helia_core_tester") -> logging.Logger:
    """Get logger instance."""
    return logging.getLogger(name)
