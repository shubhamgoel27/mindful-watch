"""
Centralized logging configuration for MindfulWatch.

Provides consistent logging across local and Streamlit Cloud environments.
"""

import logging
import sys

# Create logger
logger = logging.getLogger("mindfulwatch")

def setup_logging(level=logging.INFO):
    """
    Configures the MindfulWatch logger with console output.
    
    On Streamlit Cloud, stdout/stderr are captured, so we use StreamHandler.
    The format includes timestamp, level, and source for easy debugging.
    """
    if logger.handlers:
        # Already configured, skip
        return logger
    
    logger.setLevel(level)
    
    # Console handler (captured by Streamlit Cloud)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # Format: timestamp - level - module - message
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(module)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Prevent duplicate logs from propagating to root logger
    logger.propagate = False
    
    logger.info("MindfulWatch logging initialized")
    return logger

# Auto-initialize on import
setup_logging()
