"""
Logging configuration for the IBKR trading application.
"""

import logging
import os
import time
from typing import Optional


def setup_logging(level: str = 'INFO', log_file: Optional[str] = None) -> None:
    """
    Set up logging configuration for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path. If None, only console logging is used.
    """
    # Create log directory if it doesn't exist
    if not os.path.exists("log"):
        os.makedirs("log")
    
    # Set up logging format
    recfmt = '(%(threadName)s) %(asctime)s.%(msecs)03d %(levelname)s %(filename)s:%(lineno)d %(message)s'
    timefmt = '%y%m%d_%H:%M:%S'
    
    # Convert string level to logging level
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    if log_file:
        # Log to specified file
        logging.basicConfig(
            filename=log_file,
            filemode="w",
            level=numeric_level,
            format=recfmt,
            datefmt=timefmt
        )
    else:
        # Default: log to file with timestamp and console for errors
        log_filename = time.strftime("log/pyibapi.%y%m%d_%H%M%S.log")
        logging.basicConfig(
            filename=log_filename,
            filemode="w",
            level=numeric_level,
            format=recfmt,
            datefmt=timefmt
        )
    
    # Always add console handler for errors and warnings
    logger = logging.getLogger()
    console = logging.StreamHandler()
    
    # Set console level based on the main level
    if numeric_level <= logging.DEBUG:
        console.setLevel(logging.DEBUG)
    elif numeric_level <= logging.INFO:
        console.setLevel(logging.INFO)
    elif numeric_level <= logging.WARNING:
        console.setLevel(logging.WARNING)
    else:
        console.setLevel(logging.ERROR)
    
    # Use a simpler format for console output
    console_formatter = logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s',
        datefmt='%H:%M:%S'
    )
    console.setFormatter(console_formatter)
    
    logger.addHandler(console)
