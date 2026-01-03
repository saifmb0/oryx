"""ORYX - SEO keyword discovery and clustering CLI."""

import logging
import sys
from pathlib import Path

# =============================================================================
# Structured Logging Configuration
# =============================================================================

def setup_logging(
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    log_file: str = "oryx.log",
) -> logging.Logger:
    """
    Configure structured logging for ORYX.
    
    Sets up logging to both console (INFO level) and file (DEBUG level).
    
    Args:
        console_level: Logging level for console output
        file_level: Logging level for file output
        log_file: Path to the log file
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("oryx")
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.DEBUG)  # Capture all levels
    
    # Console handler (INFO and above)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_format = logging.Formatter(
        "%(levelname)s: %(message)s"
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler (DEBUG and above)
    try:
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(file_level)
        file_format = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    except (OSError, PermissionError) as e:
        # If we can't write to log file, just use console
        logger.warning(f"Could not create log file '{log_file}': {e}")
    
    return logger


def get_logger(name: str = "oryx") -> logging.Logger:
    """
    Get a logger instance for ORYX modules.
    
    Args:
        name: Logger name (typically module name)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


# Initialize logging on module import
_logger = setup_logging()


# Import typed config classes for convenience
from oryx.config import (
    KeywordLabConfig,
    IntentRulesConfig,
    NlpConfig,
    ScrapeConfig,
    ClusterConfig,
    LlmConfig,
    OutputConfig,
    load_config,
    ConfigValidationError,
)

__all__ = [
    # Modules
    "io",
    "scrape",
    "nlp",
    "cluster",
    "metrics",
    "schema",
    "pipeline",
    "config",
    # Logging
    "setup_logging",
    "get_logger",
    # Config types
    "KeywordLabConfig",
    "IntentRulesConfig",
    "NlpConfig",
    "ScrapeConfig",
    "ClusterConfig",
    "LlmConfig",
    "OutputConfig",
    "load_config",
    "ConfigValidationError",
]

__version__ = "3.4.0"
