"""Centralized logging configuration for jactus.

This module provides a flexible logging setup that can be configured via
environment variables or programmatically. It supports multiple handlers,
structured logging, and performance monitoring.
"""

import json
import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Any

# Default logging configuration
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
DEFAULT_LOG_FILE = "jactus.log"

# Environment variable names
ENV_LOG_LEVEL = "ACTUS_JAX_LOG_LEVEL"
ENV_LOG_FILE = "ACTUS_JAX_LOG_FILE"
ENV_LOG_FORMAT = "ACTUS_JAX_LOG_FORMAT"
ENV_STRUCTURED_LOGS = "ACTUS_JAX_STRUCTURED_LOGS"


class StructuredFormatter(logging.Formatter):
    """Formatter that outputs log records as JSON for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as JSON.

        Args:
            record: The log record to format

        Returns:
            JSON string representation of the log record
        """
        log_data: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add any extra fields from the record
        for key, value in record.__dict__.items():
            if key not in [
                "name",
                "msg",
                "args",
                "created",
                "filename",
                "funcName",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "message",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "thread",
                "threadName",
                "exc_info",
                "exc_text",
                "stack_info",
            ]:
                log_data[key] = value

        return json.dumps(log_data)


def get_logger(name: str, level: str | None = None) -> logging.Logger:
    """Get a logger instance with the specified name and level.

    Args:
        name: Name of the logger (typically __name__ of the calling module)
        level: Optional log level override. If not specified, uses environment
               variable or default level.

    Returns:
        Configured logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing contract", extra={"contract_id": "PAM-001"})
    """
    logger = logging.getLogger(name)

    # Don't add handlers if they're already configured
    if logger.handlers:
        return logger

    # Determine log level
    log_level_str = level or os.getenv(ENV_LOG_LEVEL) or DEFAULT_LOG_LEVEL
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)
    logger.setLevel(log_level)

    # Prevent propagation to root logger to avoid duplicate logs
    logger.propagate = False

    return logger


def configure_logging(
    level: str | None = None,
    log_file: str | None = None,
    console: bool = True,
    structured: bool = False,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
) -> None:
    """Configure logging for the entire jactus package.

    This function sets up logging handlers and formatters for the package.
    It should typically be called once at application startup.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
               Defaults to environment variable or INFO.
        log_file: Path to log file. Defaults to environment variable or
                 jactus.log in current directory.
        console: Whether to log to console (stdout). Default: True
        structured: Whether to use JSON structured logging. Default: False
                   Can be overridden by ACTUS_JAX_STRUCTURED_LOGS env var.
        max_bytes: Maximum size of log file before rotation (default: 10MB)
        backup_count: Number of backup log files to keep (default: 5)

    Example:
        >>> # Simple configuration
        >>> configure_logging(level="DEBUG")
        >>>
        >>> # Full configuration with file logging
        >>> configure_logging(
        ...     level="INFO",
        ...     log_file="/var/log/jactus.log",
        ...     structured=True
        ... )
    """
    # Get the root logger for jactus
    logger = logging.getLogger("jactus")

    # Remove any existing handlers
    logger.handlers.clear()

    # Determine log level
    log_level_str = level or os.getenv(ENV_LOG_LEVEL) or DEFAULT_LOG_LEVEL
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)
    logger.setLevel(log_level)

    # Determine if structured logging is enabled
    use_structured = structured or os.getenv(ENV_STRUCTURED_LOGS, "").lower() in (
        "true",
        "1",
        "yes",
    )

    # Create formatter
    if use_structured:
        formatter: logging.Formatter = StructuredFormatter(datefmt=DEFAULT_DATE_FORMAT)
    else:
        log_format = os.getenv(ENV_LOG_FORMAT, DEFAULT_LOG_FORMAT)
        formatter = logging.Formatter(log_format, datefmt=DEFAULT_DATE_FORMAT)

    # Add console handler if requested
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Add file handler if log file is specified
    log_file_path = log_file or os.getenv(ENV_LOG_FILE)
    if log_file_path:
        # Create log directory if it doesn't exist
        log_path = Path(log_file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Use rotating file handler to prevent unbounded growth
        file_handler = logging.handlers.RotatingFileHandler(
            log_file_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False


def get_performance_logger(name: str) -> logging.Logger:
    """Get a logger configured for performance monitoring.

    Performance loggers are typically used to log timing information,
    memory usage, and other performance metrics. They default to DEBUG
    level and can be controlled separately from regular loggers.

    Args:
        name: Name of the performance logger

    Returns:
        Logger configured for performance monitoring

    Example:
        >>> perf_logger = get_performance_logger("jactus.engine")
        >>> import time
        >>> start = time.time()
        >>> # ... do work ...
        >>> elapsed = time.time() - start
        >>> perf_logger.debug("Portfolio simulation completed",
        ...                   extra={"duration_ms": elapsed * 1000,
        ...                          "num_contracts": 1000})
    """
    logger = logging.getLogger(f"jactus.performance.{name}")

    # Performance loggers typically log at DEBUG level
    perf_level = os.getenv("ACTUS_JAX_PERF_LOG_LEVEL", "DEBUG")
    logger.setLevel(getattr(logging, perf_level.upper(), logging.DEBUG))

    return logger


def disable_logging() -> None:
    """Disable all jactus logging.

    This is useful for tests or applications that want to suppress
    all logging output from the package.
    """
    logger = logging.getLogger("jactus")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    logger.propagate = False


# Configure basic logging on module import
# This provides sensible defaults without explicit configuration
if not logging.getLogger("jactus").handlers:
    configure_logging()
