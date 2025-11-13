import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

from loguru import logger

from .config import config

Path("logs").mkdir(exist_ok=True)


class InterceptHandler(logging.Handler):
    def emit(self, record):
        if record.name == "uvicorn" and record.levelno < logging.INFO:
            return

        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def _truncate_value(value: Any, max_len: int = 100) -> str:
    """Truncate and format a value for logging."""
    if isinstance(value, str):
        result = value
    elif isinstance(value, (dict, list)):
        try:
            result = json.dumps(value, separators=(",", ":"), default=str)
        except Exception:
            result = str(value)
    else:
        result = str(value)

    return result[: max_len - 3] + "..." if len(result) > max_len else result


def _strip_color_tags(text: str) -> str:
    """Remove loguru color tags from formatted text."""
    return re.sub(r"</?(?:green|level|cyan|blue|yellow)>", "", text)


def format_record(record: dict[str, Any]) -> str:
    """Format log records with extra fields displayed inline."""

    base = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )

    extra = record.get("extra", {})
    if extra:
        display_extra = {k: v for k, v in extra.items() if not k.startswith("_")}

        if display_extra:
            extra_parts = []
            for key, value in display_extra.items():
                safe_key = str(key).replace("{", "{{").replace("}", "}}")
                truncated = _truncate_value(value)
                safe_value = truncated.replace("{", "{{").replace("}", "}}")

                extra_parts.append(
                    f"<blue>{safe_key}</blue>=<yellow>{safe_value}</yellow>"
                )

            base += " | " + " ".join(extra_parts)

    return base + "\n{exception}"


logger.remove()

logger.add(
    sys.stderr,
    level=config.logging_level or "INFO",
    format=format_record,
    colorize=True,
    backtrace=True,
    diagnose=True,
    enqueue=True,
)

logger.add(
    "logs/app.log",
    rotation="00:00",
    retention="7 days",
    level=config.logging_level or "INFO",
    format=lambda r: _strip_color_tags(format_record(r)),
    compression="zip",
    backtrace=True,
    diagnose=True,
    enqueue=True,
)

logger.add(
    "logs/app-structured.log",
    rotation="100 MB",
    retention="30 days",
    level="DEBUG",
    enqueue=True,
    serialize=True,
)

logger.add(
    "logs/errors.log",
    rotation="50 MB",
    retention="30 days",
    level="ERROR",
    format=lambda r: _strip_color_tags(format_record(r)),
    backtrace=True,
    diagnose=True,
    enqueue=True,
)

logging.basicConfig(handlers=[InterceptHandler()], level=logging.INFO, force=True)

for logger_name in ["uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"]:
    logging_logger = logging.getLogger(logger_name)
    logging_logger.handlers = [InterceptHandler()]
    logging_logger.propagate = False

logging.getLogger("uvicorn.access").setLevel(logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)


class LoggerWrapper:
    """
    A wrapper class for loguru logger that adds support for 'extra' parameter
    without modifying the original logger instance.
    """

    def __init__(self, wrapped_logger):
        self._logger = wrapped_logger

    def info(self, message, *args, extra=None, **kwargs):
        if extra:
            self._logger.bind(**extra).opt(depth=1).info(message, *args, **kwargs)
        else:
            self._logger.opt(depth=1).info(message, *args, **kwargs)

    def debug(self, message, *args, extra=None, **kwargs):
        if extra:
            self._logger.bind(**extra).opt(depth=1).debug(message, *args, **kwargs)
        else:
            self._logger.opt(depth=1).debug(message, *args, **kwargs)

    def warning(self, message, *args, extra=None, **kwargs):
        if extra:
            self._logger.bind(**extra).opt(depth=1).warning(message, *args, **kwargs)
        else:
            self._logger.opt(depth=1).warning(message, *args, **kwargs)

    def error(self, message, *args, extra=None, **kwargs):
        if extra:
            self._logger.bind(**extra).opt(depth=1).error(message, *args, **kwargs)
        else:
            self._logger.opt(depth=1).error(message, *args, **kwargs)

    def critical(self, message, *args, extra=None, **kwargs):
        if extra:
            self._logger.bind(**extra).opt(depth=1).critical(message, *args, **kwargs)
        else:
            self._logger.opt(depth=1).critical(message, *args, **kwargs)

    def bind(self, **context):
        """Create a new logger instance with bound context."""
        return LoggerWrapper(self._logger.bind(**context))

    def opt(self, **options):
        """Pass through to the wrapped logger's opt method."""
        return self._logger.opt(**options)

    def __getattr__(self, name):
        """Delegate any other attributes/methods to the wrapped logger."""
        return getattr(self._logger, name)


def get_logger(**context):
    """
    Get a logger instance with bound context.

    Example:
        logger = get_logger(user_id="123", request_id="abc")
        logger.info("Processing request")  # Will include user_id and request_id

    Args:
        **context: Key-value pairs to bind to the logger

    Returns:
        LoggerWrapper instance with bound context
    """
    return LoggerWrapper(logger.bind(**context))


# Wrap the logger to support 'extra' parameter
# Note: This reassigns 'logger' from the loguru import
logger = LoggerWrapper(logger)

__all__ = ["logger", "get_logger"]
