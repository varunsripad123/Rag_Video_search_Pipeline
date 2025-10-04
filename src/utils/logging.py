"""Structured logging configuration."""
from __future__ import annotations

import logging
import logging.config
from pathlib import Path
from typing import Optional

try:  # pragma: no cover - optional dependency fallback
    from pythonjsonlogger import jsonlogger
except ImportError:  # pragma: no cover
    class _FallbackJsonFormatter(logging.Formatter):
        def add_fields(self, log_record, record, message_dict):
            log_record.update(message_dict)

    class jsonlogger:  # type: ignore
        JsonFormatter = _FallbackJsonFormatter

from src.config import AppConfig


class _JsonFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter that ensures consistent keys."""

    def add_fields(self, log_record, record, message_dict):  # type: ignore[override]
        super().add_fields(log_record, record, message_dict)
        log_record.setdefault("level", record.levelname)
        log_record.setdefault("logger", record.name)
        log_record.setdefault("message", record.getMessage())


def configure_logging(config: AppConfig) -> None:
    """Configure logging for the entire application."""

    handlers: dict[str, dict[str, object]] = {
        "default": {
            "class": "logging.StreamHandler",
            "formatter": "json" if config.logging.json_output else "standard",
        }
    }

    if config.logging.file:
        Path(config.logging.file).parent.mkdir(parents=True, exist_ok=True)
        handlers["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": str(config.logging.file),
            "maxBytes": 50_000_000,
            "backupCount": 5,
            "formatter": "json" if config.logging.json_output else "standard",
        }

    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {
                "()": _JsonFormatter,
                "fmt": "%(asctime)s %(levelname)s %(name)s %(message)s",
            },
            "standard": {
                "format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            },
        },
        "handlers": handlers,
        "loggers": {
            "": {
                "level": config.logging.level,
                "handlers": list(handlers.keys()),
            }
        },
    }

    logging.config.dictConfig(logging_config)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Retrieve a configured logger."""

    return logging.getLogger(name)
