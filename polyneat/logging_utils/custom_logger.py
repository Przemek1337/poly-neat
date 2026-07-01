from __future__ import annotations

import logging
import sys
import uuid
from pathlib import Path

from polyneat.config.logging_config import LoggingConfig
from polyneat.logging_utils.colored_level_formatter import ColoredLevelFormatter

_active_logging_config: LoggingConfig = LoggingConfig()


def set_logging_config(new_logging_config: LoggingConfig) -> None:
    """Replace the active LoggingConfig used by newly-created loggers.

    Existing loggers keep their already-attached handlers.
    """
    global _active_logging_config
    _active_logging_config = new_logging_config


def get_active_logging_config() -> LoggingConfig:
    return _active_logging_config


class CustomLogger(logging.Logger):
    """Logger with a colored stream handler and optional per-logger file handler.

    Instances are created lazily by ``logging.getLogger`` after
    ``logging.setLoggerClass(CustomLogger)`` runs at module import time.
    Handlers are attached exactly once per logger name, so repeated
    ``get_logger(__name__)`` calls remain idempotent.
    """

    def __init__(self, name: str, level: int = logging.NOTSET) -> None:
        active_config = _active_logging_config
        resolved_level = level if level != logging.NOTSET else active_config.log_level
        super().__init__(name=name, level=resolved_level)
        self.propagate = False
        self._attach_stream_handler(active_config)
        if active_config.file_log_directory is not None:
            self._attach_file_handler(active_config)

    def _attach_stream_handler(self, active_config: LoggingConfig) -> None:
        stream_handler = logging.StreamHandler(stream=sys.stdout)
        stream_handler.setFormatter(ColoredLevelFormatter(active_config.log_format))
        self.addHandler(stream_handler)

    def _attach_file_handler(self, active_config: LoggingConfig) -> None:
        log_directory_path = Path(active_config.file_log_directory)  # type: ignore[arg-type]
        log_directory_path.mkdir(parents=True, exist_ok=True)
        log_file_path = log_directory_path / f"polyneat_{uuid.uuid4()}.log"
        file_handler = logging.FileHandler(
            filename=str(log_file_path), mode="a", encoding="utf-8"
        )
        file_handler.setFormatter(ColoredLevelFormatter(active_config.log_format))
        self.addHandler(file_handler)


logging.setLoggerClass(CustomLogger)


def get_logger(logger_name: str) -> CustomLogger:
    """Return the ``CustomLogger`` bound to ``logger_name``.

    Because ``logging.setLoggerClass(CustomLogger)`` runs at module import
    time, the returned logger is always a ``CustomLogger`` instance.
    """
    return logging.getLogger(logger_name)  # type: ignore[return-value]
