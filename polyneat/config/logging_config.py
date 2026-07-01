from __future__ import annotations

import logging
from dataclasses import dataclass


@dataclass
class LoggingConfig:
    """Configuration for the PolyNEAT logging subsystem.

    All modules obtain their logger via
    ``polyneat.logging_utils.custom_logger.get_logger``. The active
    ``LoggingConfig`` is consulted once per logger name to attach a colored
    stream handler and (optionally) a per-logger file handler.
    """

    log_level: int = logging.INFO
    log_format: str = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    file_log_directory: str | None = None
