from __future__ import annotations

import logging

import colorama
from colorama import Fore

colorama.init(autoreset=True)


class ColoredLevelFormatter(logging.Formatter):
    """Log formatter that colors the message body according to its level."""

    LEVEL_NAME_TO_COLOR_CODE: dict[str, str] = {
        "DEBUG": Fore.LIGHTBLUE_EX,
        "INFO": Fore.LIGHTGREEN_EX,
        "WARNING": Fore.LIGHTYELLOW_EX,
        "ERROR": Fore.LIGHTRED_EX,
        "CRITICAL": Fore.RED,
    }

    def format(self, record: logging.LogRecord) -> str:
        formatted_full_line = super().format(record)
        color_code = self.LEVEL_NAME_TO_COLOR_CODE.get(record.levelname, Fore.RESET)
        colored_message_body = f"{color_code}{record.getMessage()}{Fore.RESET}"
        return formatted_full_line.replace(record.getMessage(), colored_message_body)
