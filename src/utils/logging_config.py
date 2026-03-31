"""Logging configuration utility."""

import sys
from loguru import logger


def setup_logger(name: str = "default"):
    """Set up and configure a logger instance.

    Args:
        name: Logger name

    Returns:
        Configured logger instance
    """
    # Remove default logger
    logger.remove()

    # Add stdout logger with custom format
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )

    # Add file logger
    logger.add(
        "logs/{time:YYYY-MM-DD}.log",
        rotation="00:00",
        retention="30 days",
        compression="zip",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG"
    )

    return logger.bind(name=name)