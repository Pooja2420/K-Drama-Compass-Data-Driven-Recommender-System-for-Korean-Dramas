import sys
from pathlib import Path

from loguru import logger


def get_logger(name: str = "kdrama") -> logger:
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logger.remove()
    logger.add(
        sys.stdout,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level}</level> | "
            "<cyan>{name}</cyan> - {message}"
        ),
        level="INFO",
    )
    logger.add(
        log_dir / f"{name}.log",
        rotation="10 MB",
        retention="7 days",
        level="DEBUG",
    )
    return logger
