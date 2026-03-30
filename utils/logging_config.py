"""
Configuración centralizada de logging para todo el proyecto.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_dir: str | Path | None = None,
    log_to_file: bool = True,
    log_to_console: bool = True,
) -> logging.Logger:
    """
    Configura y retorna un logger con handlers de consola y archivo.

    Args:
        name: Nombre del logger (usualmente __name__ del módulo).
        level: Nivel de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_dir: Directorio para guardar logs. Si None, usa 'logs/' en la raíz.
        log_to_file: Si True, guarda logs en archivo.
        log_to_console: Si True, muestra logs en consola.

    Returns:
        Logger configurado.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Evitar duplicar handlers si ya está configurado
    if logger.handlers:
        return logger

    # Formato de logs
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Handler de consola
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Handler de archivo
    if log_to_file:
        if log_dir is None:
            log_dir = Path(__file__).parent.parent / "logs"
        elif not isinstance(log_dir, Path):
            log_dir = Path(log_dir)

        log_dir.mkdir(parents=True, exist_ok=True)

        # Nombre de archivo con fecha
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{name}_{timestamp}.log"

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Guardar también un log general combinado
        general_log = log_dir / "general.log"
        general_handler = logging.FileHandler(general_log, encoding="utf-8")
        general_handler.setLevel(logging.INFO)
        general_handler.setFormatter(formatter)
        logger.addHandler(general_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Obtiene un logger existente por nombre.

    Args:
        name: Nombre del logger.

    Returns:
        Logger existente o uno nuevo con configuración básica.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name, log_to_file=False)
    return logger
