#!/usr/bin/env python3
"""
Script para descargar automaticamente el dataset de Flight Delays desde Kaggle.
Usa kagglehub para la descarga y copia los archivos a data/raw/.
"""

import os
import shutil
import sys
from pathlib import Path

# Forzar UTF-8 en Windows
sys.stdout.reconfigure(encoding="utf-8")

# Import utilities
from utils.pipeline_common import add_project_root_to_path
add_project_root_to_path()

from utils.logging_config import setup_logger

# Importar kagglehub para descarga
try:
    import kagglehub
except ImportError:
    print("ERROR: kagglehub no esta instalado. Ejecuta: pip install kagglehub")
    raise SystemExit(1)

# Configurar logger
logger = setup_logger("00_download_data")


def download_and_organize_data(raw_data_dir=None):
    """Descarga dataset de Kaggle y lo organiza en data/raw/."""
    logger.info("Iniciando descarga del dataset 'usdot/flight-delays'...")

    # Descargar dataset (kagglehub guarda en cache)
    try:
        cache_path = kagglehub.dataset_download("usdot/flight-delays")
        logger.info(f"Dataset descargado en cache: {cache_path}")
    except Exception as e:
        logger.error(f"Error descargando dataset: {e}")
        raise SystemExit(1)

    # Definir directorio destino
    if raw_data_dir is None:
        project_root = Path(__file__).parent.parent.parent
        raw_data_dir = project_root / "data" / "raw"
    raw_data_dir.mkdir(parents=True, exist_ok=True)

    # Copiar archivos del cache a data/raw/
    logger.info(f"Copiando archivos a: {raw_data_dir}")

    files_copied = 0
    for item in Path(cache_path).iterdir():
        if item.is_file():
            dest_file = raw_data_dir / item.name
            shutil.copy2(item, dest_file)
            logger.info(f"{item.name} -> {dest_file}")
            files_copied += 1

    if files_copied == 0:
        logger.warning("No se encontraron archivos en el cache.")
    else:
        logger.info(f"Descarga completada! {files_copied} archivos copiados a data/raw/")
        logger.info("Archivos en data/raw/:")
        for f in sorted(raw_data_dir.iterdir()):
            if f.is_file() and f.name != ".gitkeep":
                size_mb = f.stat().st_size / (1024 * 1024)
                logger.info(f"  - {f.name} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    download_and_organize_data()
