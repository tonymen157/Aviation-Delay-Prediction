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

# Importar kagglehub para descarga
try:
    import kagglehub
except ImportError:
    print("ERROR: kagglehub no esta instalado. Ejecuta: pip install kagglehub")
    raise SystemExit(1)


def download_and_organize_data():
    """Descarga dataset de Kaggle y lo organiza en data/raw/."""
    print("[*] Iniciando descarga del dataset 'usdot/flight-delays'...")

    # Descargar dataset (kagglehub guarda en cache)
    try:
        cache_path = kagglehub.dataset_download("usdot/flight-delays")
        print(f"[OK] Dataset descargado en cache: {cache_path}")
    except Exception as e:
        print(f"[ERROR] Error descargando dataset: {e}")
        raise SystemExit(1)

    # Definir directorio destino
    project_root = Path(__file__).parent.parent.parent
    raw_data_dir = project_root / "data" / "raw"
    raw_data_dir.mkdir(parents=True, exist_ok=True)

    # Copiar archivos del cache a data/raw/
    print(f"[*] Copiando archivos a: {raw_data_dir}")

    files_copied = 0
    for item in Path(cache_path).iterdir():
        if item.is_file():
            dest_file = raw_data_dir / item.name
            shutil.copy2(item, dest_file)
            print(f"   [+] {item.name} -> {dest_file}")
            files_copied += 1

    if files_copied == 0:
        print("[!] No se encontraron archivos en el cache.")
    else:
        print(
            f"\n[OK] Descarga completada! {files_copied} archivos copiados a data/raw/"
        )
        print("[INFO] Archivos en data/raw/:")
        for f in sorted(raw_data_dir.iterdir()):
            if f.is_file() and f.name != ".gitkeep":
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"   - {f.name} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    download_and_organize_data()
