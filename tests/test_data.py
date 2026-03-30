"""
Pruebas de calidad de datos para el dataset de vuelos.
"""

import polars as pl
from pathlib import Path
import pytest


def test_target_column_exists_and_no_nulls():
    """Verifica que la columna TARGET_IS_DELAYED existe y no tiene valores nulos."""
    # Ruta al archivo parquet procesado
    parquet_path = (
        Path(__file__).parent.parent / "data" / "processed" / "flights_cleaned.parquet"
    )

    # Verificar que el archivo existe
    assert parquet_path.exists(), f"Archivo no encontrado: {parquet_path}"

    # Cargar datos con Polars
    df = pl.read_parquet(parquet_path)

    # Verificar que la columna TARGET_IS_DELAYED existe
    assert "TARGET_IS_DELAYED" in df.columns, (
        "Columna TARGET_IS_DELAYED no encontrada en el dataset"
    )

    # Verificar que no hay valores nulos en TARGET_IS_DELAYED
    null_count = df["TARGET_IS_DELAYED"].null_count()
    assert null_count == 0, (
        f"La columna TARGET_IS_DELAYED tiene {null_count} valores nulos"
    )

    # Verificar que solo contiene valores 0 y 1
    unique_values = df["TARGET_IS_DELAYED"].unique().to_list()
    assert set(unique_values).issubset({0, 1}), (
        f"TARGET_IS_DELAYED contiene valores inesperados: {unique_values}"
    )

    # Verificar que hay al menos una fila
    assert df.shape[0] > 0, "El dataset está vacío"

    # Opcional: Verificar distribución de clases (debe ser aproximadamente 82%/18%)
    class_distribution = df["TARGET_IS_DELAYED"].value_counts()
    total = df.shape[0]
    for row in class_distribution.iter_rows(named=True):
        proportion = row["count"] / total
        # Verificar que ninguna clase tiene menos del 10% (para detectar desbalance extremo)
        assert proportion > 0.1, (
            f"Clase {row['TARGET_IS_DELAYED']} tiene solo {proportion:.2%} de los datos"
        )

    print(
        f"✅ Test de datos pasado: {df.shape[0]:,} filas, columna TARGET_IS_DELAYED válida"
    )
