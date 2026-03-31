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


def test_no_duplicate_rows():
    """No duplicate rows in the processed dataset."""
    parquet_path = (
        Path(__file__).parent.parent / "data" / "processed" / "flights_cleaned.parquet"
    )
    df = pl.read_parquet(parquet_path)
    # Polars no tiene unique_rows(); usamos unique() y comparamos el número de filas
    # Si existen duplicados, el conteo será menor tras aplicar unique()
    assert df.shape[0] == df.unique().shape[0], "Dataset contains duplicate rows"


def test_expected_columns_exist():
    """Ensure all expected columns are present."""
    # Columnas que el pipeline de limpieza realmente genera
    expected_cols = {
        "MONTH",
        "DAY",
        "DAY_OF_WEEK",
        "AIRLINE",
        "ORIGIN_AIRPORT",
        "DESTINATION_AIRPORT",
        "SCHEDULED_DEPARTURE",
        "DISTANCE",
        "DEPARTURE_DELAY",
        "TARGET_IS_DELAYED",
    }
    parquet_path = (
        Path(__file__).parent.parent / "data" / "processed" / "flights_cleaned.parquet"
    )
    df = pl.read_parquet(parquet_path)
    missing = expected_cols - set(df.columns)
    assert not missing, f"Missing columns: {missing}"


def test_numeric_columns_are_numeric():
    """Check that numeric columns have appropriate dtypes."""
    parquet_path = (
        Path(__file__).parent.parent / "data" / "processed" / "flights_cleaned.parquet"
    )
    df = pl.read_parquet(parquet_path)
    # Ampliamos los tipos admitidos para cubrir Int16, Int32, Int64, UInt*, Float*
    numeric_cols = ["DEPARTURE_DELAY"]
    for col in numeric_cols:
        assert df[col].dtype in [
            pl.Int8,
            pl.Int16,
            pl.Int32,
            pl.Int64,
            pl.UInt8,
            pl.UInt16,
            pl.UInt32,
            pl.UInt64,
            pl.Float32,
            pl.Float64,
        ], f"{col} is not numeric (dtype={df[col].dtype})"