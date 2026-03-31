#!/usr/bin/env python3
"""
Script de limpieza y transformación usando Polars para procesar Big Data.
Pipeline: Lectura → Filtrado → Feature Engineering → Exportación a Parquet.
Incluye generación automática de reporte de calidad de datos (Data Quality Report).
"""

import sys
from pathlib import Path

import polars as pl

# ----------------------------------------------------------------------
# Add project root to sys.path so that `utils` can be imported.
# Mirrors the logic in utils/pipeline_common.add_project_root_to_path().
# ----------------------------------------------------------------------
project_root = Path(__file__).resolve().parents[2]  # repo root
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Shared utilities
from utils.logging_config import setup_logger
logger = setup_logger("01_clean_and_transform")


def load_raw_data(raw_dir: Path) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Carga los tres archivos CSV desde data/raw/."""
    logger.info("Cargando datos crudos...")

    # Cargar flights.csv con optimizaciones para Big Data
    flights_path = raw_dir / "flights.csv"
    if not flights_path.exists():
        raise FileNotFoundError(f"No se encontró {flights_path}")

    flights = pl.read_csv(
        flights_path,
        infer_schema_length=10000,  # Inferir esquema con más muestras para precisión
        ignore_errors=True,  # Ignorar filas con errores de parseo
        null_values=["NA", ""],  # Tratar "NA" y "" como nulos
    )
    logger.info(f"flights.csv: {flights.shape[0]:,} filas × {flights.shape[1]} columnas")

    # Cargar airlines.csv y airports.csv (pequeños)
    airlines = pl.read_csv(raw_dir / "airlines.csv")
    airports = pl.read_csv(raw_dir / "airports.csv")
    logger.info(f"airlines.csv: {airlines.shape[0]} filas")
    logger.info(f"airports.csv: {airports.shape[0]} filas")

    return flights, airlines, airports


def clean_flights(flights: pl.DataFrame) -> pl.DataFrame:
    """Aplica filtros de negocio y crea variable objetivo."""
    logger.info("Aplicando filtros de negocio...")

    # 1. Filtrar solo vuelos NO cancelados
    initial_rows = flights.shape[0]
    flights_clean = flights.filter(pl.col("CANCELLED") == 0)
    logger.info(f"Filtrado cancelados: {initial_rows:,} -> {flights_clean.shape[0]:,} filas")

    # 2. Eliminar filas donde DEPARTURE_DELAY sea nulo
    before_null = flights_clean.shape[0]
    flights_clean = flights_clean.filter(pl.col("DEPARTURE_DELAY").is_not_null())
    logger.info(f"Filtrado nulos DEPARTURE_DELAY: {before_null:,} -> {flights_clean.shape[0]:,} filas")

    # 3. Crear variable objetivo TARGET_IS_DELAYED (1 si retraso >15 min, 0 si no)
    logger.info("Creando variable objetivo TARGET_IS_DELAYED...")
    flights_clean = flights_clean.with_columns(
        pl.when(pl.col("DEPARTURE_DELAY") > 15)
        .then(1)
        .otherwise(0)
        .cast(pl.Int8)  # Usar Int8 para ahorrar memoria (solo 0 o 1)
        .alias("TARGET_IS_DELAYED")
    )

    # Verificar distribución de la variable objetivo
    delay_stats = flights_clean.group_by("TARGET_IS_DELAYED").agg(
        pl.count().alias("count")
    )
    logger.info("Distribución de TARGET_IS_DELAYED:")
    for row in delay_stats.iter_rows(named=True):
        label = "Retrasado (>15 min)" if row["TARGET_IS_DELAYED"] == 1 else "A tiempo"
        logger.info(f"  {label}: {row['count']:,} filas")

    return flights_clean


def select_columns(flights_clean: pl.DataFrame) -> pl.DataFrame:
    """Selecciona solo las columnas necesarias para la tabla de hechos.
    Elimina duplicados basado en las columnas del modelo para garantizar unicidad en la salida."""
    logger.info("Seleccionando columnas para tabla de hechos...")

    # Columnas necesarias para el modelo
    model_columns = [
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
    ]

    # Verificar que todas las columnas existen
    missing_cols = [col for col in model_columns if col not in flights_clean.columns]
    if missing_cols:
        raise ValueError(f"Columnas faltantes en el DataFrame: {missing_cols}")

    # Seleccionar solo las columnas del modelo y eliminar duplicados
    # Esto garantiza que la salida no tenga filas duplicadas en las columnas del modelo
    flights_selected = flights_clean.select(model_columns).unique()

    initial_rows = flights_clean.shape[0]
    unique_rows = flights_selected.shape[0]
    logger.info(f"Columnas seleccionadas: {len(model_columns)}")
    logger.info(f"Filas después de eliminar duplicados (basado en columnas de modelo): {initial_rows:,} -> {unique_rows:,}")
    if initial_rows != unique_rows:
        logger.info(f"Se eliminaron {initial_rows - unique_rows} filas duplicadas")
    logger.info(f"Filas finales: {unique_rows:,}")

    return flights_selected


def export_to_parquet(
    flights_clean: pl.DataFrame,
    airlines: pl.DataFrame,
    airports: pl.DataFrame,
    processed_dir: Path,
):
    """Exporta los DataFrames a formato Parquet en data/processed/."""
    logger.info("Exportando a Parquet...")

    # Crear directorio si no existe
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Exportar flights_cleaned.parquet
    flights_path = processed_dir / "flights_cleaned.parquet"
    flights_clean.write_parquet(flights_path)
    logger.info(f"flights_cleaned.parquet: {flights_path} ({flights_path.stat().st_size / (1024 * 1024):.2f} MB)")

    # Exportar airlines.parquet
    airlines_path = processed_dir / "airlines.parquet"
    airlines.write_parquet(airlines_path)
    logger.info(f"airlines.parquet: {airlines_path}")

    # Exportar airports.parquet
    airports_path = processed_dir / "airports.parquet"
    airports.write_parquet(airports_path)
    logger.info(f"airports.parquet: {airports_path}")

    logger.info("Exportación completada exitosamente.")

    # Generar automáticamente el reporte de calidad (centralizado en pipeline_common)
    try:
        logger.info("Generando reporte de calidad...")
        from utils.pipeline_common import run_quality_report
        run_quality_report()
    except Exception as exc:  # pragma: no cover – defensive
        logger.error(f"Error al ejecutar el reporte de calidad: {exc}")


def main():
    """Función principal del pipeline ETL."""
    logger.info("=" * 60)
    logger.info("INICIANDO PIPELINE DE LIMPIEZA Y TRANSFORMACIÓN")
    logger.info("=" * 60)

    # Definir rutas del proyecto
    project_root = Path(__file__).parent.parent.parent
    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"

    try:
        # 1. Cargar datos crudos
        flights, airlines, airports = load_raw_data(raw_dir)

        # 2. Limpiar y transformar flights
        flights_clean = clean_flights(flights)

        # 3. Seleccionar columnas específicas
        flights_final = select_columns(flights_clean)

        # 4. Exportar a Parquet
        export_to_parquet(flights_final, airlines, airports, processed_dir)

        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETADO EXITOSAMENTE")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"ERROR en el pipeline: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()