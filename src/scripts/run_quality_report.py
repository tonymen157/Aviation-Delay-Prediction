#!/usr/bin/env python3
"""
Script de reporte de calidad de datos para el dataset de vuelos procesado.
Calcula métricas de nulos, duplicados, outliers y distribuciones básicas.
Guarda un reporte JSON en data/reports/ y opcionalmente un resumen en logs.
"""

import json
from pathlib import Path

# Import utilities
from utils.pipeline_common import add_project_root_to_path
add_project_root_to_path()

from utils.logging_config import setup_logger

try:
    import polars as pl
except ImportError:
    print("ERROR: Polars no está instalado. Ejecuta: pip install polars")
    raise SystemExit(1)

# Configurar logger
logger = setup_logger("02_data_quality_report")


def load_clean_parquet(processed_dir: Path) -> pl.DataFrame:
    """Carga el archivo Parquet de vuelos limpios."""
    parquet_path = processed_dir / "flights_cleaned.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"No se encontró {parquet_path}")
    logger.info(f"Cargando datos limpios desde {parquet_path}")
    df = pl.read_parquet(parquet_path)
    logger.info(f"Datos cargados: {df.shape[0]:,} filas × {df.shape[1]} columnas")
    return df


def compute_nulls(df: pl.DataFrame) -> dict:
    """Calcula el número y porcentaje de valores nulos por columna."""
    null_counts = df.null_count()
    total_rows = df.shape[0]
    null_info = {}
    for col in df.columns:
        null_count = null_counts[col][0]  # null_count devuelve una fila
        null_pct = (null_count / total_rows) * 100 if total_rows > 0 else 0
        null_info[col] = {
            "null_count": int(null_count),
            "null_percentage": round(null_pct, 2)
        }
    return null_info


def compute_duplicates(df: pl.DataFrame) -> dict:
    """Calcula el número de filas duplicadas (considerando todas las columnas)."""
    total_rows = df.shape[0]
    unique_rows = df.unique().shape[0]
    duplicate_count = total_rows - unique_rows
    duplicate_pct = (duplicate_count / total_rows) * 100 if total_rows > 0 else 0
    return {
        "duplicate_count": int(duplicate_count),
        "duplicate_percentage": round(duplicate_pct, 2),
        "unique_rows": int(unique_rows)
    }


def compute_basic_stats(df: pl.DataFrame) -> dict:
    """Calcula estadísticas básicas (min, max, mean, median) para columnas numéricas."""
    numeric_cols = [col for col in df.columns if df[col].dtype in (pl.Int64, pl.Int32, pl.Float64, pl.Float32)]
    stats = {}
    for col in numeric_cols:
        col_data = df[col]
        stats[col] = {
            "min": float(col_data.min()) if col_data.min() is not None else None,
            "max": float(col_data.max()) if col_data.max() is not None else None,
            "mean": float(col_data.mean()) if col_data.mean() is not None else None,
            "median": float(col_data.median()) if col_data.median() is not None else None,
        }
    return stats


def compute_outliers_iqr(df: pl.DataFrame) -> dict:
    """
    Detecta outliers usando el método IQR (Intervalo Cuartílico) para columnas numéricas.
    Define outliers como valores < Q1 - 1.5*IQR o > Q3 + 1.5*IQR.
    """
    numeric_cols = [col for col in df.columns if df[col].dtype in (pl.Int64, pl.Int32, pl.Float64, pl.Float32)]
    outliers_info = {}
    for col in numeric_cols:
        col_data = df[col].drop_nulls()  # Ignorar nulos para el cálculo de IQR
        if col_data.len() == 0:
            outliers_info[col] = {"outlier_count": 0, "outlier_percentage": 0.0, "lower_bound": None, "upper_bound": None}
            continue
        q1 = col_data.quantile(0.25)
        q3 = col_data.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_mask = (col_data < lower_bound) | (col_data > upper_bound)
        outlier_count = outlier_mask.sum()
        outlier_pct = (outlier_count / col_data.len()) * 100 if col_data.len() > 0 else 0
        outliers_info[col] = {
            "outlier_count": int(outlier_count),
            "outlier_percentage": round(outlier_pct, 2),
            "lower_bound": float(lower_bound) if lower_bound is not None else None,
            "upper_bound": float(upper_bound) if upper_bound is not None else None
        }
    return outliers_info


def generate_report(df: pl.DataFrame) -> dict:
    """Genera un diccionario con todas las métricas de calidad."""
    logger.info("Calculando métricas de calidad de datos...")
    report = {
        "total_rows": int(df.shape[0]),
        "total_columns": int(df.shape[1]),
        "nulls": compute_nulls(df),
        "duplicates": compute_duplicates(df),
        "basic_statistics": compute_basic_stats(df),
        "outliers_iqr": compute_outliers_iqr(df)
    }
    return report


def save_report(report: dict, report_dir: Path):
    """Guarda el reporte como JSON en el directorio especificado."""
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "data_quality_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"Reporte de calidad guardado en: {report_path}")


def log_summary(report: dict):
    """Imprime un resumen legible en los logs."""
    logger.info("=" * 60)
    logger.info("RESUMEN DE CALIDAD DE DATOS")
    logger.info("=" * 60)
    logger.info(f"Total de filas: {report['total_rows']:,}")
    logger.info(f"Total de columnas: {report['total_columns']}")
    logger.info("--- Nulos por columna (top 5) ---")
    nulls = report["nulls"]
    sorted_nulls = sorted(nulls.items(), key=lambda x: x[1]["null_count"], reverse=True)
    for col, info in sorted_nulls[:5]:
        logger.info(f"  {col}: {info['null_count']:,} nulos ({info['null_percentage']}%)")
    logger.info("--- Duplicados ---")
    dup = report["duplicates"]
    logger.info(f"  Filas duplicadas: {dup['duplicate_count']:,} ({dup['duplicate_percentage']}%)")
    logger.info(f"  Filas únicas: {dup['unique_rows']:,}")
    logger.info("--- Outliers (IQR) por columna (top 5) ---")
    outliers = report["outliers_iqr"]
    sorted_outliers = sorted(outliers.items(), key=lambda x: x[1]["outlier_count"], reverse=True)
    for col, info in sorted_outliers[:5]:
        logger.info(f"  {col}: {info['outlier_count']:,} outliers ({info['outlier_percentage']}%)")
    logger.info("=" * 60)


def main():
    """Función principal del reporte de calidad de datos."""
    logger.info("=" * 60)
    logger.info("INICIANDO REPORTE DE CALIDAD DE DATOS")
    logger.info("=" * 60)

    # Definir rutas del proyecto
    project_root = Path(__file__).parent.parent.parent
    processed_dir = project_root / "data" / "processed"
    report_dir = project_root / "data" / "reports"

    try:
        # 1. Cargar datos limpios
        df = load_clean_parquet(processed_dir)

        # 2. Generar reporte
        report = generate_report(df)

        # 3. Guardar reporte
        save_report(report, report_dir)

        # 4. Mostrar resumen en logs
        log_summary(report)

        logger.info("=" * 60)
        logger.info("REPORTE DE CALIDAD DE DATOS COMPLETADO")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"ERROR en el reporte de calidad: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()