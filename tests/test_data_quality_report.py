"""
Tests unitarios para el módulo de reporte de calidad de datos (02_data_quality_report.py).
Se utilizan DataFrames sintéticos para probar las funciones de métricas.
"""

import importlib.util
import json
import sys
from pathlib import Path
import pytest

try:
    import polars as pl
except ImportError:
    pytest.skip("Polars no está instalado", allow_module_level=True)

# Ruta al script de calidad
MODULE_PATH = Path(__file__).parent.parent / "src" / "etl" / "02_data_quality_report.py"


def load_quality_report_module():
    """Carga dinámicamente el módulo 02_data_quality_report."""
    spec = importlib.util.spec_from_file_location("quality_report", MODULE_PATH)
    if spec is None:
        raise FileNotFoundError(f"No se pudo encontrar el módulo en {MODULE_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["quality_report"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def qr_module():
    return load_quality_report_module()


def test_compute_nulls(qr_module):
    """Prueba el cálculo de nulos."""
    df = pl.DataFrame(
        {
            "a": [1, None, 3, None],
            "b": [None, 2, 3, 4],
            "c": [1, 2, 3, 4],
        }
    )
    nulls = qr_module.compute_nulls(df)
    assert nulls["a"]["null_count"] == 2
    assert nulls["a"]["null_percentage"] == 50.0
    assert nulls["b"]["null_count"] == 1
    assert nulls["c"]["null_count"] == 0


def test_compute_duplicates(qr_module):
    """Prueba el cálculo de duplicados."""
    df = pl.DataFrame(
        {
            "x": [1, 1, 2, 3],
            "y": ["a", "a", "b", "c"],
        }
    )
    dup = qr_module.compute_duplicates(df)
    assert dup["duplicate_count"] == 1
    assert dup["unique_rows"] == 3
    assert dup["duplicate_percentage"] == pytest.approx(25.0)


def test_compute_outliers_iqr(qr_module):
    """Prueba la detección de outliers por IQR."""
    df = pl.DataFrame({"val": [1, 2, 3, 4, 5, 100]})  # 100 es outlier claro
    outliers = qr_module.compute_outliers_iqr(df)
    assert outliers["val"]["outlier_count"] == 1
    assert outliers["val"]["outlier_percentage"] == pytest.approx(16.666, rel=0.01)


def test_generate_report_structure(qr_module):
    """Verifica que generate_report devuelve un diccionario con todas las claves."""
    df = pl.DataFrame(
        {
            "num": [1, 2, 3, 4],
            "cat": ["a", "b", "a", "b"],
        }
    )
    report = qr_module.generate_report(df)
    assert "total_rows" in report
    assert "total_columns" in report
    assert "nulls" in report
    assert "duplicates" in report
    assert "basic_statistics" in report
    assert "outliers_iqr" in report


def test_save_report(tmp_path, qr_module):
    """Prueba que save_report crea un archivo JSON válido."""
    report = {
        "total_rows": 10,
        "total_columns": 2,
        "nulls": {},
        "duplicates": {"duplicate_count": 0},
        "basic_statistics": {},
        "outliers_iqr": {},
    }
    qr_module.save_report(report, tmp_path)
    report_file = tmp_path / "data_quality_report.json"
    assert report_file.exists()
    with open(report_file) as f:
        loaded = json.load(f)
    assert loaded["total_rows"] == 10


def test_log_summary(caplog, qr_module):
    """Verifica que log_summary escribe en el logger (no falla)."""
    import logging
    caplog.set_level(logging.INFO)
    report = {
        "total_rows": 100,
        "total_columns": 3,
        "nulls": {"col1": {"null_count": 5, "null_percentage": 5.0}},
        "duplicates": {"duplicate_count": 2, "duplicate_percentage": 2.0, "unique_rows": 98},
        "basic_statistics": {},
        "outliers_iqr": {"num": {"outlier_count": 1, "outlier_percentage": 1.0}},
    }
    qr_module.log_summary(report)
    # Verificar que algunas frases clave aparecen en los logs
    assert any("Total de filas:" in rec.message for rec in caplog.records)
    assert any("duplicados" in rec.message.lower() for rec in caplog.records)
