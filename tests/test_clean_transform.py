"""
Tests unitarios para el módulo de limpieza y transformación (01_clean_and_transform.py).

Se utilizan datos mínimos en memoria para probar cada función de forma aislada.
"""

import importlib.util
import sys
from pathlib import Path

import polars as pl
import pytest

# Ruta al script original (con nombre numérico)
MODULE_PATH = Path(__file__).parent.parent / "src" / "etl" / "01_clean_and_transform.py"


def load_clean_transform_module():
    """Carga el módulo 01_clean_and_transform.py dinámicamente."""
    spec = importlib.util.spec_from_file_location("clean_transform", MODULE_PATH)
    if spec is None:
        raise FileNotFoundError(f"No se pudo encontrar el módulo en {MODULE_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["clean_transform"] = mod  # registrar para evitar duplicados
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def ct_module():
    """Fixture que carga el módulo una vez por test session."""
    return load_clean_transform_module()


def test_load_raw_data_success(tmp_path, ct_module):
    """Prueba que load_raw_data lee CSVs y devuelve DataFrames."""
    # Crear CSV mínimos
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    flights_df = pl.DataFrame(
        {
            "CANCELLED": [0, 1],
            "DEPARTURE_DELAY": [10.0, None],
            "MONTH": [1, 2],
            "DAY": [15, 16],
            "DAY_OF_WEEK": [3, 4],
            "AIRLINE": ["AA", "UA"],
            "ORIGIN_AIRPORT": ["LAX", "JFK"],
            "DESTINATION_AIRPORT": ["JFK", "LAX"],
            "SCHEDULED_DEPARTURE": [1200, 1300],
            "DISTANCE": [2500, 2600],
        }
    )
    flights_df.write_csv(raw_dir / "flights.csv")
    airlines_df = pl.DataFrame({"IATA_CODE": ["AA", "UA"], "AIRLINE": ["American", "United"]})
    airlines_df.write_csv(raw_dir / "airlines.csv")
    airports_df = pl.DataFrame({"IATA_CODE": ["LAX", "JFK"], "AIRPORT": ["Los Angeles", "JFK"]})
    airports_df.write_csv(raw_dir / "airports.csv")

    flights, airlines, airports = ct_module.load_raw_data(raw_dir)

    assert isinstance(flights, pl.DataFrame)
    assert flights.shape[0] == 2
    assert "CANCELLED" in flights.columns
    assert airlines.shape[0] == 2
    assert airports.shape[0] == 2


def test_load_raw_data_missing_flights(tmp_path, ct_module):
    """Debe fallar si falta flights.csv."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        ct_module.load_raw_data(raw_dir)


def test_clean_flights_basic(ct_module):
    """Comprueba que clean_flights filtra y crea TARGET_IS_DELAYED correctamente."""
    df = pl.DataFrame(
        {
            "CANCELLED": [0, 0, 1, 0],
            "DEPARTURE_DELAY": [10.0, None, 5.0, 30.0],
        }
    )
    result = ct_module.clean_flights(df)

    # Se mantienen filas con CANCELLED=0 y DEPARTURE_DELAY no nulo
    assert result.shape[0] == 2
    # Columna objetivo binaria
    assert "TARGET_IS_DELAYED" in result.columns
    target_vals = result.select("TARGET_IS_DELAYED").to_series().to_list()
    assert set(target_vals) <= {0, 1}
    # Verificar que delay<=15 ->0, >15 ->1 para las filas esperadas
    # Las filas que quedan corresponden a delays 10 y 30
    assert sorted(target_vals) == [0, 1]


def test_select_columns_valid(ct_module):
    """select_columns debe conservar solo las columnas requeridas."""
    df = pl.DataFrame(
        {
            "MONTH": [1],
            "DAY": [2],
            "DAY_OF_WEEK": [3],
            "AIRLINE": ["AA"],
            "ORIGIN_AIRPORT": ["LAX"],
            "DESTINATION_AIRPORT": ["JFK"],
            "SCHEDULED_DEPARTURE": [1200],
            "DISTANCE": [1000],
            "DEPARTURE_DELAY": [5],
            "TARGET_IS_DELAYED": [0],
            "EXTRA_COL": ["extra"],
        }
    )
    result = ct_module.select_columns(df)
    expected = [
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
    assert result.columns == expected


def test_select_columns_missing_raises(ct_module):
    """Debe lanzar ValueError si falta alguna columna requerida."""
    df = pl.DataFrame({"MONTH": [1], "DAY": [2]})
    with pytest.raises(ValueError, match="Columnas faltantes"):
        ct_module.select_columns(df)


def test_export_to_parquet(tmp_path, ct_module):
    """export_to_parquet debe crear tres archivos Parquet."""
    flights = pl.DataFrame({"col1": [1, 2]})
    airlines = pl.DataFrame({"col2": ["a"]})
    airports = pl.DataFrame({"col3": ["x", "y"]})
    processed_dir = tmp_path / "processed"
    ct_module.export_to_parquet(flights, airlines, airports, processed_dir)

    assert (processed_dir / "flights_cleaned.parquet").exists()
    assert (processed_dir / "airlines.parquet").exists()
    assert (processed_dir / "airports.parquet").exists()

    # Verificar que se pueden leer y tienen datos correctos
    flights_read = pl.read_parquet(processed_dir / "flights_cleaned.parquet")
    assert flights_read.shape[0] == 2
    airlines_read = pl.read_parquet(processed_dir / "airlines.parquet")
    assert airlines_read.shape[0] == 1
    airports_read = pl.read_parquet(processed_dir / "airports.parquet")
    assert airports_read.shape[0] == 2


def test_clean_flights_with_target_distribution(ct_module):
    """Verifica que la distribución de TARGET_IS_DELAYED se calcula (log only, pero no falla)."""
    df = pl.DataFrame(
        {
            "CANCELLED": [0] * 100,
            "DEPARTURE_DELAY": [10] * 80 + [30] * 20,  # 20 retrasados >15min
        }
    )
    result = ct_module.clean_flights(df)
    assert result.shape[0] == 100
    target_counts = result.group_by("TARGET_IS_DELAYED").agg(pl.count())
    target_dict = dict(zip(target_counts["TARGET_IS_DELAYED"].to_list(), target_counts["count"].to_list()))
    assert target_dict[0] == 80
    assert target_dict[1] == 20
