#!/usr/bin/env python3
"""
Script para cargar datos procesados y predicciones a PostgreSQL.
Lee datos Parquet, genera predicciones con modelo LightGBM y carga a base de datos.
"""

import pandas as pd
import joblib
from pathlib import Path
import sys

# Importar utilidades compartidas
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.database import load_environment_variables, create_database_engine
from utils.logging_config import setup_logger


# Configurar logger
logger = setup_logger("03_load_to_postgres")


def load_and_sample_data(
    parquet_path: Path, sample_size: int = 500_000
) -> pd.DataFrame:
    """Carga datos Parquet y toma muestra aleatoria representativa."""
    logger.info(f"Cargando datos desde {parquet_path}...")

    # Leer con pandas (compatible con to_sql)
    df = pd.read_parquet(parquet_path)
    logger.info(f"Datos originales: {df.shape[0]:,} filas × {df.shape[1]} columnas")

    # Tomar muestra aleatoria representativa
    if df.shape[0] > sample_size:
        logger.info(f"Tomando muestra aleatoria de {sample_size:,} filas...")
        df_sample = df.sample(n=sample_size, random_state=42)
        logger.info(f"Muestra final: {df_sample.shape[0]:,} filas")
        return df_sample
    else:
        logger.info("Dataset más pequeño que tamaño de muestra, usando datos completos")
        return df


def load_model_and_predict(df: pd.DataFrame, model_path: Path) -> pd.DataFrame:
    """Carga modelo LightGBM y genera predicciones de probabilidad."""
    logger.info(f"Cargando modelo desde {model_path}...")

    # Cargar modelo entrenado
    model = joblib.load(model_path)
    logger.info(f"Modelo cargado: {model.__class__.__name__}")

    # Preparar features para predicción (igual que en entrenamiento)
    # Eliminar columnas que no son features (TARGET_IS_DELAYED y DEPARTURE_DELAY)
    X_pred = df.drop(columns=["TARGET_IS_DELAYED", "DEPARTURE_DELAY"], errors="ignore")

    # Asegurar que columnas categóricas estén en formato correcto
    categorical_cols = ["AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT"]
    for col in categorical_cols:
        if col in X_pred.columns:
            X_pred[col] = X_pred[col].astype("category")

    logger.info(f"Generando predicciones para {X_pred.shape[0]:,} filas...")

    # Predecir probabilidades (segunda columna = probabilidad de clase 1)
    delay_probabilities = model.predict_proba(X_pred)[:, 1]

    # Agregar columna de probabilidad al dataframe original
    df["DELAY_PROBABILITY"] = delay_probabilities

    logger.info(f"Predicciones generadas. Rango de probabilidad: [{delay_probabilities.min():.3f}, {delay_probabilities.max():.3f}]")

    return df


def load_to_postgres(df: pd.DataFrame, engine, table_name: str = "fact_flights"):
    """Carga dataframe a PostgreSQL usando to_sql."""
    logger.info(f"Cargando datos a tabla '{table_name}' en PostgreSQL...")

    # Cargar a PostgreSQL con chunks para evitar problemas de memoria
    df.to_sql(
        name=table_name,
        con=engine,
        if_exists="replace",  # Reemplazar tabla si existe
        index=False,  # No incluir índice de pandas
        chunksize=10_000,  # Cargar en bloques de 10k filas
        method="multi",  # Usar inserciones múltiples para mejor rendimiento
    )

    logger.info(f"Datos cargados exitosamente: {df.shape[0]:,} filas x {df.shape[1]} columnas")


def main():
    """Función principal del pipeline de carga."""
    logger.info("=" * 60)
    logger.info("INICIANDO PIPELINE DE CARGA A POSTGRESQL")
    logger.info("=" * 60)

    # Definir rutas del proyecto
    project_root = Path(__file__).parent.parent.parent
    parquet_path = project_root / "data" / "processed" / "flights_cleaned.parquet"
    model_path = project_root / "src" / "models" / "lgbm_flight_delay.pkl"
    env_path = project_root / ".env"

    try:
        # 1. Cargar variables de entorno
        load_environment_variables(str(env_path) if env_path.exists() else None)

        # 2. Crear conexión a PostgreSQL
        engine = create_database_engine()

        # 3. Cargar datos y tomar muestra
        df_sample = load_and_sample_data(parquet_path, sample_size=500_000)

        # 4. Cargar modelo y generar predicciones
        df_with_predictions = load_model_and_predict(df_sample, model_path)

        # 5. Cargar a PostgreSQL
        load_to_postgres(df_with_predictions, engine, table_name="fact_flights")

        logger.info("=" * 60)
        logger.info("PIPELINE DE CARGA COMPLETADO EXITOSAMENTE")
        logger.info("=" * 60)

        # Resumen final
        logger.info("=== RESUMEN FINAL ===")
        logger.info(f"  Filas cargadas: {df_with_predictions.shape[0]:,}")
        logger.info(f"  Columnas: {df_with_predictions.shape[1]}")
        logger.info(f"  Tabla PostgreSQL: fact_flights")
        logger.info(f"  Probabilidad promedio de retraso: {df_with_predictions['DELAY_PROBABILITY'].mean():.3f}")

    except Exception as e:
        logger.error(f"ERROR en el pipeline: {e}", exc_info=True)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
