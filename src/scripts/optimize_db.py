#!/usr/bin/env python3
"""
Script de optimización de base de datos: crea modelo en estrella e índices.
Carga tablas de dimensión y define llaves primarias e índices para rendimiento.
"""

import pandas as pd
from pathlib import Path
from sqlalchemy import text
import sys

# Importar utilidades compartidas
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.database import load_environment_variables, create_database_engine
from utils.logging_config import setup_logger

# Configurar logger
logger = setup_logger("04_database_optimization")


def load_dimension_tables(engine, processed_dir: Path):
    """Carga tablas de dimensión desde archivos Parquet a PostgreSQL."""
    logger.info("Cargando tablas de dimensión...")

    # Cargar airlines.parquet
    airlines_path = processed_dir / "airlines.parquet"
    if not airlines_path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {airlines_path}")

    df_airlines = pd.read_parquet(airlines_path)
    logger.info(f"airlines.parquet: {df_airlines.shape[0]} filas × {df_airlines.shape[1]} columnas")

    # Cargar airports.parquet
    airports_path = processed_dir / "airports.parquet"
    if not airports_path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {airports_path}")

    df_airports = pd.read_parquet(airports_path)
    logger.info(f"airports.parquet: {df_airports.shape[0]} filas × {df_airports.shape[1]} columnas")

    # Cargar a PostgreSQL como tablas de dimensión
    logger.info("Subiendo dim_airlines a PostgreSQL...")
    df_airlines.to_sql(
        name="dim_airlines", con=engine, if_exists="replace", index=False
    )

    logger.info("Subiendo dim_airports a PostgreSQL...")
    df_airports.to_sql(
        name="dim_airports", con=engine, if_exists="replace", index=False
    )

    logger.info("Tablas de dimension cargadas exitosamente")


def optimize_database(engine):
    """Ejecuta sentencias SQL de optimización: llaves primarias e índices."""
    logger.info("Ejecutando optimizaciones SQL...")

    sql_commands = [
        # Definir llaves primarias en tablas de dimensión
        'ALTER TABLE dim_airlines ADD PRIMARY KEY ("IATA_CODE");',
        'ALTER TABLE dim_airports ADD PRIMARY KEY ("IATA_CODE");',
        # Crear índices en tabla de hechos para acelerar filtros comunes
        'CREATE INDEX IF NOT EXISTS idx_fact_month ON fact_flights ("MONTH");',
        'CREATE INDEX IF NOT EXISTS idx_fact_airline ON fact_flights ("AIRLINE");',
        'CREATE INDEX IF NOT EXISTS idx_fact_origin ON fact_flights ("ORIGIN_AIRPORT");',
        'CREATE INDEX IF NOT EXISTS idx_fact_target ON fact_flights ("TARGET_IS_DELAYED");',
        # Índice compuesto para consultas frecuentes (mes + aerolinea)
        'CREATE INDEX IF NOT EXISTS idx_fact_month_airline ON fact_flights ("MONTH", "AIRLINE");',
    ]

    with engine.connect() as conn:
        for i, sql in enumerate(sql_commands, 1):
            try:
                logger.info(f"Ejecutando comando {i}/{len(sql_commands)}: {sql[:50]}...")
                conn.execute(text(sql))
                conn.commit()
            except Exception as e:
                logger.warning(f"Advertencia (posible duplicado): {e}")
                conn.rollback()


def verify_optimizations(engine):
    """Verifica que las optimizaciones se aplicaron correctamente."""
    logger.info("Verificando optimizaciones...")

    with engine.connect() as conn:
        # Verificar índices en fact_flights
        result = conn.execute(
            text("""
            SELECT indexname, indexdef
            FROM pg_indexes
            WHERE tablename = 'fact_flights'
            ORDER BY indexname;
            """)
        )
        indexes = result.fetchall()

        logger.info(f"Índices en fact_flights: {len(indexes)}")
        for idx in indexes:
            logger.info(f"  - {idx[0]}")

        # Verificar llaves primarias en dimensiones
        for table in ["dim_airlines", "dim_airports"]:
            result = conn.execute(
                text(f"""
                SELECT constraint_name, constraint_type
                FROM information_schema.table_constraints
                WHERE table_name = '{table}' AND constraint_type = 'PRIMARY KEY';
                """)
            )
            pk = result.fetchone()
            if pk:
                logger.info(f"[OK] Llave primaria en {table}: {pk[0]}")
            else:
                logger.warning(f"No se encontro llave primaria en {table}")


def main():
    """Función principal del pipeline de optimización."""
    logger.info("=" * 60)
    logger.info("INICIANDO OPTIMIZACIÓN DE BASE DE DATOS")
    logger.info("=" * 60)

    project_root = Path(__file__).parent.parent.parent
    processed_dir = project_root / "data" / "processed"
    env_path = project_root / ".env"

    try:
        # 1. Cargar variables de entorno
        load_environment_variables(str(env_path) if env_path.exists() else None)

        # 2. Crear conexión a PostgreSQL
        engine = create_database_engine()

        # 3. Cargar tablas de dimensión
        load_dimension_tables(engine, processed_dir)

        # 4. Ejecutar optimizaciones SQL
        optimize_database(engine)

        # 5. Verificar optimizaciones
        verify_optimizations(engine)

        logger.info("=" * 60)
        logger.info("OPTIMIZACIÓN DE BASE DE DATOS COMPLETADA")
        logger.info("=" * 60)

        logger.info("MODELO EN ESTRELLA CREADO:")
        logger.info("  - dim_airlines (llave primaria: IATA_CODE)")
        logger.info("  - dim_airports (llave primaria: IATA_CODE)")
        logger.info("  - fact_flights (tabla de hechos)")
        logger.info("  - Índices creados para filtros rápidos")

    except Exception as e:
        logger.error(f"ERROR en la optimizacion: {e}", exc_info=True)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()