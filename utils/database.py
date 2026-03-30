"""
Módulo de utilidades para conexión a base de datos PostgreSQL.
"""

import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from pathlib import Path


def load_environment_variables(env_path: str | Path | None = None) -> dict:
    """
    Carga variables de entorno desde archivo .env.

    Args:
        env_path: Ruta opcional al archivo .env. Si None, busca automáticamente.

    Returns:
        Diccionario con las variables de entorno cargadas.

    Raises:
        ValueError: Si faltan variables requeridas.
    """
    # Cargar desde archivo específico o buscar automáticamente
    if env_path and Path(env_path).exists():
        load_dotenv(env_path)
    else:
        load_dotenv()

    # Verificar variables requeridas
    required_vars = ["DB_USER", "DB_PASSWORD", "DB_HOST", "DB_PORT", "DB_NAME"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        raise ValueError(f"Variables de entorno faltantes: {missing_vars}")

    return {var: os.getenv(var) for var in required_vars}


def create_database_engine(db_name: str | None = None) -> create_engine:
    """
    Crea engine de SQLAlchemy para conexión a PostgreSQL.

    Args:
        db_name: Nombre de la base de datos. Si None, usa DB_NAME del .env.

    Returns:
        Engine de SQLAlchemy configurado.

    Raises:
        ValueError: Si no hay variables de entorno cargadas.
        ConnectionError: Si no se puede conectar a PostgreSQL.
    """
    # Verificar que las variables estén cargadas
    required_vars = ["DB_USER", "DB_PASSWORD", "DB_HOST", "DB_PORT"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        raise ValueError(
            f"Variables de entorno faltantes: {missing_vars}. "
            "Ejecuta load_environment_variables() primero."
        )

    # Usar nombre de BD proporcionado o del entorno
    database = db_name or os.getenv("DB_NAME")

    # Construir URL de conexión
    db_url = (
        f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{database}"
    )

    engine = create_engine(db_url)

    # Probar conexión
    try:
        with engine.connect() as conn:
            pass
    except Exception as e:
        raise ConnectionError(f"No se pudo conectar a PostgreSQL: {e}")

    return engine
