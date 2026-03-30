# Aviation Delay Prediction

Sistema de predicción de retrasos en vuelos usando Machine Learning. Este proyecto implementa un pipeline completo de ETL, entrenamiento de modelo y almacenamiento de datos para predecir si un vuelo tendrá un retraso mayor a 15 minutos.

## Descripción del Problema

El retraso de vuelos es un problema significativo en la industria de la aviación. Este proyecto utiliza datos históricos de vuelos de la FAA para entrenar un modelo de LightGBM que predice la probabilidad de retraso, permitiendo:

- Identificar patrones de retraso por aerolínea, aeropuerto o mes
- Predecir probabilidades de retraso en tiempo real
- Generar insights para la toma de decisiones

## Dataset

Datos de [usdot/flight-delays](https://www.kaggle.com/datasets/usdot/flight-delays) en Kaggle:
- **Fuente**: U.S. Department of Transportation
- **Contenido**: Historial de vuelos domésticos en EE.UU.
- **Variables principales**: AIRLINE, ORIGIN_AIRPORT, DESTINATION_AIRPORT, SCHEDULED_DEPARTURE, DISTANCE, DEPARTURE_DELAY

## Estructura del Proyecto

```
aviation-delay-prediction/
├── data/
│   ├── raw/              # Datos crudos descargados de Kaggle
│   └── processed/        # Datos limpios en formato Parquet
├── src/
│   ├── etl/
│   │   ├── 00_download_data.py      # Descarga dataset de Kaggle
│   │   ├── 01_clean_and_transform.py # Limpieza y transformación con Polars
│   │   ├── 03_load_to_postgres.py   # Carga predicciones a PostgreSQL
│   │   └── 04_database_optimization.py # Crea índices y modelo estrella
│   └── models/
│       ├── 02_train_model.py        # Entrenamiento LightGBM
│       └── lgbm_flight_delay.pkl    # Modelo guardado
├── tests/
│   ├── test_data.py       # Tests de calidad de datos
│   └── test_model.py      # Tests del modelo
├── utils/                 # Utilidades compartidas
├── notebooks/             # Notebooks de exploración
├── .env                   # Variables de entorno (no versionar)
├── .env.example           # Ejemplo de variables de entorno
├── requirements.txt       # Dependencias
└── README.md
```

## Requisitos

- Python 3.10+
- PostgreSQL 14+
- 8GB RAM mínimo (recomendado 16GB para datasets grandes)

## Instalación

### 1. Clonar el repositorio

```bash
git clone <repo-url>
cd aviation-delay-prediction
```

### 2. Crear entorno virtual

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# o
.venv\Scripts\activate     # Windows
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Configurar variables de entorno

Copiar `.env.example` a `.env` y ajustar:

```bash
cp .env.example .env
```

Editar `.env` con tus credenciales de PostgreSQL:

```ini
DB_USER=postgres
DB_PASSWORD=tu_password
DB_HOST=localhost
DB_PORT=5432
DB_NAME=aviation_analytics
```

### 5. Configurar base de datos

```bash
psql -U postgres -c "CREATE DATABASE aviation_analytics;"
```

## Uso

### Paso 1: Descargar datos

```bash
python src/etl/00_download_data.py
```

> **Nota**: Requiere autenticación en Kaggle. Ejecuta `kaggle login` primero o configura `KAGGLE_USERNAME` y `KAGGLE_KEY` en `.env`.

### Paso 2: Limpiar y transformar datos

```bash
python src/etl/01_clean_and_transform.py
```

Este script:
- Filtra vuelos cancelados
- Elimina registros con valores nulos
- Crea variable objetivo `TARGET_IS_DELAYED` (1 si retraso > 15 min)
- Exporta a formato Parquet

### Paso 3: Entrenar modelo

```bash
python src/models/02_train_model.py
```

Este script:
- Carga datos procesados
- Divide train/test (80/20) con estratificación
- Entrena LightGBM con manejo de desbalance
- Evalúa con F1-Score, ROC-AUC, precisión y recall
- Guarda modelo en `src/models/lgbm_flight_delay.pkl`

### Paso 4: Cargar datos a PostgreSQL

```bash
python src/etl/03_load_to_postgres.py
```

Genera predicciones y carga a la tabla `fact_flights`.

### Paso 5: Optimizar base de datos

```bash
python src/etl/04_database_optimization.py
```

Crea:
- Tablas de dimensión (`dim_airlines`, `dim_airports`)
- Llaves primarias
- Índices para consultas frecuentes

## Ejecutar Tests

```bash
pytest tests/ -v
```

## Pipeline Completo

Para ejecutar todo el pipeline en orden:

```bash
python src/etl/00_download_data.py && \
python src/etl/01_clean_and_transform.py && \
python src/models/02_train_model.py && \
python src/etl/03_load_to_postgres.py && \
python src/etl/04_database_optimization.py
```

## Modelo en Estrella

El esquema de base de datos sigue un modelo dimensional:

```
                    ┌─────────────────┐
                    │  fact_flights   │
                    │  (tabla hechos) │
                    └────────┬────────┘
                             │
           ┌─────────────────┼─────────────────┐
           │                 │                 │
           ▼                 ▼                 ▼
    ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
    │dim_airlines │   │dim_airports │   │  (tiempo)   │
    │             │   │             │   │             │
    │ IATA_CODE   │   │ IATA_CODE   │   │             │
    └─────────────┘   └─────────────┘   └─────────────┘
```

## Métricas del Modelo

El modelo se evalúa con:

| Métrica | Descripción |
|---------|-------------|
| **F1-Score** | Balance entre precisión y recall |
| **ROC-AUC** | Capacidad de discriminación |
| **Precisión** | Exactitud en predicciones positivas |
| **Recall** | Capacidad de detectar retrasos reales |

## Dependencias

| Paquete | Versión | Propósito |
|---------|---------|-----------|
| pandas | >=2.0 | Manipulación de datos |
| polars | >=0.19 | ETL de alto rendimiento |
| lightgbm | >=4.0 | Modelo de ML |
| scikit-learn | >=1.3 | Utilidades de ML |
| sqlalchemy | >=2.0 | ORM para PostgreSQL |
| psycopg2-binary | >=2.9 | Driver PostgreSQL |
| python-dotenv | >=1.0 | Gestión de .env |
| pytest | >=7.0 | Testing |

## Autor

Proyecto desarrollado como parte de un portafolio de Ciencia de Datos y Machine Learning.

## Licencia

MIT License
