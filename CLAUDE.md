# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

- **Install dependencies**: `pip install -r requirements.txt`
- **Run tests**: `pytest tests/ -v`
- **Run a single test**: `pytest tests/test_name.py -v`
- **Execute ETL pipeline step**:
  - Download data: `python src/etl/00_download_data.py`
  - Clean & transform: `python src/etl/01_clean_and_transform.py`
  - Train model: `python src/models/02_train_model.py`
  - Load to PostgreSQL: `python src/etl/03_load_to_postgres.py`
  - Database optimization: `python src/etl/04_database_optimization.py`
- **Run full pipeline**:
  ```bash
  python src/etl/00_download_data.py && \
  python src/etl/01_clean_and_transform.py && \
  python src/models/02_train_model.py && \
  python src/etl/03_load_to_postgres.py && \
  python src/etl/04_database_optimization.py
  ```

## High‑Level Architecture

The project follows a classic data‑science pipeline:

1. **Raw Data (data/raw)** – Kaggle flight‑delay dataset.
2. **ETL Layer (src/etl)** –
   - `00_download_data.py`: fetches and stores raw data.
   - `01_clean_and_transform.py`: cleans, filters canceled flights, creates target `TARGET_IS_DELAYED`.
   - `03_load_to_postgres.py`: pushes predictions to a PostgreSQL table (`fact_flights`).
   - `04_database_optimization.py`: builds dimension tables (`dim_airlines`, `dim_airports`) and adds indexes.
3. **Model Layer (src/models)** –
   - `02_train_model.py`: trains a LightGBM classifier, evaluates with F1, ROC‑AUC, precision, recall, and serializes the model (`lgbm_flight_delay.pkl`).
4. **Utilities (utils)** – Shared helpers (e.g., DB connection, logging).
5. **Tests (tests)** – Unit tests for data quality and model performance.

The database schema uses a **star schema**: `fact_flights` as the fact table linked to dimension tables `dim_airlines`, `dim_airports`, and a time dimension.

## Key Scripts Overview

| Path | Purpose |
|------|---------|
| `src/etl/00_download_data.py` | Downloads the flight‑delay dataset from Kaggle (requires `KAGGLE_USERNAME`/`KAGGLE_KEY`). |
| `src/etl/01_clean_and_transform.py` | Cleans data, removes cancellations, creates binary target `TARGET_IS_DELAYED`. |
| `src/models/02_train_model.py` | Trains LightGBM, performs train/test split (80/20, stratified), saves model to `src/models/lgbm_flight_delay.pkl`. |
| `src/etl/03_load_to_postgres.py` | Loads predictions into PostgreSQL (`fact_flights` table). |
| `src/etl/04_database_optimization.py` | Creates dimension tables and adds indexes for fast look‑ups. |
| `tests/` | Contains `test_data.py` (data quality) and `test_model.py` (model evaluation). |

## Development Workflow

1. **Set up environment** – create a virtual environment, install `requirements.txt`, copy `.env.example` to `.env` and fill PostgreSQL credentials.
2. **Database preparation** – run `psql -U postgres -c "CREATE DATABASE aviation_analytics;"` before executing ETL steps.
3. **Run individual steps** – invoke the Python scripts directly to isolate each stage.
4. **Iterate** – modify scripts in `src/etl/` or `src/models/` as needed; re‑run only the changed steps.
5. **Validate** – after changes, execute `pytest tests/ -v` to ensure data quality and model correctness are maintained.

## Reference Configuration Files

- `.env.example` – Template for environment variables (DB credentials, Kaggle API keys).
- `requirements.txt` – Pinning of all Python dependencies.
- `README.md` – Full project description, installation instructions, and pipeline overview.

## Important Notes

- Do **not** commit `.env` or any credentials; they are excluded via `.gitignore`.
- The project expects PostgreSQL 14+ and a minimum of 8 GB RAM (16 GB recommended for larger datasets).
- All new code should preserve the existing star‑schema design and keep migration scripts (`04_database_optimization.py`) idempotent where possible.