# Development Operations Guide

This document describes the end‑to‑end workflow for setting up the development environment, running the ETL pipelines, executing tests, and contributing to the repository.

---

## 1. Prerequisites

| Tool | Minimum version | Installation |
|------|----------------|--------------|
| Python | 3.11 | `https://www.python.org/downloads/` |
| Git | 2.40+ | `https://git-scm.com/downloads` |
| PostgreSQL | 14+ | Use your preferred installer (e.g., Chocolatey, PostgreSQL website) |
| Docker (optional) | 20+ | `https://www.docker.com/get-started` |

---

## 2. Repository Setup

```bash
# Clone the repository
git clone https://github.com/your-org/aviation-delay-prediction.git
cd aviation-delay-prediction

# Create a virtual environment (recommended)
python -m venv .venv
.\.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/macOS

# Install runtime dependencies
pip install -r requirements.txt

# Install development dependencies (including testing tools)
pip install -r requirements-dev.txt
```

---

## 3. Pre‑Commit Hooks

The repository uses **pre‑commit** to enforce code quality automatically.

```bash
# Install the pre‑commit hook
pre-commit install
```

You can also run the hooks manually on all files:

```bash
pre-commit run --all-files
```

Hooks included:

- **ruff** – linting and style checking
- **black** – automatic code formatting
- **isort** – import ordering
- **mypy** – static type checking

> **Tip:** If a hook fails, fix the reported issues and re‑stage the changes before committing.

---

## 4. Running the ETL Pipelines

All pipeline scripts are located in `src/scripts/`. They can be executed directly or chained together.

### 4.1 Download raw data

```bash
python src/scripts/download_data.py
```

### 4.2 Clean, transform, and generate quality report

```bash
python src/scripts/clean_transform.py
```

The script will:

1. Load raw CSV files from `data/raw/`.
2. Filter, clean, and create the target column `TARGET_IS_DELAYED`.
3. Export the processed data to Parquet files in `data/processed/`.
4. Generate an automated data‑quality report (see `src/scripts/run_quality_report.py`).

### 4.3 Train the LightGBM model

```bash
python src/models/train_model.py
```

The trained model is saved as `src/models/lgbm_flight_delay.pkl`.

### 4.4 Load predictions into PostgreSQL

```bash
python src/scripts/load_to_postgres.py
```

### 4.5 Optimize the database schema

```bash
python src/scripts/optimize_db.py
```

---

## 5. Testing

Unit tests are located under the `tests/` directory and can be executed with **pytest**.

```bash
pytest -v
```

The CI workflow (`.github/workflows/ci.yml`) runs the same command on every push and pull request.

---

## 6. Continuous Integration (CI)

The repository includes a GitHub Actions workflow that:

1. Checks out the code.
2. Sets up Python 3.11.
3. Installs both runtime and development dependencies.
4. Executes `pre-commit` on all files.
5. Runs the full test suite (`pytest`).
6. Executes the data‑quality report generation script.
7. (Optional) Uploads coverage reports to Codecov.

The workflow file lives at `.github/workflows/ci.yml`.

---

## 7. Contribution Workflow

1. **Branch naming** – Use feature‑branch syntax: `feat/short-description`, `fix/issue-number`, `docs/add‑readme`.
2. **Commits** – Follow the conventions in `COMMIT_CONVENTION.md`.
3. **Pull Request** – Open a PR, ensure all checks pass, and get at least one reviewer approval.
4. **Merge** – Merge via **Squash and Merge** to keep a linear history.

---

## 8. FAQ

| Question | Answer |
|----------|--------|
| *I get “`pre‑commit` not found”?* | Ensure you activated the virtual environment and that `pre‑commit` is installed (`pip install pre‑commit`). |
| *My changes fail the mypy check* | Run `mypy .` locally, fix the reported issues, then re‑stage. |
| *The quality‑report script crashes* | Verify that `data/processed/flights_cleaned.parquet` exists and that the `polars` library is installed (`pip install polars`). |
| *How do I run a single script without executing the whole pipeline?* | Execute the script directly with `python src/scripts/<script-name>.py`. |

---

## 9. License

This project is licensed under the **MIT License** – see the `LICENSE` file for details.

---

*Last updated: 2026‑03‑29*