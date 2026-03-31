#!/usr/bin/env python3
"""
Script to retrain the flight delay prediction model.

This script:
1. Loads the processed dataset from data/processed/flights_cleaned.parquet
2. Runs the training pipeline using src/models/02_train_model.py
3. Saves the new model to src/models/lgbm_flight_delay.pkl
4. Generates a data quality report

Usage:
- python src/models/retrain_model.py
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import utilities
from utils.logging_config import setup_logger
from utils.database import load_environment_variables
from utils.pipeline_common import add_project_root_to_path
add_project_root_to_path()

# Setup logger
logger = setup_logger("retrain-model")

# Import required modules
import joblib
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
import os

def load_processed_data(processed_dir: Path) -> pd.DataFrame:
    """Load processed parquet data."""
    processed_path = processed_dir / "flights_cleaned.parquet"
    if not processed_path.exists():
        logger.error(f"Processed data file not found: {processed_path}")
        raise FileNotFoundError(f"Processed data file not found: {processed_path}")

    df = pd.read_parquet(processed_path)
    logger.info(f"Loaded processed data: {df.shape[0]:,} rows, {df.shape[1]} columns")
    return df

def train_model(data):
    """Train LightGBM model and save to file."""
    logger.info("Starting model training...")

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score, roc_auc_score
    import lightgbm as lgb

    # Define features and target
    feature_cols = [col for col in data.columns if col not in ['TARGET_IS_DELAYED', 'DEPARTURE_DELAY']]
    X = data[feature_cols]
    y = data['TARGET_IS_DELAYED']

    # Stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # LightGBM parameters
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbment',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0
    }

    # Train model
    logger.info("Training LightGBM model...")
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    logger.info(f"Model trained - F1: {f1:.3f}, ROC-AUC: {roc_auc:.3f}")

    # Save model
    model_path = Path("../models/lgbm_flight_delay.pkl")
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")

    # Generate data quality report
    from src.scripts.run_quality_report import generate_report
    df = data.drop(columns=['TARGET_IS_DELAYED'], errors='ignore')
    report = generate_report(skill.model if 'skill' in globals() else data)  # Placeholder
    # Actually generate report using existing script
    from subprocess import run
    result = run([sys.executable, 'src/scripts/run_quality_report.py'],
                 capture_output=True, text=True)
    logger.info("Data quality report generated")
    if result.returncode != 0:
        logger.warning(f"Report generation error: {result.stderr}")

    return model

def main():
    """Main function for retraining pipeline."""
    logger.info("=" * 60)
    logger.info("INICIANDO REENTRENAMIENTO DEL MODELO")
    logger.info("=" * 60)

    try:
        # Load processed data
        processed_dir = Path("../data/processed")
        data = load_processed_data(processed_dir)

        # Train model
        model = train_model(data)

        logger.info("REENTRENAMIENTO COMPLETADO")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"REENTRENAMIENTO FALLÓ: {e}", exc_info=True)
        raise SystemExit(1)

if __name__ == "__main__":
    main()