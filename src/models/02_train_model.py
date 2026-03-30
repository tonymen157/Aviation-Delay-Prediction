#!/usr/bin/env python3
"""
Script de entrenamiento de modelo LightGBM para predicción de retrasos de vuelos.
Pipeline: Lectura → Preparación → Entrenamiento → Evaluación → Persistencia.
"""

import polars as pl
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import lightgbm as lgb
import joblib
from pathlib import Path
import sys

# Importar utilidades
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.logging_config import setup_logger

# Configurar logger
logger = setup_logger("02_train_model")


def load_and_prepare_data(parquet_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Carga datos desde Parquet, convierte categóricas y prepara features/target."""
    log("Cargando datos desde Parquet...")

    # Leer con Polars (rápido para Big Data)
    df_pl = pl.read_parquet(parquet_path)
    log(f"  Datos cargados: {df_pl.shape[0]:,} filas × {df_pl.shape[1]} columnas")

    # Convertir columnas de alta cardinalidad a Categorical (LightGBM las maneja nativamente)
    categorical_cols = ["AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT"]
    for col in categorical_cols:
        if col in df_pl.columns:
            df_pl = df_pl.with_columns(pl.col(col).cast(pl.Categorical))
            log(f"  Columna {col} convertida a Categorical")

    # Convertir a pandas para scikit-learn
    log("Convirtiendo a pandas DataFrame...")
    df = df_pl.to_pandas()

    # Definir features (X) y target (y)
    # Eliminar TARGET_IS_DELAYED (target) y DEPARTURE_DELAY (para evitar data leakage)
    X = df.drop(columns=["TARGET_IS_DELAYED", "DEPARTURE_DELAY"])
    y = df["TARGET_IS_DELAYED"]

    log(f"  Features shape: {X.shape}")
    log(f"  Target distribution:\n{y.value_counts(normalize=True).to_string()}")

    return X, y


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> lgb.LGBMClassifier:
    """Entrena modelo LightGBM con manejo de desbalance de clases."""
    log("Configurando LightGBM...")

    # Calcular scale_pos_weight para manejar desbalance (82% vs 18%)
    # scale_pos_weight = (n_muestras_clase0) / (n_muestras_clase1)
    n_negative = (y_train == 0).sum()
    n_positive = (y_train == 1).sum()
    scale_pos_weight = n_negative / n_positive

    log(f"  scale_pos_weight calculado: {scale_pos_weight:.2f}")
    log(f"  Clase 0 (a tiempo): {n_negative:,} muestras")
    log(f"  Clase 1 (retrasado): {n_positive:,} muestras")

    # Configurar clasificador LightGBM
    model = lgb.LGBMClassifier(
        n_estimators=100,  # Número de árboles
        learning_rate=0.1,  # Tasa de aprendizaje
        max_depth=-1,  # Sin límite de profundidad
        num_leaves=31,  # Número de hojas por árbol
        scale_pos_weight=scale_pos_weight,  # CRÍTICO: Manejo de desbalance
        random_state=42,  # Reproducibilidad
        n_jobs=-1,  # Usar todos los cores
        verbose=-1,  # Silenciar warnings
    )

    log("Entrenando modelo LightGBM...")
    # LightGBM maneja columnas categóricas automáticamente si están en el dtype correcto
    model.fit(X_train, y_train)
    log("[OK] Modelo entrenado exitosamente.")

    return model


def evaluate_model(model: lgb.LGBMClassifier, X_test: pd.DataFrame, y_test: pd.Series):
    """Evalúa el modelo y muestra métricas de rendimiento."""
    log("Evaluando modelo en conjunto de prueba...")

    # Predicciones
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probabilidades para clase 1

    # Métricas
    log("\n" + "=" * 60)
    log("REPORTE DE CLASIFICACIÓN:")
    log("=" * 60)
    report = classification_report(
        y_test, y_pred, target_names=["A tiempo", "Retrasado"]
    )
    print(report)

    # ROC-AUC
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    log(f"\nROC-AUC Score: {roc_auc:.4f}")

    # Métricas específicas de la clase 1 (retrasados)
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    f1_clase1 = report_dict["1"]["f1-score"]
    precision_clase1 = report_dict["1"]["precision"]
    recall_clase1 = report_dict["1"]["recall"]

    log("\n" + "=" * 60)
    log("MÉTRICAS CLASE 1 (RETRASADOS >15 min):")
    log("=" * 60)
    log(f"  F1-Score:  {f1_clase1:.4f}")
    log(f"  Precision: {precision_clase1:.4f}")
    log(f"  Recall:    {recall_clase1:.4f}")
    log(f"  ROC-AUC:   {roc_auc:.4f}")

    return {
        "f1_clase1": f1_clase1,
        "roc_auc": roc_auc,
        "precision_clase1": precision_clase1,
        "recall_clase1": recall_clase1,
    }


def save_model(model: lgb.LGBMClassifier, model_path: Path):
    """Guarda el modelo entrenado en disco."""
    log(f"Guardando modelo en: {model_path}")

    # Crear directorio si no existe
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # Guardar con joblib (eficiente para modelos sklearn/lightgbm)
    joblib.dump(model, model_path)
    log(
        f"[OK] Modelo guardado exitosamente ({model_path.stat().st_size / (1024 * 1024):.2f} MB)"
    )


def main():
    """Función principal del pipeline de ML."""
    log("=" * 60)
    log("INICIANDO PIPELINE DE MACHINE LEARNING")
    log("=" * 60)

    # Definir rutas del proyecto
    project_root = Path(__file__).parent.parent.parent
    parquet_path = project_root / "data" / "processed" / "flights_cleaned.parquet"
    model_path = project_root / "src" / "models" / "lgbm_flight_delay.pkl"

    try:
        # 1. Cargar y preparar datos
        X, y = load_and_prepare_data(parquet_path)

        # 2. División de datos (80/20) con estratificación
        log("Dividiendo datos en train/test (80/20) con estratificación...")
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,  # CRÍTICO: Mantener proporción de clases
        )
        log(f"  Train: {X_train.shape[0]:} muestras")
        log(f"  Test:  {X_test.shape[0]:} muestras")

        # 3. Entrenar modelo
        model = train_model(X_train, y_train)

        # 4. Evaluar modelo
        metrics = evaluate_model(model, X_test, y_test)

        # 5. Guardar modelo
        save_model(model, model_path)

        log("=" * 60)
        log("PIPELINE DE ML COMPLETADO EXITOSAMENTE")
        log("=" * 60)

        # Resumen final
        log("\n[RESUMEN] RESUMEN FINAL:")
        log(f"  F1-Score (Clase 1): {metrics['f1_clase1']:.4f}")
        log(f"  ROC-AUC:            {metrics['roc_auc']:.4f}")
        log(f"  Modelo guardado en: {model_path}")

    except Exception as e:
        log(f"[ERROR] ERROR en el pipeline: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
