"""
Pruebas de calidad del modelo de Machine Learning.
"""
import joblib
from pathlib import Path
import pytest


def test_model_has_predict_method():
    """Verifica que el modelo cargado tiene el método predict."""
    # Ruta al modelo guardado
    model_path = Path(__file__).parent.parent / "src" / "models" / "lgbm_flight_delay.pkl"
    
    # Verificar que el archivo del modelo existe
    assert model_path.exists(), f"Modelo no encontrado: {model_path}"
    
    # Cargar el modelo con joblib
    model = joblib.load(model_path)
    
    # Verificar que el objeto cargado tiene el método predict (propio de estimadores sklearn/lightgbm)
    assert hasattr(model, "predict"), "El modelo cargado no tiene el método 'predict'"
    
    # Verificar que también tiene predict_proba (para probabilidades)
    assert hasattr(model, "predict_proba"), "El modelo cargado no tiene el método 'predict_proba'"
    
    # Verificar que es una instancia de LGBMClassifier (opcional, para asegurar tipo correcto)
    assert model.__class__.__name__ == "LGBMClassifier", f"Modelo no es LGBMClassifier, es {model.__class__.__name__}"
    
    # Verificar que el modelo tiene atributos básicos de entrenamiento
    assert hasattr(model, "n_features_in_"), "El modelo no tiene atributo n_features_in_ (posiblemente no entrenado)"
    
    # Verificar que tiene los parámetros configurados
    assert hasattr(model, "scale_pos_weight"), "El modelo no tiene parámetro scale_pos_weight"
    assert model.scale_pos_weight > 0, f"scale_pos_weight inválido: {model.scale_pos_weight}"
    
    print(f"✅ Test de modelo pasado: {model.__class__.__name__} cargado correctamente")
    print(f"   - Características de entrada: {model.n_features_in_}")
    print(f"   - scale_pos_weight: {model.scale_pos_weight:.2f}")
    print(f"   - Número de árboles: {model.n_estimators}")
