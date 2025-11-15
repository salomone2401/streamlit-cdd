import os
import pickle
import joblib
import streamlit as st
import warnings
import xgboost as xgb
import numpy as np
import pandas as pd

from huggingface_hub import hf_hub_download, login

warnings.filterwarnings("ignore")

if "HF_TOKEN" in st.secrets:
    login(st.secrets["HF_TOKEN"])

# ==============================
# 📌 CONFIGURACIÓN HF
# ==============================
HF_REPO = "gilmagali14/spotify-popularity-models"    # ← CAMBIAR ESTO


# ==============================
# 📌 VARIABLES GLOBALES
# ==============================
MODEL_1 = MODEL_2 = MODEL_3 = None
scaler = None
feature_names = None
X_train = None
y_train = None


# ==============================
# 📌 FUNCIÓN SEGURA PARA CARGAR ARCHIVOS
# ==============================
def safe_model_load(path, description="archivo", model_type="pickle"):
    """Carga segura de archivos de modelos con múltiples métodos."""
    if not path or not os.path.exists(path):
        st.error(f"❌ Archivo no encontrado: {description}")
        return None
    
    # Lista de métodos de carga a intentar
    loading_methods = []
    
    if model_type == "xgboost":
        loading_methods = [
            ("XGBoost JSON", lambda p: load_xgboost_model(p)),
            ("XGBoost Pickle", lambda p: load_with_pickle(p)),
            ("Joblib", lambda p: joblib.load(p))
        ]
    else:
        loading_methods = [
            ("Pickle", lambda p: load_with_pickle(p)),
            ("Joblib", lambda p: joblib.load(p)),
            ("Pickle Protocol 2", lambda p: load_with_pickle(p, protocol=2))
        ]
    
    # Intentar cada método de carga
    for method_name, load_func in loading_methods:
        try:
            result = load_func(path)
            if result is not None:
                return result
        except Exception as e:
            continue
    
    st.error(f"❌ No se pudo cargar {description} con ningún método")
    return None


def load_with_pickle(path, protocol=None):
    """Carga archivo con pickle, con opción de protocolo específico."""
    with open(path, "rb") as f:
        if protocol:
            return pickle.load(f)
        else:
            return pickle.load(f)


def load_xgboost_model(path):
    """Carga modelo XGBoost desde archivo JSON o binario."""
    model = xgb.XGBClassifier()
    
    # Intentar cargar como JSON primero
    try:
        model.load_model(path)
        return model
    except Exception:
        # Si falla, intentar como pickle
        raise Exception("XGBoost JSON loading failed")


def load_training_data(path, description="datos"):
    """Carga específica para datos de entrenamiento - versión simplificada."""
    if not path or not os.path.exists(path):
        return None
    
    # Métodos ordenados por probabilidad de éxito
    loading_methods = [
        ("Joblib", lambda p: joblib.load(p)),
        ("NumPy con encoding latin1", lambda p: np.load(p, allow_pickle=True, encoding='latin1')),
        ("Pandas Pickle", lambda p: pd.read_pickle(p)),
        ("Pickle con encoding latin1", lambda p: load_with_pickle_encoding(p, 'latin1'))
    ]
    
    # Intentar cada método de carga
    for method_name, load_func in loading_methods:
        try:
            result = load_func(path)
            if result is not None:
                st.success(f"✅ {description} cargado con {method_name}")
                return result
        except Exception:
            continue
    
    return None


def load_with_pickle_encoding(path, encoding):
    """Carga archivo con pickle usando encoding específico."""
    with open(path, "rb") as f:
        return pickle.load(f, encoding=encoding)


def load_from_csv_fallback():
    """Intenta cargar datos desde CSV local como fallback."""
    try:
        # Buscar archivos CSV en la carpeta data
        csv_files = ["data/data-limpio.csv"]
        
        for csv_file in csv_files:
            if os.path.exists(csv_file):
                st.info(f"📊 Intentando cargar datos desde {csv_file}")
                df = pd.read_csv(csv_file)
                
                # Asumir que la última columna es el target
                if len(df.columns) > 1:
                    X = df.iloc[:, :-1]
                    y = df.iloc[:, -1]
                    
                    st.success(f"✅ Datos cargados desde CSV: {X.shape[0]} filas, {X.shape[1]} características")
                    return X, y
                    
        return None, None
    except Exception as e:
        st.warning(f"⚠️ Error cargando desde CSV: {e}")
        return None, None


# ==============================
# 📌 DESCARGA DE ARCHIVOS DESDE HF
# ==============================
def load_from_hf(filename, description):
    """Descarga un archivo del repo de HuggingFace."""
    try:
        path = hf_hub_download(repo_id=HF_REPO, filename=filename)
        return path
    except Exception as e:
        # No mostrar error para archivos opcionales (como diferentes formatos de XGBoost)
        if "xgb" in filename.lower() or "json" in filename.lower():
            st.info(f"ℹ️ {filename} no disponible, intentando alternativas...")
        else:
            st.warning(f"⚠️ No se pudo descargar {description}: {str(e)[:100]}...")
        return None


# ==============================
# 📌 CARGA PRINCIPAL DE MODELOS
# ==============================
def load_all():
    global MODEL_1, MODEL_2, MODEL_3
    global scaler, feature_names, X_train, y_train

    # ---- Logistic Regression ----
    p = load_from_hf("logistic_regression_model.pkl", "Modelo 1 (LR)")
    MODEL_1 = safe_model_load(p, "Modelo 1 (LR)", "pickle") if p else None

    # ---- XGBoost Model ----
    # Intentar primero con formato JSON, luego PKL
    xgb_files = ["xgb_model.json", "xgboost_model.pkl", "xgboost_model.json"]
    MODEL_2 = None
    
    for filename in xgb_files:
        p = load_from_hf(filename, f"XGBoost ({filename})")
        if p:
            MODEL_2 = safe_model_load(p, f"XGBoost ({filename})", "xgboost")
            if MODEL_2:
                break

    # ---- Random Forest ----
    p = load_from_hf("random_forest_model.pkl", "Modelo 3 (RF)")
    MODEL_3 = safe_model_load(p, "Modelo 3 (RF)", "pickle") if p else None

    # ---- Scaler ----
    p = load_from_hf("scaler.pkl", "Scaler")
    scaler = safe_model_load(p, "Scaler", "pickle") if p else None

    # ---- Feature names ----
    p = load_from_hf("feature_names.pkl", "Feature Names")
    feature_names = safe_model_load(p, "Feature Names", "pickle") if p else None

    # ---- Datos de entrenamiento ----
    st.info("📊 Cargando datos de entrenamiento...")
    
    # Intentar cargar X_test
    p = load_from_hf("X_test.pkl", "X_test")
    X_train = load_training_data(p, "X_test") if p else None
    
    # Intentar cargar y_test
    p = load_from_hf("y_test.pkl", "y_test")
    y_train = load_training_data(p, "y_test") if p else None

    # ---- Fallback: Cargar desde CSV local ----
    if X_train is None or y_train is None:
        st.info("🔄 Intentando cargar datos desde CSV local...")
        X_csv, y_csv = load_from_csv_fallback()
        
        if X_csv is not None and y_csv is not None:
            # Usar solo una muestra pequeña para testing si viene del CSV
            from sklearn.model_selection import train_test_split
            X_train, _, y_train, _ = train_test_split(X_csv, y_csv, test_size=0.8, random_state=42)
            st.success(f"✅ Datos de prueba creados desde CSV: {X_train.shape[0]} muestras")

    # Mostrar resumen de carga
    loaded_models = sum([1 for m in [MODEL_1, MODEL_2, MODEL_3] if m is not None])
    
    # Función auxiliar para verificar si un componente está cargado
    def is_loaded(component):
        if component is None:
            return False
        if hasattr(component, 'empty'):  # DataFrame
            return not component.empty
        return True
    
    loaded_components = sum([1 for c in [scaler, feature_names, X_train, y_train] if is_loaded(c)])
    
    if loaded_models > 0 or loaded_components > 0:
        st.success(f"✅ Carga completada: {loaded_models}/3 modelos, {loaded_components}/4 componentes")
    else:
        st.error("❌ No se pudo cargar ningún modelo o componente")
        
    # Información adicional para debugging
    with st.expander("🔍 Detalles de carga"):
    
        
        # Manejo especial para DataFrames/arrays
        x_status = '✅' if X_train is not None and (not hasattr(X_train, 'empty') or not X_train.empty) else '❌'
        y_status = '✅' if y_train is not None and (not hasattr(y_train, 'empty') or not y_train.empty) else '❌'
        


# ==============================
# 📌 GETTERS
# ==============================
def get_models():
    return {
        "Logistic Regression": MODEL_1,
        "XGBoost": MODEL_2,
        "Random Forest": MODEL_3,
    }


def get_training_data():
    return X_train, y_train


def get_scaler_and_features():
    return scaler, feature_names