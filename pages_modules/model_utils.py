import os
import joblib
import pickle
import streamlit as st
import warnings
import xgboost as xgb  # 👈 necesario para leer el .json

warnings.filterwarnings("ignore")

MODEL_1 = MODEL_2 = MODEL_3 = None
X_train = y_train = None
scaler = None
feature_names = None


def get_models_dir():
    current_dir = os.path.dirname(__file__)
    project_root = os.path.dirname(current_dir)
    return os.path.join(project_root, "models")


MODELS_DIR = get_models_dir()

MODEL_1_PATH = os.path.join(MODELS_DIR, "logistic_regression_model.pkl")
MODEL_2_JSON_PATH = os.path.join(MODELS_DIR, "xgb_model.json")  # 👈 usamos el JSON
MODEL_3_PATH = os.path.join(MODELS_DIR, "random_forest_model.pkl")

SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
FEATURE_NAMES_PATH = os.path.join(MODELS_DIR, "feature_names.pkl")
XTRAIN_PATH = os.path.join(MODELS_DIR, "X_test.pkl")
YTRAIN_PATH = os.path.join(MODELS_DIR, "y_test.pkl")


def safe_load(path, description="archivo"):
    """Carga segura para archivos .pkl o .joblib"""
    if not os.path.exists(path):
        st.warning(f"⚠️ {description} no encontrado en {path}")
        return None
    try:
        obj = joblib.load(path)
        st.success(f"✅ {description} cargado correctamente (joblib)")
        return obj
    except Exception:
        try:
            with open(path, "rb") as f:
                obj = pickle.load(f)
                st.success(f"✅ {description} cargado correctamente (pickle)")
                return obj
        except Exception as e2:
            st.error(f"❌ Error cargando {description}: {e2}")
            return None


def load_all():
    """Carga todos los modelos y objetos auxiliares"""
    global MODEL_1, MODEL_2, MODEL_3, scaler, feature_names, X_train, y_train

    st.write("### 🧠 Cargando modelos...")

    # Modelo 1: Logistic Regression (.pkl)
    MODEL_1 = safe_load(MODEL_1_PATH, "Modelo 1 (Logistic Regression)")

    # Modelo 2: XGBoost (.json) ✅
    if os.path.exists(MODEL_2_JSON_PATH):
        try:
            xgb_model = xgb.XGBClassifier()
            xgb_model.load_model(MODEL_2_JSON_PATH)
            MODEL_2 = xgb_model
            st.success("✅ Modelo 2 (XGBoost) cargado correctamente desde JSON")
        except Exception as e:
            st.error(f"❌ Error cargando Modelo 2 (XGBoost JSON): {e}")
            MODEL_2 = None
    else:
        st.warning(f"⚠️ Modelo 2 (XGBoost JSON) no encontrado en {MODEL_2_JSON_PATH}")
        MODEL_2 = None

    # Modelo 3: Random Forest (.pkl)
    MODEL_3 = safe_load(MODEL_3_PATH, "Modelo 3 (Random Forest)")

    st.write("### ⚙️ Cargando scaler y datos de entrenamiento...")
    scaler = safe_load(SCALER_PATH, "Scaler")
    feature_names = safe_load(FEATURE_NAMES_PATH, "Feature Names")
    X_train = None
    y_train = None

    # --- Diagnóstico adicional ---
    st.write("### 🧾 Diagnóstico final:")
    st.write({
        "Logistic Regression": type(MODEL_1),
        "XGBoost": type(MODEL_2),
        "Random Forest": type(MODEL_3),
        "Scaler": type(scaler),
        "Feature Names": type(feature_names),
        "X_train": type(X_train),
        "y_train": type(y_train)
    })


def get_models():
    """Devuelve los modelos cargados en un diccionario"""
    return {
        "Logistic Regression": MODEL_1,
        "XGBoost": MODEL_2,
        "Random Forest": MODEL_3,
    }


def get_training_data():
    return X_train, y_train


def get_scaler_and_features():
    return scaler, feature_names
