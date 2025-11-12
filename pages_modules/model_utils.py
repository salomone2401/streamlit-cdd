import os
import joblib
import pickle
import streamlit as st
import warnings

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
MODEL_2_PATH = os.path.join(MODELS_DIR, "xgboost_model.pkl")
MODEL_3_PATH = os.path.join(MODELS_DIR, "random_forest_model.pkl")

SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
FEATURE_NAMES_PATH = os.path.join(MODELS_DIR, "feature_names.pkl")
XTRAIN_PATH = os.path.join(MODELS_DIR, "X_test.pkl")
YTRAIN_PATH = os.path.join(MODELS_DIR, "y_test.pkl")


def safe_load(path, description="archivo"):
    if not os.path.exists(path):
        return None
    try:
        obj = joblib.load(path)
        return obj
    except Exception as e1:
        try:
            with open(path, "rb") as f:
                obj = pickle.load(f)
                return obj
        except Exception as e2:
            st.error(f"❌ Error cargando {description}: {e2}")
            return None


def load_all():
    global MODEL_1, MODEL_2, MODEL_3, scaler, feature_names, X_train, y_train

    MODEL_1 = safe_load(MODEL_1_PATH, "Modelo 1 (Logistic Regression)")
    MODEL_2 = safe_load(MODEL_2_PATH, "Modelo 2 (XGBoost)")
    MODEL_3 = safe_load(MODEL_3_PATH, "Modelo 3 (Random Forest)")

    scaler = safe_load(SCALER_PATH, "Scaler")
    feature_names = safe_load(FEATURE_NAMES_PATH, "Feature Names")
    X_train = safe_load(XTRAIN_PATH, "X_train")
    y_train = safe_load(YTRAIN_PATH, "y_train")

    # --- Diagnóstico adicional ---
 

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