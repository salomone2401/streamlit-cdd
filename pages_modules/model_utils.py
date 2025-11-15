import os
import joblib
import pickle
import streamlit as st
import warnings

from huggingface_hub import hf_hub_download, login

warnings.filterwarnings("ignore")

if "HF_TOKEN" in st.secrets:
    login(st.secrets["HF_TOKEN"])

# ==============================
# 📌 CONFIGURACIÓN HF
# ==============================
HF_REPO = "gilmagali14/spotify-popularity-models"

MODEL_1 = MODEL_2 = MODEL_3 = None
X_train = y_train = None
scaler = None
feature_names = None


def get_models_dir():
    current_dir = os.path.dirname(__file__)
    project_root = os.path.dirname(current_dir)
    return os.path.join(project_root, "models")


MODELS_DIR = get_models_dir()

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




def load_from_hf(filename, description):
    """Descarga un archivo del repo de HuggingFace."""
    try:
        path = hf_hub_download(repo_id=HF_REPO, filename=filename)
        return path
    except Exception as e:
        st.warning(f"⚠️ No se pudo descargar {description}: {str(e)[:100]}...")
        return None


def load_all():
    global MODEL_1, MODEL_2, MODEL_3, scaler, feature_names, X_train, y_train

    MODEL_1 = MODEL_2 = MODEL_3 = None
    scaler = feature_names = X_train = y_train = None

    # Cargar modelos desde HuggingFace
    model_1_path = load_from_hf("logistic_regression_model.pkl", "Modelo 1 (Logistic Regression)")
    MODEL_1 = safe_load(model_1_path, "Modelo 1 (Logistic Regression)") if model_1_path else None

    model_2_path = load_from_hf("xgboost_model.pkl", "Modelo 2 (XGBoost)")
    MODEL_2 = safe_load(model_2_path, "Modelo 2 (XGBoost)") if model_2_path else None

    model_3_path = load_from_hf("random_forest_model.pkl", "Modelo 3 (Random Forest)")
    MODEL_3 = safe_load(model_3_path, "Modelo 3 (Random Forest)") if model_3_path else None

    # Cargar scaler y feature names desde HuggingFace
    scaler_path = load_from_hf("scaler.pkl", "Scaler")
    scaler = safe_load(scaler_path, "Scaler") if scaler_path else None
    
    feature_names_path = load_from_hf("feature_names.pkl", "Feature Names")
    feature_names = safe_load(feature_names_path, "Feature Names") if feature_names_path else None
    
    # Cargar datos de entrenamiento/test desde archivos LOCALES
    xtrain_path = load_from_hf("X_test.pkl", "x tests")
    X_train = safe_load(xtrain_path, "x tests Names") if xtrain_path else None
    ytrain_path = load_from_hf("y_test.pkl", "y tests")

    y_train = safe_load(ytrain_path, "y tests Names") if ytrain_path else None

    # Diagnóstico de carga
    loaded_models = sum([1 for m in [MODEL_1, MODEL_2, MODEL_3] if m is not None])
    loaded_components = sum([1 for c in [scaler, feature_names, X_train, y_train] if c is not None])
    
    if loaded_models > 0 or loaded_components > 0:
        st.success(f"✅ Carga completada: {loaded_models}/3 modelos, {loaded_components}/4 componentes")
        st.info(f"📁 Modelos desde HuggingFace, datos desde archivos locales")
    else:
        st.error("❌ No se pudo cargar ningún modelo o componente")
        
    # Mostrar detalles de carga
    with st.expander("🔍 Detalles de carga"):
        st.write("**Modelos (desde HuggingFace):**")
        st.write(f"- Logistic Regression: {'✅' if MODEL_1 else '❌'}")
        st.write(f"- XGBoost: {'✅' if MODEL_2 else '❌'}")
        st.write(f"- Random Forest: {'✅' if MODEL_3 else '❌'}")
        
        st.write("**Componentes (desde HuggingFace):**")
        st.write(f"- Scaler: {'✅' if scaler else '❌'}")
        st.write(f"- Feature Names: {'✅' if feature_names else '❌'}")
        
        st.write("**Datos (desde archivos locales):**")
        st.write(f"- X_train: {'✅' if X_train is not None else '❌'}")
        st.write(f"- y_train: {'✅' if y_train is not None else '❌'}")
        
        if X_train is not None:
            st.write(f"- Forma de X_train: {X_train.shape}")
        if y_train is not None:
            st.write(f"- Forma de y_train: {y_train.shape}")
        


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