import pickle
import joblib
import streamlit as st
import os
from typing import Any, Optional

import warnings
warnings.filterwarnings("ignore")

MODEL_1 = None
MODEL_2 = None
MODEL_3 = None

def get_models_dir():
    """Obtiene la ruta del directorio de modelos"""
    current_dir = os.path.dirname(__file__)
    project_root = os.path.dirname(current_dir)
    models_dir = os.path.join(project_root, 'models')
    return models_dir

MODELS_DIR = get_models_dir()


MODEL_1_PATH = os.path.join(MODELS_DIR, 'logistic_regression_model.pkl')
MODEL_2_PATH = os.path.join(MODELS_DIR, 'xgboost_model.pkl')
MODEL_3_PATH = os.path.join(MODELS_DIR, 'random_forest_model.pkl')


scaler = joblib.load("models/scaler.pkl")
feature_names = joblib.load("models/feature_names.pkl")

def load_model_1(model_path: str = None, model_object: Any = None):
    global MODEL_1
    
    try:
        if model_object is not None:
            MODEL_1 = model_object
        elif model_path:
            try:
                MODEL_1 = joblib.load(model_path)
            except:
                with open(model_path, 'rb') as f:
                    MODEL_1 = pickle.load(f)
        else:
            st.warning("Debes proporcionar model_path o model_object")
            return None
        
        st.success("Modelo 1 cargado exitosamente")
        return MODEL_1
    except Exception as e:
        st.error(f"Error al cargar Modelo 1: {str(e)}")
        return None


def load_model_2(model_path: str = None, model_object: Any = None):
    global MODEL_2
    
    try:
        if model_object is not None:
            MODEL_2 = model_object
        elif model_path:
            try:
                MODEL_2 = joblib.load(model_path)
            except:
                with open(model_path, 'rb') as f:
                    MODEL_2 = pickle.load(f)
        else:
            st.warning("Debes proporcionar model_path o model_object")
            return None
        
        st.success("Modelo 2 cargado exitosamente")
        return MODEL_2
    except Exception as e:
        st.error(f"Error al cargar Modelo 2: {str(e)}")
        return None


def load_model_3(model_path: str = None, model_object: Any = None):
    global MODEL_3
    
    try:
        if model_object is not None:
            MODEL_3 = model_object
        elif model_path:
            try:
                MODEL_3 = joblib.load(model_path)
            except:
                with open(model_path, 'rb') as f:
                    MODEL_3 = pickle.load(f)
        else:
            st.warning("Debes proporcionar model_path o model_object")
            return None
        
        st.success("Modelo 3 cargado exitosamente")
        return MODEL_3
    except Exception as e:
        st.error(f"Error al cargar Modelo 3: {str(e)}")
        return None


def get_model_1() -> Optional[Any]:
    return MODEL_1


def get_model_2() -> Optional[Any]:
    return MODEL_2


def get_model_3() -> Optional[Any]:
    """Obtiene el modelo 3 cargado"""
    return MODEL_3

def predict_model_1(X):
    if MODEL_1 is None:
        raise ValueError("Modelo 1 no ha sido cargado")
    return MODEL_1.predict(X)


def predict_model_2(X):
    if MODEL_2 is None:
        raise ValueError("Modelo 2 no ha sido cargado")
    return MODEL_2.predict(X)


def predict_model_3(X):
    if MODEL_3 is None:
        raise ValueError("Modelo 3 no ha sido cargado")
    return MODEL_3.predict(X)


def load_all_models():
    """Carga automáticamente todos los modelos disponibles"""
    global MODEL_1, MODEL_2, MODEL_3
    

    if os.path.exists(MODEL_1_PATH):
        try:
            print(f"Cargando Modelo 1 desde: {MODEL_1_PATH}")
            MODEL_1 = joblib.load(MODEL_1_PATH)
            print("✅ Modelo 1 (Logistic Regression) cargado exitosamente")
        except Exception as e1:
            try:
                print(f"Intentando cargar Modelo 1 con pickle...")
                with open(MODEL_1_PATH, 'rb') as f:
                    MODEL_1 = pickle.load(f)
                print("✅ Modelo 1 (Logistic Regression) cargado exitosamente con pickle")
            except Exception as e2:
                print(f"❌ Error al cargar Modelo 1: {str(e1)} o {str(e2)}")
    else:
        print(f"⚠️ Modelo 1 no encontrado en: {MODEL_1_PATH}")
    
    if os.path.exists(MODEL_2_PATH):
        try:
            print(f"Cargando Modelo 2 desde: {MODEL_2_PATH}")
            MODEL_2 = joblib.load(MODEL_2_PATH)
            print("✅ Modelo 2 (XGBoost) cargado exitosamente")
        except Exception as e1:
            try:
                print(f"Intentando cargar Modelo 2 con pickle...")
                with open(MODEL_2_PATH, 'rb') as f:
                    MODEL_2 = pickle.load(f)
                print("✅ Modelo 2 (XGBoost) cargado exitosamente con pickle")
            except Exception as e2:
                print(f"❌ Error al cargar Modelo 2: {str(e1)} o {str(e2)}")
    else:
        print(f"⚠️ Modelo 2 no encontrado en: {MODEL_2_PATH}")
    
    if MODEL_3_PATH and os.path.exists(MODEL_3_PATH):
        try:
            print(f"Cargando Modelo 3 desde: {MODEL_3_PATH}")
            MODEL_3 = joblib.load(MODEL_3_PATH)
            print("✅ Modelo 3 (Random Forest) cargado exitosamente")
        except Exception as e1:
            try:
                print(f"Intentando cargar Modelo 3 con pickle...")
                with open(MODEL_3_PATH, 'rb') as f:
                    MODEL_3 = pickle.load(f)
                print("✅ Modelo 3 (Random Forest) cargado exitosamente con pickle")
            except Exception as e2:
                print(f"❌ Error al cargar Modelo 3: {str(e2)}")
    else:
        if MODEL_3_PATH:
            print(f"⚠️ Modelo 3 no encontrado en: {MODEL_3_PATH}")
