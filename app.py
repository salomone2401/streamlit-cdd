import streamlit as st
import os
import sys
import warnings
import traceback, sys, time

st.set_page_config(
    page_title="Proyecto Popularidad",
    page_icon="🎧",
    layout="wide"
)

def log(msg):
    print(f"[DEBUG] {msg}", file=sys.stderr)
    sys.stderr.flush()

pages_modules_path = os.path.join(os.path.dirname(__file__), 'pages_modules')
if pages_modules_path not in sys.path:
    sys.path.insert(0, pages_modules_path)

utils_path = os.path.join(os.path.dirname(__file__), 'utils')
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from pages_modules import eda, machine_learning, prediction, model_utils

warnings.filterwarnings("ignore")

with open("utils/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ===========================
# Load Models Only Once
# ===========================

if "models_loaded" not in st.session_state:
    with st.spinner("Cargando modelos por única vez..."):
        model_utils.load_all()
        st.session_state["models"] = model_utils.get_models()
        st.session_state["scaler"], st.session_state["feature_names"] = model_utils.get_scaler_and_features()
        st.session_state["X_train"], st.session_state["y_train"] = model_utils.get_training_data()
        st.session_state["models_loaded"] = True

models_dict = st.session_state["models"]
scaler = st.session_state["scaler"]
feature_names = st.session_state["feature_names"]
X_train = st.session_state["X_train"]
y_train = st.session_state["y_train"]


def safe_render(func, name="Sección"):
    try:
        func()
    except Exception as e:
        st.error(f"❌ Error en {name}: {e}")
        st.code(traceback.format_exc())
        print(traceback.format_exc(), file=sys.stderr)
        st.stop()


opcion = st.sidebar.radio(
    label="Selecciona una sección",  
    options=["Introducción", "EDA", "Machine Learning Models", "Predicción"],
    label_visibility="hidden"
)

if opcion == "Introducción":
    st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
    st.image("assets/logo.png", width=140)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:right">
        <h1 style="color:#1DB954;">Popularidad de Canciones</h1>
        <p style="color:#b3b3b3;">Proyecto de Machine Learning — UTN FRM</p>
    </div>
    <hr style="border: 1px solid #1DB954;">
    <h2 style="color:#1DB954;">¿Qué vamos a predecir?</h2>
    <p>En este proyecto desarrollamos un modelo de <b>Machine Learning</b> capaz de
    predecir si una canción es popular o no (<code>spotify_artist_popularity</code>), 
    utilizando tanto sus características musicales como la información del artista.</p>
    <ul>
        <li>🎧 Energía, bailabilidad, tempo y duración de la canción.</li>
        <li>🎤 Popularidad y número de seguidores del artista.</li>
        <li>🎶 Presencia de elementos acústicos o electrónicos.</li>
    </ul>
    <p>Se trata de un problema de <b>clasificación binaria</b>, 
    donde el objetivo es estimar si la canción es Popular o No Popular.</p>
    <hr>
    <p style="text-align:center; color:#b3b3b3;">
    Proyecto desarrollado por: 
    <b style="color:#1DB954;">Magalí Gil,</b> 
    <b style="color:#1DB954;">Tito Vaieretti</b> y
    <b style="color:#1DB954;">Ana Paula Salomone</b>
    </p>
    """, unsafe_allow_html=True)
elif opcion == "EDA":
    safe_render(lambda: eda.render(), name="EDA")

elif opcion == "Machine Learning Models":
    safe_render(lambda: machine_learning.render(models_dict, X_train, y_train), name="Machine Learning Models")

elif opcion == "Predicción":
    safe_render(lambda: prediction.render(models_dict, scaler, feature_names), name="Predicción")