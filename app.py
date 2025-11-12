import streamlit as st
from pages import eda, machine_learning, prediction

import sys
import os

import warnings
warnings.filterwarnings("ignore")


pages_path = os.path.join(os.path.dirname(__file__), 'pages')
if pages_path not in sys.path:
    sys.path.insert(0, pages_path)

utils_path = os.path.join(os.path.dirname(__file__), 'utils')
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

with open("utils/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.set_page_config(
    page_title="Proyecto Popularidad",
    page_icon="🎧",
    layout="wide"
)

from models import (
    get_model_1, get_model_2, get_model_3,
    scaler, feature_names, load_all_models
)

@st.cache_resource
def load_models_once():
    load_all_models()
    return {
        "Logistic Regression": get_model_1(),
        "XGBoost": get_model_2(),
        "Random Forest": get_model_3(),
    }

models = load_models_once()

opcion = st.sidebar.radio(
    label="Selecciona un modelo",  # ✅ obligatorio: label no vacío
    options= ["Introducción", "EDA", "Machine Learning Models", "Predicción"],
    label_visibility="hidden"  # 👈 oculta visualmente el label si no lo quieres mostrar
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
    <p>En este proyecto desarrollaremos un modelo de <b>Machine Learning</b> capaz de
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
    eda.render()

elif opcion == "Machine Learning Models":
    machine_learning.render(models)

elif opcion == "Predicción":
    prediction.render(models, scaler, feature_names)
