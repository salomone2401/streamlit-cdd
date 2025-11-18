import streamlit as st
import pandas as pd
import numpy as _np
import numpy as np
from pages_modules.gemini import gemini_explain

import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression  
from xgboost import XGBClassifier                   
from sklearn.ensemble import RandomForestClassifier
import shap
import warnings
warnings.filterwarnings("ignore")

# Fix NumPy
if not hasattr(_np, "obj2sctype"):
    def _obj2sctype(x):
        return _np.dtype(x).type
    _np.obj2sctype = _obj2sctype



# =====================================================
#  CARGA DE EXPLICADORES CACHEADOS
# =====================================================
@st.cache_resource
def get_explainers(_models, X_train_scaled):
    explainers = {}

    for name, model in _models.items():

        # Modelos de árbol → TreeExplainer
        if hasattr(model, "feature_importances_"):
            explainer = shap.TreeExplainer(
                model,
                feature_perturbation="tree_path_dependent"
            )

        # Modelos lineales → Explainer + Independent Masker
        else:
            masker = shap.maskers.Independent(X_train_scaled)
            explainer = shap.Explainer(model, masker)

        explainers[name] = explainer

    return explainers



# =====================================================
#  UI PRINCIPAL
# =====================================================
def render(models, scaler, feature_names, X_train):

    # Quitar "popularity" si aparece
    clean_features = [f for f in feature_names if f != "popularity"]

    st.markdown("""
    <div class="models-header">
        <h2 style="color:#1DB954;">Predicción de Popularidad</h2>
        <p style="color:#b3b3b3;">Estima la popularidad de una canción según sus características.</p>
    </div>
    """, unsafe_allow_html=True)

    modelo_nombre = st.selectbox("🧠 Selecciona el modelo", list(models.keys()))
    model = models[modelo_nombre]

    # -----------------------------------------
    # INPUTS NUMÉRICOS
    # -----------------------------------------
    st.subheader("🎛️ Características numéricas")
    vals = {}
    vals['danceability']     = st.slider("Danceability", 0.0, 1.0, 0.645)
    vals['energy']           = st.slider("Energy", 0.0, 1.0, 0.418)
    vals['loudness']         = st.slider("Loudness (dB)", -60.0, 0.0, -10.065)
    vals['speechiness']      = st.slider("Speechiness", 0.0, 1.0, 0.29)
    vals['acousticness']     = st.slider("Acousticness", 0.0, 1.0, 0.558)
    vals['instrumentalness'] = st.slider("Instrumentalness", 0.0, 1.0, 0.0003)
    vals['liveness']         = st.slider("Liveness", 0.0, 1.0, 0.562)
    vals['valence']          = st.slider("Valence", 0.0, 1.0, 0.123)
    vals['tempo']            = st.slider("Tempo (BPM)", 0.0, 250.0, 96.963)
    vals['duration_ms']      = st.number_input("Duración (ms)", 30000, 600000, 140760)

    # -----------------------------------------
    # INPUTS CATEGÓRICOS
    # -----------------------------------------
    st.subheader("🎵 Características categóricas")

    genre_features = [
        'genre_A Capella', 'genre_Alternative', 'genre_Anime', 'genre_Blues', 'genre_Children’s Music',
        'genre_Classical', 'genre_Comedy', 'genre_Country', 'genre_Dance', 'genre_Electronic',
        'genre_Folk', 'genre_Hip-Hop', 'genre_Indie', 'genre_Jazz', 'genre_Movie',
        'genre_Opera', 'genre_Pop', 'genre_R&B', 'genre_Rap', 'genre_Reggae',
        'genre_Reggaeton', 'genre_Rock', 'genre_Ska', 'genre_Soul', 'genre_Soundtrack', 'genre_World'
    ]

    key_features = [
        'key_A', 'key_A#', 'key_B', 'key_C', 'key_C#',
        'key_D', 'key_D#', 'key_E', 'key_F', 'key_F#',
        'key_G', 'key_G#'
    ]

    mode_features = ["mode_Major", "mode_Minor"]

    time_signature_features = [
        'time_signature_0/4', 'time_signature_1/4', 'time_signature_3/4',
        'time_signature_4/4', 'time_signature_5/4'
    ]

    selected_genre = st.selectbox("Género musical",
        [g.replace("genre_", "") for g in genre_features]
    )
    selected_key   = st.selectbox("Tonalidad (Key)",
        [k.replace("key_", "") for k in key_features]
    )
    selected_mode  = st.selectbox("Modo", ["Major", "Minor"])
    selected_time  = st.selectbox("Compás (Time Signature)",
        [t.replace("time_signature_", "") for t in time_signature_features]
    )

    # Crear X_input
    X_input = {}
    X_input.update(vals)

    # Inicializar todas en 0
    for f in genre_features + key_features + mode_features + time_signature_features:
        X_input[f] = 0

    # Activar las seleccionadas
    X_input[f"genre_{selected_genre}"] = 1
    X_input[f"key_{selected_key}"] = 1
    X_input[f"mode_{selected_mode}"] = 1
    X_input[f"time_signature_{selected_time}"] = 1

    # Convertir a DataFrame
    X_input = pd.DataFrame([X_input])

    # Asegurar columnas faltantes
    for col in clean_features:
        if col not in X_input.columns:
            X_input[col] = 0

    X_input = X_input[clean_features]

    st.markdown("---")

    # =====================================================
    # BOTÓN DE PREDICCIÓN
    # =====================================================
    if st.button("🎯 Predecir popularidad"):

        st.session_state.pop("friendly_explanation", None)

        X_train_scaled = scaler.transform(X_train)
        explainers = get_explainers(models, X_train_scaled)

        render_prediction(
            model, modelo_nombre, explainers,
            X_input, scaler, X_train_scaled
        )



# =====================================================
#   PREDICCIÓN + SHAP ULTRARRÁPIDO
# =====================================================
def render_prediction(model, modelo_nombre, explainers, X_input, scaler, X_train_scaled):

    # Seguridad extra
    if "popularity" in X_input.columns:
        X_input = X_input.drop(columns=["popularity"])

    X_prepared = scaler.transform(X_input)

    # PREDICCIÓN
    pred_proba = float(model.predict_proba(X_prepared)[0][1])

    st.write(f"🎵 Probabilidad de popularidad: **{pred_proba:.2f}**")
    st.success("✅ ¡Tu canción será popular!") if pred_proba >= 0.5 else st.warning("⚠️ Tu canción NO será popular")

    st.subheader("🧩 Explicación de la predicción (SHAP)")

    try:
        explainer = explainers[modelo_nombre]

        shap_values = explainer(X_prepared)

        values = shap_values.values

        if values.ndim == 3:  # (1, n_features, 2)
            values = values[:, :, 1]  # tomar clase positiva

        values = values[0]  # ahora sí es 1D

        effects = pd.Series(values, index=X_input.columns)


        top_effects = effects.abs().sort_values(ascending=False).head(6)

        # ----------------------------
        # GRÁFICO
        # ----------------------------
        fig, ax = plt.subplots(figsize=(3.5, 2.3))
        top_effects.sort_values().plot(kind="barh", ax=ax, fontsize=8)
        ax.set_title("Impacto SHAP en la predicción", fontsize=10)
        ax.set_xlabel("Contribución al resultado", fontsize=9)
        st.pyplot(fig, use_container_width=False)

        # ----------------------------
        # EXPLICACIÓN NATURAL (Gemini)
        # ----------------------------
        st.markdown("### 📝 Explicación textual")

        if "friendly_explanation" not in st.session_state:
            friendly = gemini_explain(pred_proba, top_effects, effects, X_input)
            st.session_state["friendly_explanation"] = friendly

        st.markdown(st.session_state["friendly_explanation"])

    except Exception as e:
        st.error(f"No se pudo generar explicación SHAP: {e}")
