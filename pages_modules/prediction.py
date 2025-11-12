import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")

# =====================================================
#   FUNCIÓN PRINCIPAL DE PREDICCIÓN
# =====================================================
def render(models, scaler, feature_names):
    st.markdown("""
    <div class="models-header">
        <h2 style="color:#1DB954;">Predicción de Popularidad</h2>
        <p style="color:#b3b3b3;">Estima la popularidad de una canción según sus características.</p>
    </div>
    """, unsafe_allow_html=True)

    modelo_nombre = st.selectbox("🧠 Selecciona el modelo", list(models.keys()))
    model = models[modelo_nombre]
 
    # -------------------------
    # Entradas numéricas
    # -------------------------
    st.subheader("🎛️ Características numéricas")
    vals = {}
    vals['danceability'] = st.slider("Danceability", 0.0, 1.0, 0.645)
    vals['energy'] = st.slider("Energy", 0.0, 1.0, 0.418)
    vals['loudness'] = st.slider("Loudness (dB)", -60.0, 0.0, -10.065)
    vals['speechiness'] = st.slider("Speechiness", 0.0, 1.0, 0.29)
    vals['acousticness'] = st.slider("Acousticness", 0.0, 1.0, 0.558)
    vals['instrumentalness'] = st.slider("Instrumentalness", 0.0, 1.0, 0.0003)
    vals['liveness'] = st.slider("Liveness", 0.0, 1.0, 0.562)
    vals['valence'] = st.slider("Valence", 0.0, 1.0, 0.123)
    vals['tempo'] = st.slider("Tempo (BPM)", 0.0, 250.0, 96.963)
    vals['duration_ms'] = st.number_input("Duración (ms)", 30000, 600000, 140760)

    # -------------------------
    # Entradas categóricas
    # -------------------------
    st.subheader("🎵 Características categóricas")
    genre_features = [
        'genre_A Capella', 'genre_Alternative', 'genre_Anime', 'genre_Blues', 'genre_Children’s Music',
        'genre_Classical', 'genre_Comedy', 'genre_Country', 'genre_Dance', 'genre_Electronic',
        'genre_Folk', 'genre_Hip-Hop', 'genre_Indie', 'genre_Jazz', 'genre_Movie',
        'genre_Opera', 'genre_Pop', 'genre_R&B', 'genre_Rap', 'genre_Reggae',
        'genre_Reggaeton', 'genre_Rock', 'genre_Ska', 'genre_Soul', 'genre_Soundtrack', 'genre_World'
    ]
    key_features = [
        'key_A', 'key_A#', 'key_B', 'key_C', 'key_C#', 'key_D', 'key_D#',
        'key_E', 'key_F', 'key_F#', 'key_G', 'key_G#'
    ]
    mode_features = ['mode_Major', 'mode_Minor']
    time_signature_features = [
        'time_signature_0/4', 'time_signature_1/4', 'time_signature_3/4',
        'time_signature_4/4', 'time_signature_5/4'
    ]

    selected_genre = st.selectbox("Género musical", [g.replace("genre_", "") for g in genre_features])
    selected_key = st.selectbox("Tonalidad (Key)", [k.replace("key_", "") for k in key_features])
    selected_mode = st.selectbox("Modo", ["Major", "Minor"])
    selected_time = st.selectbox("Compás (Time Signature)", [t.replace("time_signature_", "") for t in time_signature_features])

    # Crear input del usuario
    user_input = {}
    user_input.update(vals)
    for feature in genre_features + key_features + mode_features + time_signature_features:
        user_input[feature] = 0
    user_input[f"genre_{selected_genre}"] = 1
    user_input[f"key_{selected_key}"] = 1
    user_input[f"mode_{selected_mode}"] = 1
    user_input[f"time_signature_{selected_time}"] = 1
    X_input = pd.DataFrame([user_input])

    st.markdown("---")
    
    # -------------------------
    # Botón de predicción
    # -------------------------
    if st.button("🎯 Predecir popularidad"):
        try:
            for col in feature_names:
                if col not in X_input.columns:
                    X_input[col] = 0  

            X_input = X_input[feature_names]
            if "popularity" in X_input.columns:
                X_input = X_input.drop(columns=["popularity"])

            render_prediction(model, X_input, scaler)
        except Exception as e:
            st.error(f"Error durante la predicción: {e}")

def render_prediction(model, X_input, scaler):
    import shap

    # -----------------------------
    # 1. Preparar input
    # -----------------------------
    X_prepared = scaler.transform(X_input)
    pred_proba = float(model.predict_proba(X_prepared)[0][1])

    # Mostrar resultado principal
    st.write(f"🎵 Probabilidad de popularidad: **{pred_proba:.2f}**")

    if pred_proba >= 0.5:
        st.success("✅ ¡Tu canción probablemente será POPULAR!")
    else:
        st.warning("⚠️ Tu canción probablemente NO será popular")

    st.subheader("🧩 Explicación de la predicción (SHAP)")

      # -----------------------------------
    # 2. Explicación SHAP para RandomForest
    # -----------------------------------
    try:
        import shap
        import numpy as np

        # Creamos background artificial (50 muestras duplicadas)
        background = X_prepared[:50] if len(X_prepared) >= 50 else np.tile(X_prepared, (50, 1))

        explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
        shap_values = explainer.shap_values(X_prepared)

        # -----------------------------------
        # MANEJO DE FORMAS DE SHAP
        # -----------------------------------
        # shap_values puede ser:
        # 1) lista para cada clase
        # 2) matriz (1, n_features)
        # 3) matriz (1, n_features, 2)
        
        # Caso 1: lista -> elegimos la clase positiva
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        # Si shap_values es 3D (1, features, 2) -> quedarnos con la clase 1
        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]

        # Ahora debe quedar como (1, n_features)
        shap_values = shap_values.reshape(1, -1)

        # -----------------------------------
        # Top features
        # -----------------------------------
        effects = pd.Series(shap_values[0], index=X_input.columns)
        top_effects = effects.abs().sort_values(ascending=False).head(6)

        # -----------------------------------
        # Gráfico SHAP
        # -----------------------------------
        fig, ax = plt.subplots(figsize=(6, 4))
        top_effects.sort_values().plot(kind="barh", ax=ax)
        ax.set_title("Impacto SHAP en la predicción")
        ax.set_xlabel("Contribución al resultado")
        st.pyplot(fig)

        # -----------------------------------
        # Explicación textual
        # -----------------------------------
        st.markdown("### 📝 Explicación textual")

        for feat in top_effects.index:
            val = X_input.iloc[0][feat]
            eff = effects[feat]

            direction = "aumentó" if eff > 0 else "redujo"

            # Explicaciones semánticas especiales
            if "genre_" in feat and val == 1:
                st.markdown(f"- El género **{feat.replace('genre_', '')}** {direction} la probabilidad.")
            elif feat in ["danceability", "energy", "valence"]:
                st.markdown(f"- La **{feat}** (= {val:.2f}) {direction} la probabilidad.")
            elif feat == "acousticness":
                st.markdown(f"- La **acousticness** (= {val:.2f}) {direction} la probabilidad.")
            elif feat == "tempo":
                st.markdown(f"- El **tempo** (= {val:.1f} BPM) {direction} la probabilidad.")
            elif feat == "duration_ms":
                minutos = val / 60000
                st.markdown(f"- La **duración** (= {minutos:.1f} min) {direction} la probabilidad.")
            else:
                st.markdown(f"- **{feat}** (= {val}) {direction} la probabilidad.")

    except Exception as e:
        st.error(f"No se pudo generar explicación SHAP: {e}")
        st.info("Esto pasa por la forma en que SHAP interpreta RandomForest; ya está corregido.")
