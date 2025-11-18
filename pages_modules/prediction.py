import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression  # <--- Necesario para 'isinstance'
from xgboost import XGBClassifier                   # <--- Necesario para 'isinstance'
from sklearn.ensemble import RandomForestClassifier # <--- Necesario para 'isinstance'
import warnings
warnings.filterwarnings("ignore")


# 1. ACEPTA X_train EN LA FIRMA DE 'render'
def render(models, scaler, feature_names, X_train):
    st.markdown("""
    <div class="models-header">
        <h2 style="color:#1DB954;">Predicción de Popularidad</h2>
        <p style="color:#b3b3b3;">Estima la popularidad de una canción según sus características.</p>
    </div>
    """, unsafe_allow_html=True)

    modelo_nombre = st.selectbox("🧠 Selecciona el modelo", list(models.keys()))
    model = models[modelo_nombre]
 
    # ... (todo tu código de sliders y selectbox no cambia) ...
    
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

            # 2. PASA X_train A 'render_prediction'
            render_prediction(model, X_input, scaler, X_train)
        except Exception as e:
            st.error(f"Error durante la predicción: {e}")


# 3. ACEPTA X_train EN LA FIRMA DE 'render_prediction'
def render_prediction(model, X_input, scaler, X_train):
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

    try:
        # Preparamos los datos de fondo (escalados)
        X_train_scaled = scaler.transform(X_train)
        
        # Valor SHAP final (plano)
        shap_values_flat = None

        # -----------------------------------
        # 4. LÓGICA DE SELECCIÓN DE EXPLAINER
        # -----------------------------------
        
        # CASO 1: Modelos de Árbol
        if isinstance(model, (RandomForestClassifier, XGBClassifier)):
            explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
            shap_values = explainer.shap_values(X_prepared)

            # Manejo de formas de SHAP
            if isinstance(shap_values, list):
                shap_values = shap_values[1] # Clase positiva
            if shap_values.ndim == 3:
                shap_values = shap_values[:, :, 1]
            
            shap_values_flat = shap_values.reshape(1, -1)

        # CASO 2: Modelos Lineales
        elif isinstance(model, LogisticRegression):
            
            # Usamos un resumen de X_train (p.ej. 100 muestras) como fondo
            background = shap.sample(X_train_scaled, 100) 
            
            explainer = shap.LinearExplainer(model, background)
            shap_values = explainer.shap_values(X_prepared)
            
            shap_values_flat = shap_values.reshape(1, -1)
        
        else:
            st.warning(f"Tipo de modelo {type(model).__name__} aún no soportado por SHAP en esta app.")
            return

        # -----------------------------------
        # Top features
        # -----------------------------------
        effects = pd.Series(shap_values_flat[0], index=X_input.columns)
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
        # 5. MENSAJE DE ERROR CORREGIDO
        st.info("Ocurrió un error al calcular los valores SHAP para este modelo específico.")