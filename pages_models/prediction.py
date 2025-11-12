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


# =====================================================
#   FUNCIÓN DE PREDICCIÓN + EXPLICACIÓN BASADA EN PESOS
# =====================================================
def render_prediction(model, X_input, scaler):
    X_prepared = scaler.transform(X_input)
    pred_proba = float(model.predict_proba(X_prepared)[0][1])

    # Mostrar resultado
    st.write(f"🎵 Probabilidad de popularidad: **{pred_proba:.2f}**")

    if pred_proba >= 0.5:
        st.success("✅ ¡Tu canción probablemente será POPULAR!")
    else:
        st.warning("⚠️ Tu canción probablemente NO será popular")

    st.subheader("🧩 Explicación de la predicción")

    try:
        # Obtener importancia de las características según tipo de modelo
        if hasattr(model, "coef_"):  # LogisticRegression
            importances = model.coef_[0]
        elif hasattr(model, "feature_importances_"):  # RandomForest o XGBoost
            importances = model.feature_importances_
        else:
            importances = np.zeros(X_input.shape[1])

        # Crear serie de importancia
        feature_importance = pd.Series(importances, index=X_input.columns)
        top_features = feature_importance.abs().sort_values(ascending=False).head(5).index.tolist()

        # Texto explicativo
        explanation = []
        for feat in top_features:
            value = X_input.iloc[0][feat]
            if "genre_" in feat and value == 1:
                explanation.append(f"- El género **{feat.replace('genre_', '')}** tuvo un peso importante en la predicción.")
            elif feat in ["danceability", "energy", "valence"]:
                level = "alta" if value > 0.6 else "baja"
                explanation.append(f"- La característica **{feat}** es {level}, lo que influyó en la {'popularidad' if value > 0.6 else 'falta de popularidad'}.")
            elif feat == "acousticness":
                if value > 0.6:
                    explanation.append("- Alta **acousticness**: el modelo asocia canciones muy acústicas con menor popularidad.")
                else:
                    explanation.append("- Baja **acousticness**: el modelo asocia canciones más eléctricas con mayor popularidad.")
            elif feat == "tempo":
                explanation.append(f"- El **tempo** de {value:.1f} BPM tuvo un impacto moderado.")
            elif feat == "duration_ms":
                mins = value / 60000
                explanation.append(f"- La duración de **{mins:.1f} minutos** influyó ligeramente en la predicción.")

        # Mostrar explicación según el resultado
        if pred_proba >= 0.5:
            st.markdown("🟢 **Tu canción tiene características asociadas con temas exitosos:**")
        else:
            st.markdown("🔴 **Tu canción tiene características asociadas con menor popularidad:**")

        for e in explanation:
            st.markdown(e)

        # Mostrar gráfico simple de las top features
        fig, ax = plt.subplots()
        top_imp = feature_importance.abs().sort_values(ascending=True).tail(5)
        top_imp.plot(kind="barh", ax=ax, color="#1DB954")
        plt.xlabel("Importancia relativa")
        plt.title("Características más influyentes en la predicción")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"No se pudo generar la explicación: {e}")