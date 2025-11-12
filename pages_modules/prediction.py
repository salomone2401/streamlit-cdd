import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import streamlit as st

import warnings
warnings.filterwarnings("ignore")

def render(models, scaler, feature_names):
    st.markdown("""
    <div class="models-header">
        <h2 style="color:#1DB954;">Predicción de Popularidad</h2>
        <p style="color:#b3b3b3;">Estima la popularidad de una canción según sus características.</p>
    </div>
    """, unsafe_allow_html=True)

    modelo_nombre = st.selectbox("🧠 Selecciona el modelo", list(models.keys()))
    model = models[modelo_nombre]
 
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
    
    if st.button("🎯 Predecir popularidad"):
        try:
            for col in feature_names:
                if col not in X_input.columns:
                    X_input[col] = 0  

            X_input = X_input[feature_names]
            if "popularity" in X_input.columns:
                        X_input = X_input.drop(columns=["popularity"])

            X_prepared = scaler.transform(X_input)

            pred = model.predict(X_prepared)[0]
            print(type(model))
            print(hasattr(model, "predict_proba"))
            if pred == 1:
                st.success("✅ ¡Tu canción probablemente será POPULAR!")
            else:
                st.warning("⚠️ Tu canción probablemente NO será popular")

            st.markdown("### 🧾 Detalle de los valores enviados al modelo")
            st.dataframe(X_input.T)
            render_prediction(model, X_input)
        except Exception as e:
            st.error(f"Error durante la predicción: {e}")

def render_prediction(model, X_input):
    import numpy as np

    # Probabilidad de ser popular
    pred_proba = float(model.predict_proba(X_input)[0][1])
    st.write(f"🎵 Probabilidad de popularidad: **{pred_proba:.2f}**")

    st.subheader("🧩 ¿Por qué el modelo tomó esta decisión?")
    st.write("El gráfico muestra qué características influyeron más en la predicción:")

    try:
        # Convertir a numpy si el modelo no acepta DataFrames
        X_array = X_input.to_numpy()

        # Detectar tipo de modelo (árbol, lineal u otro)
        model_name = type(model).__name__.lower()

        if any(x in model_name for x in ["forest", "tree", "xgb", "lgbm", "boost"]):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_array)
        else:
            explainer = shap.Explainer(model, X_array)
            shap_values = explainer(X_array)

        # Si devuelve lista (modelos binarios)
        if isinstance(shap_values, list):
            shap_array = shap_values[1]
        elif hasattr(shap_values, "values"):
            shap_array = shap_values.values
        else:
            shap_array = shap_values

        # Mostrar gráfico de barras
        fig, ax = plt.subplots()
        shap.summary_plot(shap_array, X_array, feature_names=X_input.columns, plot_type="bar", show=False)
        st.pyplot(fig)

        # Mostrar texto de explicación
        vals = shap_array[0] if len(shap_array.shape) > 1 else shap_array
        top_features = sorted(zip(X_input.columns, vals), key=lambda x: abs(x[1]), reverse=True)[:3]

        st.markdown("**Principales factores que influyeron:**")
        for name, val in top_features:
            direction = "aumentó" if val > 0 else "disminuyó"
            st.write(f"- `{name}` {direction} la probabilidad de ser popular ({val:.3f})")

    except Exception as e:
        st.error(f"No se pudo generar la explicación SHAP: {e}")
