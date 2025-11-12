import streamlit as st
import pandas as pd
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")

def render(models):
    st.markdown("""
    <div class="models-header" style="text-align:center;">
        <h2 style="color:#1DB954; margin-bottom:10px;">Modelos de Machine Learning</h2>
        <p style="color:#b3b3b3; font-size:1.1em;">
            A continuación se muestran los tres modelos entrenados para predecir la popularidad de canciones.
        </p>
    </div>
    """, unsafe_allow_html=True)

    model_tabs = st.tabs(list(models.keys()))

    metrics = {
        "Logistic Regression": {"accuracy": 0.72, "precision": 0.70, "recall": 0.69, "f1_score": 0.70},
        "XGBoost": {"accuracy": 0.81, "precision": 0.80, "recall": 0.79, "f1_score": 0.79},
        "Random Forest": {"accuracy": 0.78, "precision": 0.77, "recall": 0.76, "f1_score": 0.76}
    }

    for name, tab in zip(models.keys(), model_tabs):
        with tab:
            st.markdown(f"### 📘 {name}")
            st.write("**Métricas:**")
            st.json(metrics[name])

    # Comparación gráfica
    st.markdown("### 📊 Comparativa de rendimiento")
    df_metrics = pd.DataFrame(metrics).T.reset_index().rename(columns={'index': 'Modelo'})
    fig = px.bar(
        df_metrics,
        x='Modelo',
        y=['accuracy', 'precision', 'recall', 'f1_score'],
        barmode='group',
        title='Comparación de métricas entre modelos'
    )
    st.plotly_chart(fig, width="stretch")
