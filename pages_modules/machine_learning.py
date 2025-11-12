import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc
import joblib
import warnings
warnings.filterwarnings("ignore")

def render(models, X_test, y_test):
    st.markdown("""
    <div class="models-header" style="text-align:center;">
        <h2 style="color:#1DB954; margin-bottom:10px;">Modelos de Machine Learning</h2>
        <p style="color:#b3b3b3; font-size:1.1em;">
            A continuación se muestran los tres modelos entrenados para predecir la popularidad de canciones.
        </p>
    </div>
    """, unsafe_allow_html=True)



    metrics = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = None

        # Calcular curva ROC si hay probabilidades
        if y_proba is not None:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
        else:
            fpr, tpr, roc_auc = None, None, None

        metrics[name] = {
            "AUC": round(roc_auc, 3) if roc_auc else "N/A"
        }

    # --- Crear pestañas ---
    model_tabs = st.tabs(list(models.keys()))

    for name, tab in zip(models.keys(), model_tabs):
        with tab:
            st.markdown(f"### 📘 {name}")
            st.write(f"**AUC:** {metrics[name]['AUC']}")

            # Dibujar curva ROC si se pudo calcular
            model = models[name]
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_auc = auc(fpr, tpr)

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    mode='lines',
                    name=f'ROC (AUC = {roc_auc:.3f})',
                    line=dict(color='#1DB954', width=3)
                ))
                fig.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1],
                    mode='lines',
                    name='Random',
                    line=dict(color='gray', dash='dash')
                ))
                fig.update_layout(
                    title=f'Curva ROC - {name}',
                    xaxis_title='Tasa de Falsos Positivos (FPR)',
                    yaxis_title='Tasa de Verdaderos Positivos (TPR)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    legend=dict(x=0.6, y=0.1)
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Este modelo no soporta probabilidades, no se puede generar curva ROC.")

    # --- Comparación general de AUCs ---
    st.markdown("### 📊 Comparativa general de AUC entre modelos")
    df_metrics = pd.DataFrame(metrics).T.reset_index().rename(columns={'index': 'Modelo'})
    fig_comp = px.bar(
        df_metrics,
        x='Modelo',
        y='AUC',
        title='Comparación de AUC entre Modelos',
        text='AUC',
        color='Modelo'
    )
    st.plotly_chart(fig_comp, use_container_width=True)
