import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc
import joblib
import warnings
warnings.filterwarnings("ignore")

def render(models, X_train, y_train):
    st.markdown("""
    <div class="models-header" style="text-align:center;">
        <h2 style="color:#1DB954; margin-bottom:10px;">Modelos de Machine Learning</h2>
        <p style="color:#b3b3b3; font-size:1.1em;">
            A continuación se muestran los tres modelos entrenados para predecir la popularidad de canciones.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Validar que tenemos datos de entrenamiento/test
    if X_train is None or y_train is None:
        st.warning("⚠️ No se pudieron cargar los datos de entrenamiento/test desde HuggingFace.")
        st.info("📝 Los modelos están disponibles para hacer predicciones en la sección 'Predicción', pero no se pueden mostrar métricas de evaluación sin datos de entrenamiento/test.")
        
        # Mostrar información básica de los modelos
        st.markdown("### 📊 Modelos Disponibles")
        for name, model in models.items():
            if model is not None:
                st.success(f"✅ **{name}**: Modelo cargado correctamente")
            else:
                st.error(f"❌ **{name}**: No se pudo cargar")
        return

    # Validar que los modelos no sean None
    valid_models = {name: model for name, model in models.items() if model is not None}
    
    if not valid_models:
        st.error("❌ No hay modelos válidos cargados.")
        return

    metrics = {}
    for name, model in valid_models.items():
        try:
            y_pred = model.predict(X_train)
            y_proba = safe_predict_proba(model, X_train)

            # Calcular curva ROC si hay probabilidades
            if y_proba is not None:
                fpr, tpr, _ = roc_curve(y_train, y_proba)
                roc_auc = auc(fpr, tpr)
            else:
                fpr, tpr, roc_auc = None, None, None

            metrics[name] = {
                "AUC": round(roc_auc, 3) if roc_auc else "N/A"
            }
        except Exception as e:
            st.error(f"❌ Error evaluando modelo {name}: {e}")
            metrics[name] = {"AUC": "Error"}

    # --- Crear pestañas ---
    model_tabs = st.tabs(list(valid_models.keys()))

    for name, tab in zip(valid_models.keys(), model_tabs):
        with tab:
            st.markdown(f"### 📘 {name}")
            st.write(f"**AUC:** {metrics[name]['AUC']}")

            # Dibujar curva ROC si se pudo calcular
            model = valid_models[name]
            if hasattr(model, "predict_proba") and metrics[name]['AUC'] != "Error":
                try:
                    y_proba = model.predict_proba(X_train)[:, 1]
                    fpr, tpr, _ = roc_curve(y_train, y_proba)
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
                except Exception as e:
                    st.error(f"Error generando curva ROC: {e}")
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

def safe_predict_proba(model, X):
    """Devuelve predicciones tipo probabilidad si el modelo las soporta."""
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)
            if proba.ndim == 2 and proba.shape[1] >= 2:
                return proba[:, 1]
        except Exception:
            pass

    # XGBoost suele tener predict() como probabilidad ya escalada
    try:
        raw = model.predict(X)
        # Si son probabilidades, están entre 0 y 1
        if raw.ndim == 1 and raw.min() >= 0 and raw.max() <= 1:
            return raw
    except:
        pass

    return None
