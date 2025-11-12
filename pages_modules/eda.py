import streamlit as st
import pandas as pd
import altair as alt
import os
import numpy as np

def render():
    st.markdown("""
    <div class="models-header" style="text-align:center;">
        <h2 style="color:#1DB954; margin-bottom:5px;">📊 Análisis Visual Interactivo</h2>
        <p style="color:#b3b3b3; font-size:1.1em;">
            Personaliza los gráficos y explora patrones ocultos en los datos de popularidad de Spotify.
        </p>
    </div>
    """, unsafe_allow_html=True)

    @st.cache_data
    def load_data():
        try:
            data_path = "data/data.csv"
            df = pd.read_csv(data_path)
            return df
        except Exception as e:
            st.error(f"❌ Error al cargar datos: {e}.")
            return pd.DataFrame()

    df = load_data()

    if df.empty:
        return

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()


    tab1, tab2, tab3, tab4 = st.tabs([
        "Distribución",
        "Comparaciones",
        "Correlaciones",
        "Mapa de Densidad"
    ])

    with tab1:
        st.subheader("📈 Distribución de una variable")
        variable = st.selectbox("Selecciona la variable numérica:", numeric_cols, index=0)
        bins = 30
        color = st.color_picker("Color del gráfico", "#1DB954")

        chart1 = alt.Chart(df).mark_bar(color=color).encode(
            alt.X(f"{variable}:Q", bin=alt.Bin(maxbins=bins), title=variable.capitalize()),
            y="count():Q"
        ).properties(width=700, height=400)
        st.altair_chart(chart1, width="stretch")

    with tab2:
        st.subheader("🎵 Comparación por Categoria")
        cat_var = st.selectbox("Categoria (eje X):", categorical_cols, index=0)
        num_var = st.selectbox("Variable numérica (eje Y):", numeric_cols, index=numeric_cols.index("popularity") if "popularity" in numeric_cols else 0)
        top_n = st.slider(f"Mostrar top categorías más frecuentes", 5, 20, 10)
        df_top = df[df[cat_var].isin(df[cat_var].value_counts().index[:top_n])]

        tipo = st.radio("Tipo de gráfico:", ["Boxplot", "Barra Promedio"])
        if tipo == "Boxplot":
            chart2 = alt.Chart(df_top).mark_boxplot(color="#1DB954").encode(
                x=alt.X(f"{cat_var}:N", sort="-y", title=cat_var.capitalize()),
                y=alt.Y(f"{num_var}:Q", title=num_var.capitalize())
            )
        else:
            chart2 = alt.Chart(df_top).mark_bar(color="#1DB954").encode(
                x=alt.X(f"{cat_var}:N", sort="-y"),
                y=alt.Y(f"mean({num_var}):Q", title=f"Promedio de {num_var}")
            )

        st.altair_chart(chart2.properties(width=700, height=400), width="stretch")

    # --- TAB 3: Correlaciones ---
    with tab3:
        st.subheader("🔍 Correlación entre Variables")
        col1, col2 = st.columns(2)
        with col1:
            x_axis = st.selectbox("Eje X", numeric_cols, index=0)
        with col2:
            y_axis = st.selectbox("Eje Y", numeric_cols, index=1)
        color_var = st.selectbox("Color según:", ["popularity"] + numeric_cols, index=0)

        chart3 = alt.Chart(df).mark_circle(opacity=0.6, size=50).encode(
            x=alt.X(f"{x_axis}:Q"),
            y=alt.Y(f"{y_axis}:Q"),
            color=alt.Color(f"{color_var}:Q", scale=alt.Scale(scheme="viridis")),
            tooltip=[x_axis, y_axis, color_var]
        ).properties(width=700, height=450).interactive()
        st.altair_chart(chart3, width="stretch")

    with tab4:
        st.subheader("🌊 Mapa de Densidad 2D")
        st.caption("Visualiza regiones con alta concentración de canciones.")

        x_var = st.selectbox("Eje X:", numeric_cols, index=numeric_cols.index("danceability") if "danceability" in numeric_cols else 0)
        y_var = st.selectbox("Eje Y:", numeric_cols, index=numeric_cols.index("energy") if "energy" in numeric_cols else 1)

        chart4 = alt.Chart(df).mark_rect().encode(
            x=alt.X(f"{x_var}:Q", bin=alt.Bin(maxbins=30)),
            y=alt.Y(f"{y_var}:Q", bin=alt.Bin(maxbins=30)),
            color=alt.Color("count():Q", scale=alt.Scale(scheme="greens")),
            tooltip=["count():Q"]
        ).properties(width=700, height=450)

        if st.checkbox("Superponer puntos"):
            sample = df.sample(n=min(500, len(df)), random_state=42)
            points = alt.Chart(sample).mark_circle(size=30, color="white", opacity=0.4).encode(
                x=f"{x_var}:Q",
                y=f"{y_var}:Q"
            )
            chart4 = chart4 + points

        st.altair_chart(chart4.interactive(), width="stretch")

    with st.expander("👀 Vista previa de datos"):
        st.dataframe(df.head(10))
