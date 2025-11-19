# eda.py (versión optimizada)
import streamlit as st
import pandas as pd
import altair as alt
import os
import numpy as np
import gc

# -------------------------
# CONFIG
# -------------------------
DEFAULT_MAX_ROWS = 5000   # máximo de filas que renderizamos con Altair (ajustá si querés)
SCATTER_MAX_POINTS = 3000
HEATMAP_BINS = 30

# -------------------------
# UI HEADER
# -------------------------
def render_header():
    st.markdown(
        """
        <div class="models-header" style="text-align:center;">
            <h2 style="color:#1DB954; margin-bottom:5px;">📊 Análisis Visual Interactivo</h2>
            <p style="color:#b3b3b3; font-size:1.1em;">
                Personaliza los gráficos y explora patrones ocultos en los datos de popularidad de Spotify.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


# -------------------------
# DATA LOADING & CACHING
# -------------------------
def load_data(path="data/data.csv", max_rows=DEFAULT_MAX_ROWS):
    """Carga el CSV una sola vez por sesión/entrada de datos."""
    if not os.path.exists(path):
        return pd.DataFrame()
    # low_memory=False para evitar avisos y lecturas parciales en datasets grandes
    df = pd.read_csv(path, nrows=max_rows)
    return df


def sample_df_for_viz(df: pd.DataFrame, max_rows: int = DEFAULT_MAX_ROWS):
    """
    Devuelve un sample reproducible (random_state fijo).
    `frames_key` se usa para invalidar el cache si cambia la fuente/params.
    """
    if df is None or df.empty:
        return df
    if len(df) <= max_rows:
        return df.copy()
    return df.sample(n=max_rows, random_state=42).reset_index(drop=True)


# -------------------------
# CACHED CHART PREPARATIONS
# -------------------------
@st.cache_data(show_spinner=False)
def prepare_histogram(df: pd.DataFrame, variable: str, bins: int):
    """Prepara DataFrame agregado para histograma (bins + counts)."""
    if df is None or df.empty or variable not in df.columns:
        return pd.DataFrame(columns=["bin_left", "bin_right", "count", variable])

    # drop NAs
    arr = pd.to_numeric(df[variable], errors="coerce").dropna()
    if arr.empty:
        return pd.DataFrame(columns=["bin_left", "bin_right", "count", variable])

    # calcular bins con pandas.cut para tener control y luego pasar a Altair
    bins_series = pd.cut(arr, bins=bins, include_lowest=True)
    agg = arr.groupby(bins_series).agg(["count", "min", "max"])
    agg = agg.reset_index()
    agg["bin_left"] = agg["min"]
    agg["bin_right"] = agg["max"]
    agg = agg.rename(columns={"count": "count"})
    agg = agg[["bin_left", "bin_right", "count"]]
    # crear etiqueta numérica en el centro del bin para usar en eje X cuantitativo
    agg["bin_mid"] = (agg["bin_left"] + agg["bin_right"]) / 2
    return agg


@st.cache_data
def prepare_boxplot(df: pd.DataFrame, cat_var: str, num_var: str, top_n: int):
    """Filtra top categorias y devuelve dataframe listo para boxplot."""
    if df is None or df.empty or cat_var not in df.columns or num_var not in df.columns:
        return pd.DataFrame()
    top_cats = df[cat_var].value_counts().index[:top_n].tolist()
    df_top = df[df[cat_var].isin(top_cats)].copy()
    # convertir num_var a numérico seguro
    df_top[num_var] = pd.to_numeric(df_top[num_var], errors="coerce")
    return df_top


@st.cache_data
def prepare_scatter(df: pd.DataFrame, x_axis: str, y_axis: str, color_var: str, max_points: int):
    """Sample para scatter: limitado a max_points y solo columnas necesarias."""
    if df is None or df.empty:
        return pd.DataFrame()
    cols = [c for c in [x_axis, y_axis, color_var] if c in df.columns]
    subset = df[cols].dropna()
    n = min(len(subset), max_points)
    if len(subset) > n:
        subset = subset.sample(n=n, random_state=42).reset_index(drop=True)
    return subset


@st.cache_data
def prepare_heatmap(df, x_var, y_var, x_bins=20, y_bins=20):
    if df is None or df.empty or x_var not in df.columns or y_var not in df.columns:
        return pd.DataFrame()

    # Convertir a numérico y eliminar nulos en las dos variables
    df_clean = df[[x_var, y_var]].copy()
    df_clean[x_var] = pd.to_numeric(df_clean[x_var], errors="coerce")
    df_clean[y_var] = pd.to_numeric(df_clean[y_var], errors="coerce")
    df_clean = df_clean.dropna()

    if df_clean.empty:
        return pd.DataFrame()

    try:
        # Crear bins usando intervalos fijos (no quantiles), pero con cuidado en bordes
        x_min, x_max = df_clean[x_var].min(), df_clean[x_var].max()
        y_min, y_max = df_clean[y_var].min(), df_clean[y_var].max()

        # Evitar división por cero o bins degenerados
        if x_min == x_max or y_min == y_max:
            return pd.DataFrame()

        # Usamos pd.cut manualmente con linspace para control total
        x_edges = np.linspace(x_min, x_max, x_bins + 1)
        y_edges = np.linspace(y_min, y_max, y_bins + 1)

        # Asignar bins (right=True por defecto en cut → [a, b))
        df_clean["x_bin_idx"] = pd.cut(df_clean[x_var], bins=x_edges, include_lowest=True, labels=False)
        df_clean["y_bin_idx"] = pd.cut(df_clean[y_var], bins=y_edges, include_lowest=True, labels=False)

        # Eliminar filas que no encajaron (puede pasar si hay infinitos o extremos)
        df_clean = df_clean.dropna(subset=["x_bin_idx", "y_bin_idx"])

        if df_clean.empty:
            return pd.DataFrame()

        # Contar ocurrencias por celda
        grouped = df_clean.groupby(["x_bin_idx", "y_bin_idx"]).size().reset_index(name="count")

        # Convertir índices de bin a bordes reales para tooltip y posicionamiento
        grouped["x_left"] = grouped["x_bin_idx"].apply(lambda i: x_edges[int(i)])
        grouped["x_right"] = grouped["x_bin_idx"].apply(lambda i: x_edges[int(i) + 1])
        grouped["y_left"] = grouped["y_bin_idx"].apply(lambda i: y_edges[int(i)])
        grouped["y_right"] = grouped["y_bin_idx"].apply(lambda i: y_edges[int(i) + 1])

        # Calcular centro para posición del rectángulo
        grouped["x_mid"] = (grouped["x_left"] + grouped["x_right"]) / 2
        grouped["y_mid"] = (grouped["y_left"] + grouped["y_right"]) / 2

        # ✅ Filtrar celdas vacías (count == 0 no debería pasar, pero por si acaso)
        grouped = grouped[grouped["count"] > 0].copy()

        # Asegurar que 'count' sea entero (Altair prefiere int para escala cuantitativa discreta)
        grouped["count"] = grouped["count"].astype(int)

        return grouped

    except Exception as e:
        st.warning(f"⚠️ Error al construir heatmap: {str(e)}")
        return pd.DataFrame()

# -------------------------
# CHARTS (construcción rápida, Altair hará rendering liviano)
# -------------------------
def build_histogram_from_agg(agg_df, variable, width=550, height=320):
    if agg_df is None or agg_df.empty:
        return alt.Chart(pd.DataFrame()).mark_bar().encode()
    chart = alt.Chart(agg_df).mark_bar().encode(
        x=alt.X("bin_mid:Q", title=variable, axis=alt.Axis(format="~s")),
        y=alt.Y("count:Q", title="Count")
    ).properties(width=width, height=height)
    return chart


def build_scatter(df_scatter, x_axis, y_axis, color_var, width=650, height=420):
    if df_scatter is None or df_scatter.empty:
        return alt.Chart(pd.DataFrame()).mark_circle().encode()
    enc = {
        "x": alt.X(f"{x_axis}:Q", title=x_axis),
        "y": alt.Y(f"{y_axis}:Q", title=y_axis),
        "tooltip": [x_axis, y_axis]
    }
    if color_var and color_var in df_scatter.columns:
        enc["color"] = alt.Color(f"{color_var}:Q", scale=alt.Scale(scheme="viridis"))
        enc["tooltip"].append(color_var)
    chart = alt.Chart(df_scatter).mark_circle(opacity=0.6, size=20).encode(**enc).properties(width=width, height=height).interactive()
    return chart


def build_heatmap(heat_df, x_var, y_var, width=550, height=320):
    if heat_df is None or heat_df.empty or "count" not in heat_df.columns:
        st.info("No hay datos suficientes para generar el mapa de densidad.")
        return alt.Chart(pd.DataFrame()).mark_rect()

    # Escala logarítmica si el rango es muy amplio (opcional, pero ayuda)
    max_count = heat_df["count"].max()
    if max_count > 100:
        color_scale = alt.Scale(scheme="greens", type="log")
    else:
        color_scale = alt.Scale(scheme="greens")

    chart = alt.Chart(heat_df).mark_rect().encode(
        x=alt.X("x_mid:Q", title=x_var, axis=alt.Axis(format=".2f")),
        y=alt.Y("y_mid:Q", title=y_var, axis=alt.Axis(format=".2f")),
        color=alt.Color(
            "count:Q",
            title="Frecuencia",
            scale=color_scale,
            legend=alt.Legend(format="~s")
        ),
        tooltip=[
            alt.Tooltip("x_left:Q", title="X desde", format=".2f"),
            alt.Tooltip("x_right:Q", title="X hasta", format=".2f"),
            alt.Tooltip("y_left:Q", title="Y desde", format=".2f"),
            alt.Tooltip("y_right:Q", title="Y hasta", format=".2f"),
            alt.Tooltip("count:Q", title="Canciones"),
        ]
    ).properties(
        width=width,
        height=height,
        title=f"Mapa de Densidad: {x_var} vs {y_var}"
    ).interactive()

    return chart


# -------------------------
# RENDER MAIN
# -------------------------
def render():
    render_header()

    max_rows = st.sidebar.slider("Máx filas para visualización (sampling)", 1000, 20000, DEFAULT_MAX_ROWS, step=500)
    path = "data/data.csv"
    if not os.path.exists(path):
        st.error("❌ No se encontraron datos en data/data.csv. Asegurate de tener el archivo y el path correctos.")
        return
    # Cargar sólo la muestra limitada
    try:
        total_rows = sum(1 for _ in open(path)) - 1  # descontar header
    except Exception:
        total_rows = 0
    df = pd.read_csv(path, nrows=max_rows)
    st.info(f"➡️ Mostrando solo las primeras {len(df)}/{total_rows} filas (límite visualización)")
    if total_rows > max_rows:
        st.warning(f"⚠️ El archivo de datos tiene {total_rows} filas, pero por memoria solo se usan {max_rows}. Usa filtros para otro análisis.")

    if df.empty:
        st.error("❌ No hay registros para analizar.")
        return

    # Variables numéricas y categóricas detectadas
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    # incluir columnas que puedan convertirse a numéricas
    possible_numeric = [c for c in df.columns if c not in numeric_cols]
    numeric_candidates = []
    for c in possible_numeric:
        try:
            pd.to_numeric(df[c].dropna().iloc[:20])
            numeric_candidates.append(c)
        except Exception:
            pass
    numeric_cols = numeric_cols + numeric_candidates

    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    # Sidebar options (caché del sample para toda la sesión para evitar recalcular)
    st.sidebar.subheader("Ajustes de visualización")
    show_points_in_heatmap = st.sidebar.checkbox("Superponer puntos en heatmap (limitado)", value=False)

    # Guardar sample en session_state para usar en las diferentes pestañas sin recalcular
    frames_key = f"sample_{max_rows}"
    if "df_samples" not in st.session_state:
        st.session_state["df_samples"] = {}
    if frames_key not in st.session_state["df_samples"]:
        st.session_state["df_samples"][frames_key] = sample_df_for_viz(df, max_rows=max_rows)

    df_vis = st.session_state["df_samples"][frames_key]

    tab1, tab2, tab3, tab4 = st.tabs([
        "Distribución",
        "Comparaciones",
        "Correlaciones",
        "Mapa de Densidad"
    ])

    with tab1:
        st.subheader("📈 Distribución de una variable")
        variable = st.selectbox("Selecciona la variable numérica:", numeric_cols, index=0)
        bins = st.slider("Número de bins", 5, 100, 30)
        color = st.color_picker("Color del gráfico", "#1DB954")

        # Preparar agregado (cacheado)
        agg = prepare_histogram(df_vis, variable, bins)
        chart1 = build_histogram_from_agg(agg, variable, width=650, height=340)
        # aplicar color con transform si el chart no está vacío
        chart1 = chart1.encode(color=alt.value(color)) if not agg.empty else chart1

        st.altair_chart(chart1, use_container_width=True)

    with tab2:
        st.subheader("🎵 Comparación por Categoria")
        if not categorical_cols:
            st.info("No hay columnas categóricas detectadas.")
        else:
            cat_var = st.selectbox("Categoria (eje X):", categorical_cols, index=0)
            num_default_index = 0
            if "popularity" in numeric_cols:
                num_default_index = numeric_cols.index("popularity")
            num_var = st.selectbox("Variable numérica (eje Y):", numeric_cols, index=num_default_index)
            top_n = st.slider("Mostrar top categorías más frecuentes", 5, 50, 10)

            df_top = prepare_boxplot(df_vis, cat_var, num_var, top_n)
            bar_data = df_top.groupby(cat_var)[num_var].mean().reset_index().rename(columns={num_var: "mean_val"})
            chart2 = alt.Chart(bar_data).mark_bar().encode(
                x=alt.X(f"{cat_var}:N", sort="-y"),
                y=alt.Y("mean_val:Q", title=f"Promedio de {num_var}"),
                tooltip=[cat_var, "mean_val"]
            ).properties(width=700, height=350)
            st.altair_chart(chart2, use_container_width=True)

    with tab3:
        st.subheader("🔍 Correlación entre Variables")
        if len(numeric_cols) < 2:
            st.info("Necesitás al menos 2 variables numéricas para este gráfico.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                x_axis = st.selectbox("Eje X", numeric_cols, index=0)
            with col2:
                y_axis = st.selectbox("Eje Y", numeric_cols, index=min(1, len(numeric_cols)-1))

            # Eliminamos `color_var` -> scatter sin color dinámico
            df_scatter = prepare_scatter(df_vis, x_axis, y_axis, None, max_points=SCATTER_MAX_POINTS)
            chart3 = build_scatter(df_scatter, x_axis, y_axis, None, width=700, height=420)

            st.altair_chart(chart3, use_container_width=True)
            st.caption(f"Mostrando {len(df_scatter):,} puntos (máx {SCATTER_MAX_POINTS}).")
# ---------- TAB 4: HEATMAP (REEMPLAZAR BLOQUE EXISTENTE) ----------
    with tab4:
        st.subheader("🌊 Mapa de Densidad 2D")
        st.caption("Visualiza regiones con alta concentración de canciones.")
        x_var = st.selectbox("Eje X:", numeric_cols, index=numeric_cols.index("danceability") if "danceability" in numeric_cols else 0)
        y_var = st.selectbox("Eje Y:", numeric_cols, index=numeric_cols.index("energy") if "energy" in numeric_cols else min(1, len(numeric_cols)-1))

        # Usamos Altair para hacer binning + agregación en el browser de forma eficiente
        # Partimos de df_vis (sample cacheado) para mantener performance
        if df_vis is None or df_vis.empty:
            st.info("No hay datos para mostrar el heatmap.")
        else:
            chart4 = (
                alt.Chart(df_vis)
                .mark_rect()
                .encode(
                    x=alt.X(f"{x_var}:Q", bin=alt.Bin(maxbins=HEATMAP_BINS), title=x_var),
                    y=alt.Y(f"{y_var}:Q", bin=alt.Bin(maxbins=HEATMAP_BINS), title=y_var),
                    color=alt.Color("count():Q", title="Frecuencia", scale=alt.Scale(scheme="greens")),
                    tooltip=[alt.Tooltip("count():Q", title="Frecuencia")]
                )
                .properties(width=650, height=420)
                .interactive()
            )

            st.altair_chart(chart4, use_container_width=True)

            if show_points_in_heatmap:
                sample_pts = df_vis.sample(n=min(500, len(df_vis)), random_state=42)
                pts = alt.Chart(sample_pts).mark_circle(size=18, opacity=0.6, color="white").encode(
                    x=alt.X(f"{x_var}:Q"),
                    y=alt.Y(f"{y_var}:Q"),
                    tooltip=[x_var, y_var]
                )
                st.altair_chart((chart4 + pts).interactive(), use_container_width=True)


    with st.expander("👀 Vista previa de datos"):
        st.dataframe(df.head(10))

    # Limpieza explícita: liberar df, llamar a garbage collector
    del df
    gc.collect()
