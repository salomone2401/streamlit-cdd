import streamlit as st


with open("utils/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)



st.set_page_config(
    page_title="Proyecto Popularidad",
    page_icon="🎧",
    layout="wide"
)
st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
st.image("assets/logo.png", width=140)
st.markdown("</div>", unsafe_allow_html=True)


# TITULO CENTRADO
st.markdown("""
<div style="text-align:right">
    <h1 style="color:#1DB954;">
        Popularidad de Canciones
    </h1>
</div>
""", unsafe_allow_html=True)


st.write("""
Este proyecto tiene como objetivo analizar la popularidad de una cancion en base a sus caracteristicas musicales. Para ello se utilizaran datos de Spotify y se entrenara un modelo de machine learning para predecir la popularidad de una cancion en base a sus caracteristicas musicales. 
""")

opcion = st.sidebar.radio(
    "",
    ["Introducción", "EDA", "Modelo", "Predicción", "Conclusiones"]
)
