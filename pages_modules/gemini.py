import google.generativeai as genai
import streamlit as st

def gemini_explain(pred_proba, X_input):
    """
    Explicación basada únicamente en:
    - probabilidad predicha
    - valores reales que ingresó el usuario
    """

    api_key = st.secrets["GEMINI"]["api_key"]
    genai.configure(api_key=api_key)

    predicted_label = "popular" if pred_proba >= 0.5 else "no popular"

    # Convertir los valores de la fila en texto
    rows = []
    for feat in X_input.columns:
        val = X_input.iloc[0][feat]
        rows.append(f"{feat}: {val}")
    structured_values = "\n".join(rows)

    prompt = f"""
Genera una explicación corta y clara sobre por qué la canción fue clasificada como '{predicted_label}'.

DATOS:
- Probabilidad estimada: {pred_proba:.2f}
- Valores reales ingresados por el usuario:
{structured_values}

INSTRUCCIONES:
- Produce SOLO UN PÁRRAFO.
- No inventes valores ni datos que no estén en la lista.
- Describe cómo los valores de las características pueden influir en que una canción sea popular o no.
- No menciones SHAP ni modelos internos, solo habla en términos intuitivos.
- Mantén el tono simple, humano y directo.
"""

    model = genai.GenerativeModel("models/gemini-2.5-pro")
    response = model.generate_content(prompt)

    return response.text.strip()
