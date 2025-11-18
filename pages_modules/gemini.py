import google.generativeai as genai
import streamlit as st

def gemini_explain(pred_proba, top_effects, effects, X_input):
    """
    pred_proba: float con prob de popularidad
    top_effects: Series con top features (ordenadas por importancia)
    effects: todas las contribuciones SHAP
    X_input: DataFrame con valores reales
    """

    api_key = st.secrets["GEMINI"]["api_key"]
    genai.configure(api_key=api_key)

    # Clasificación binaria clara
    predicted_label = "popular" if pred_proba >= 0.5 else "no popular"

    # Armamos tabla estructurada sin libertad creativa
    rows = []
    for feat in top_effects.index:
        val = X_input.iloc[0][feat]
        shap_val = effects[feat]
        direction = "aumenta" if shap_val > 0 else "reduce"
        rows.append(
            f"{feat} | valor={val} | shap={shap_val:.4f} | efecto={direction}"
        )
    structured_data = "\n".join(rows)

    # Prompt blindado
    prompt = f"""
Genera UNA explicación corta y humana basada exclusivamente en los datos brindados.

PREDICCIÓN:
- Probabilidad de popularidad: {pred_proba:.2f}
- Resultado: {predicted_label}

CARACTERÍSTICAS MÁS INFLUYENTES:
{structured_data}

INSTRUCCIONES ESTRICTAS:
- Produce SOLO UN PÁRRAFO.
- NO inventes géneros, emociones ni ningún dato extra.
- Explica únicamente cómo estas características influyeron según sus valores SHAP.
- Si un valor SHAP es negativo, explica que redujo la probabilidad.
- Si es positivo, explica que aumentó la probabilidad.
- Mantén el tono simple y claro.
"""

    model = genai.GenerativeModel("models/gemini-2.5-pro")
    response = model.generate_content(prompt)

    return response.text.strip()
