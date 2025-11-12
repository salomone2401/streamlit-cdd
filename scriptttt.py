import joblib
import xgboost as xgb
import os

# Ruta del modelo actual en formato .pkl
pkl_path = os.path.join("models", "xgboost_model.pkl")

# Ruta del nuevo modelo exportado en formato .json
json_path = os.path.join("models", "xgb_model.json")

model = joblib.load(pkl_path)

# Verificamos que sea un modelo XGBoost
if not isinstance(model, xgb.XGBClassifier):
    raise TypeError("❌ El archivo cargado no es un XGBClassifier. Revisá el nombre o tipo de modelo.")

# Guardamos el modelo en formato JSON
model.save_model(json_path)
print(f"✅ Modelo exportado correctamente a: {json_path}")

# (Opcional) Verificamos que se pueda volver a cargar
try:
    test_model = xgb.XGBClassifier()
    test_model.load_model(json_path)
    print("🔍 Verificación exitosa: el modelo JSON se puede cargar sin errores.")
except Exception as e:
    print("⚠️ Error al verificar el modelo JSON:", e)
