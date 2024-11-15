# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import uvicorn

# Cargar el modelo entrenado
model = joblib.load("random_forest_model.joblib")

# Inicializar la aplicación FastAPI
app = FastAPI()

# Definir el esquema de datos de entrada
class CustomerData(BaseModel):
    annual_income: float
    total_spent: float
    avg_purchase_value: float
    online_activity_score: float
    gender: int = 1


# Endpoint para la predicción
@app.post("/predict")
def predict(data: CustomerData):
    # Convertir la entrada en el formato esperado por el modelo
    features = np.array([[data.annual_income, data.total_spent, data.avg_purchase_value, data.online_activity_score, data.gender]])
    
    # Realizar la predicción
    prediction = model.predict(features)
    
    # Convertir la predicción en un resultado legible
    customer_segment = int(prediction[0])
    
    return {"customer_segment": customer_segment}

# Iniciar el servidor de FastAPI si el script se ejecuta directamente
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
