# Usa la imagen base oficial de Python
FROM python:3.9-slim

# Configura el directorio de trabajo
WORKDIR /app

# Copia el archivo de requerimientos y el código de la app a /app
COPY requirements.txt .
COPY app.py .

# Instala las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copia el archivo del modelo al contenedor
COPY random_forest_model.joblib /app/random_forest_model.joblib .

# Expone el puerto 8000 para FastAPI
EXPOSE 8000

# Comando para iniciar la app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
