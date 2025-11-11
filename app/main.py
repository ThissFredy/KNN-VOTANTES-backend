from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
from contextlib import asynccontextmanager
from app.model import entrenar_modelo_al_inicio, predecir_votante

model_artifacts = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("--- Evento de Inicio (Startup) ---")
    global model_artifacts
    try:
        # Aquí ocurre la "magia": llamamos a la función de entrenamiento
        # Asegúrate que la ruta al data/ sea correcta desde donde ejecutas uvicorn
        # O usa una ruta absoluta. Para Docker, 'data/...' está bien.
        model_artifacts = entrenar_modelo_al_inicio(
            file_path="data/voter_intentions_3000.csv"
        )
        print("--- Modelo entrenado y cargado en memoria. API lista. ---")
    except Exception as e:
        print(f"Error fatal durante el entrenamiento al inicio: {e}")
        model_artifacts = {"error": str(e)} # Guardar error para reportarlo
    
    yield
    
    # Esto se ejecuta al apagar (shutdown)
    print("--- Evento de Apagado (Shutdown) ---")
    model_artifacts.clear()

# --- Configuración de la App ---
app = FastAPI(
    title="API de Intención de Voto (k-NN Puro)",
    description="API que entrena un modelo k-NN desde cero al iniciar.",
    lifespan=lifespan # Asocia el evento de inicio/apagado
)

raw_origins = os.getenv("ALLOWED_ORIGINS", "*")
if raw_origins == "*":
    allowed_origins = ["*"]
else:
    # Remover espacios y filtrar vacíos
    allowed_origins = [o.strip() for o in raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Modelos de Datos (Pydantic) ---
class VoterInput(BaseModel):
    age: int = Field(..., description="Edad del votante", example=55)
    income_bracket: int = Field(..., description="Nivel de ingresos (numérico)", example=4)
    party_id_strength: int = Field(..., description="Fuerza de ID de partido (numérico)", example=8)
    tv_news_hours: int = Field(..., description="Horas de noticias de TV", example=3)
    social_media_hours: int = Field(..., description="Horas de redes sociales", example=1)
    trust_media: int = Field(..., description="Confianza en medios (numérico)", example=1)
    civic_participation: int = Field(..., description="Participación cívica (numérico)", example=2)

# Modelo de Datos de Salida
class PredictionOutput(BaseModel):
    predicted_class: int
    predicted_candidate: str

# --- Endpoints ---

@app.get("/")
async def root():
    return {"message": "¡Bienvenido! La API está corriendo."}

@app.post("/predict", response_model=PredictionOutput)
async def predict(data: VoterInput):
    """
    Realiza una predicción de intención de voto usando k-NN puro.
    """
    # Chequeo de seguridad: ¿El modelo se entrenó correctamente al inicio?
    if "error" in model_artifacts:
        raise HTTPException(status_code=503, detail=f"El modelo no pudo entrenar: {model_artifacts['error']}")
    if not model_artifacts:
         raise HTTPException(status_code=503, detail="El modelo aún no está listo (entrenando).")

    try:
        # --- 1. Preparar Datos de Entrada ---
        # Convertir el Pydantic a un dict de Python
        input_data = data.dict()
        
        # Crear el array en el orden correcto
        votante_array = [input_data.get(col) for col in model_artifacts["feature_cols"]]
        votante_df = pd.DataFrame([votante_array], columns=model_artifacts["feature_cols"])

        # --- 2. Aplicar Pre-procesadores ---
        # Usamos los artefactos cargados en 'model_artifacts'
        votante_imputed = model_artifacts["imputer"].transform(votante_df)
        votante_scaled = model_artifacts["scaler"].transform(votante_df)

        # --- 3. Realizar Predicción ---
        # Usamos tu función k-NN, pasándole X_train y y_train de 'model_artifacts'
        prediction_class = predecir_votante(
            X_entrenamiento=model_artifacts["X_train"],
            y_entrenamiento=model_artifacts["y_train"],
            nuevo_votante=votante_scaled[0], # [0] porque transform devuelve 2D
            k=8 # k=8 hardcodeado (puedes cambiarlo)
        )
        
        # --- 4. Mapear Resultado ---
        # .get() es más seguro que acceso directo por llave
        prediction_label = model_artifacts["target_map"].get(prediction_class, "Clase Desconocida")
        
        return PredictionOutput(
            predicted_class=int(prediction_class),
            predicted_candidate=prediction_label
        )

    except Exception as e:
        # Captura cualquier error durante la predicción
        raise HTTPException(status_code=400, detail=f"Error durante la predicción: {str(e)}")