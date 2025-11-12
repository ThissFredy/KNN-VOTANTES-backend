from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
from contextlib import asynccontextmanager
from typing import List
from app.model import entrenar_modelo_al_inicio, predecir_votante

# En esta varible global guardaremos los artefactos (imputer, scaler, X_train, etc.)
model_artifacts = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("--- Evento de Inicio (Startup) ---")
    global model_artifacts
    try:
        # llamamos a la función de entrenamiento
        model_artifacts = entrenar_modelo_al_inicio(
            file_path="data/voter_intentions_3000.csv"
        )
        print("--- Modelo entrenado y cargado en memoria. API lista. ---")
    except Exception as e:
        print(f"Error fatal durante el entrenamiento al inicio: {e}")
        model_artifacts = {"error": str(e)} # Guardar error para reportarlo
    
    yield
    
    # Limpiamos los artefactos al apagar (shutdown)
    print("--- Evento de Apagado (Shutdown) ---")
    model_artifacts.clear()

# --- Configuración de la App ---
app = FastAPI(
    title="API de Intención de Voto (k-NN)",
    description="API que entrena un modelo k-NN desde cero al iniciar.",
    lifespan=lifespan # Asocia el evento de inicio/apagado
)

allowed_origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:3000",
    "http://127.0.0.1:8000",
    "http://127.0.0.1:3000",
    "https://knn-votantes-frontend.onrender.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Modelos de Datos (Pydantic) ---
class VoterInput(BaseModel):
    primary_choice: str = Field(..., description="Elección primaria (texto)", example='CAND_Azon')
    public_sector: int = Field(..., description="Sector público (numérico)", example=1)
    gender: int = Field(..., description="Género (numérico)", example=2)
    job_tenure_years: int = Field(..., description="Años en el trabajo actual", example=10)
    social_media_hours: int = Field(..., description="Horas diarias en redes sociales", example=3)
    undecided: int = Field(..., description="Esta persona está indecisa", example=1)



# Modelo de Datos de Salida
class PredictionOutput(BaseModel):
    predicted_class: int
    predicted_candidate: str
    confidence: float = Field(..., description="Confianza de la predicción ")

# Modelo para un solo candidato
class CandidateInfo(BaseModel):
    code: int = Field(..., example=0, description="Código numérico interno del candidato")
    name: str = Field(..., example="Candidato_A", description="Nombre del candidato")

# Modelo para la lista completa de candidatos
class CandidatesList(BaseModel):
    candidates: List[CandidateInfo] = Field(..., description="Lista de todos los candidatos disponibles para predicción")

# --- Endpoints ---

@app.get("/")
async def root():
    return {"message": "¡Bienvenido! La API está corriendo."}

@app.post("/predict", response_model=PredictionOutput)
async def predict(data: VoterInput):
    """
    Realiza una predicción de intención de voto usando k-NN puro.
    """
    if "error" in model_artifacts:
        raise HTTPException(status_code=503, detail=f"El modelo no pudo entrenar: {model_artifacts['error']}")
    if not model_artifacts:
         raise HTTPException(status_code=503, detail="El modelo aún no está listo (entrenando).")

    try:
        # --- 1. Pre-procesamiento ---
        input_data = data.dict()
        votante_df = pd.DataFrame([input_data])
        feature_category_maps = model_artifacts.get("feature_category_maps", {})
        for col, category_map in feature_category_maps.items():
            votante_df[col] = votante_df[col].map(category_map).fillna(-1).astype(int)
        votante_df = votante_df[model_artifacts["feature_cols"]]
        votante_imputed = model_artifacts["imputer"].transform(votante_df)
        votante_scaled = model_artifacts["scaler"].transform(votante_imputed)

        # --- 2. Realizar Predicción y regresa en una tupla ---
        prediction_class, prediction_confidence = predecir_votante(
            X_entrenamiento=model_artifacts["X_train"],
            y_entrenamiento=model_artifacts["y_train"],
            nuevo_votante=votante_scaled[0],
            k=8
        )
        
        # --- 3. Mapear Resultado  ---
        prediction_label = model_artifacts["target_map"].get(prediction_class, "Candidato Desconocido")
        
        # --- 4. Devolver la respuesta con la nueva métrica ---
        return PredictionOutput(
            predicted_class=int(prediction_class),
            predicted_candidate=prediction_label,
            confidence=round(prediction_confidence, 4) # Redondeamos para una mejor visualización
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error durante la predicción: {str(e)}")
    
@app.get("/candidates", response_model=CandidatesList, tags=["Información del Modelo"])
async def get_available_candidates():
    """
    Devuelve la lista de todos los candidatos que el modelo puede predecir.
    """
    # Chequeo de seguridad: ¿El modelo se entrenó correctamente?
    if "error" in model_artifacts:
        raise HTTPException(status_code=503, detail=f"El modelo no pudo entrenar: {model_artifacts['error']}")
    if not model_artifacts:
         raise HTTPException(status_code=503, detail="El modelo aún no está listo (entrenando).")

    # Extraemos el mapa de target que guardamos durante el entrenamiento
    target_map = model_artifacts.get("target_map")
    
    if not target_map:
        raise HTTPException(status_code=404, detail="No se encontró la lista de candidatos en el modelo cargado.")

    # Transformamos el diccionario {code: name} en una lista de diccionarios
    # para que coincida con el formato de CandidateInfo
    list_of_candidates = [
        {"code": code, "name": name} for code, name in target_map.items()
    ]
    
    # Ordenamos la lista por código para que la salida sea predecible
    list_of_candidates.sort(key=lambda x: x["code"])

    return {"candidates": list_of_candidates}