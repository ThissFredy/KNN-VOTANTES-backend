from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
from contextlib import asynccontextmanager
from typing import List
from app.model import entrenar_modelo_al_inicio, predecir_votante

model_artifacts = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("--- Evento de Inicio (Startup) ---")
    global model_artifacts
    try:
        model_artifacts = entrenar_modelo_al_inicio(
            file_path="data/voter_intentions_COMPLETED.csv"
        )
        
        if "error" in model_artifacts:
            print(f"Error fatal durante la carga de datos: {model_artifacts['error']}")
        else:
            print("--- Modelo (datos procesados) cargado en memoria. API lista. ---")
            
    except Exception as e:
        print(f"Excepción fatal durante el startup: {e}")
        model_artifacts = {"error": str(e)}
    
    yield
    
    print("--- Evento de Apagado (Shutdown) ---")
    model_artifacts.clear()

app = FastAPI(
    title="API de Intención de Voto (k-NN Puro)",
    description="API que entrena un modelo k-NN desde cero al iniciar.",
    lifespan=lifespan
)

allowed_origins = [
    "http://localhost", "http://localhost:3000",
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
    # continuas
    age: int = Field(..., example=35)
    household_size: int = Field(..., example=3)
    refused_count: int = Field(..., example=0)
    tv_news_hours: int = Field(..., example=10)
    social_media_hours: int = Field(..., example=5)
    job_tenure_years: int = Field(..., example=4)


    # Asumo que tu script les puso prefijo 'ord__'
    has_children: int = Field(..., example=1.0)
    gender: int = Field(..., example=1.0)
    urbanicity: int = Field(..., example=1.0)
    education: int = Field(..., example=5.0)
    employment_status: int = Field(..., example=5.0)
    income_bracket: int = Field(..., example=5.0)
    employment_sector: int = Field(..., example=2.0)
    marital_status: int = Field(..., example=2.0)
    small_biz_owner: int = Field(..., example=0.0)
    owns_car: int = Field(..., example=0.0)
    preference_strength: int = Field(..., example=8.0)
    survey_confidence: int = Field(..., example=8.0)
    trust_media: int = Field(..., example=5.0)
    civic_participation: int = Field(..., example=4.0)
    wa_groups: int = Field(..., example=1.0)
    voted_last: int = Field(..., example=1.0)
    home_owner: int = Field(..., example=0.0)
    attention_check: int = Field(..., example=1.0)
    public_sector: int = Field(..., example=1.0)
    party_id_strength: int = Field(..., example=2.0)
    union_member: int = Field(..., example=0.0)

    primary_choice: str = Field(..., example="CAND_Gaia")
    secondary_choice: str = Field(..., example="CAND_Azon")

# Modelo de Datos de Salida
class PredictionOutput(BaseModel):
    predicted_class: int
    predicted_candidate: str
    confidence: float = Field(..., description="Confianza de la predicción")

# (Dejamos los modelos CandidateInfo y CandidatesList como estaban)
class CandidateInfo(BaseModel):
    code: int = Field(..., example=0)
    name: str = Field(..., example="Candidato_A")

class CandidatesList(BaseModel):
    candidates: List[CandidateInfo]


# * --- Endpoints ---

@app.get("/")
async def root():
    return {"message": "¡Bienvenido! La API está corriendo."}


@app.post("/predict", response_model=PredictionOutput)
async def predict(data: VoterInput):
    """
    Realiza una predicción de k-NN usando datos YA PROCESADOS.
    """
    if "error" in model_artifacts:
        raise HTTPException(status_code=503, detail=f"El modelo no pudo cargar: {model_artifacts['error']}")
    
    X_train = model_artifacts.get("X_train")
    y_train = model_artifacts.get("y_train")
    target_map = model_artifacts.get("target_map")
    preprocessor = model_artifacts.get("live_preprocessor")
    
    feature_cols = model_artifacts.get("feature_cols")
    
    if not all([X_train is not None, y_train is not None, target_map, feature_cols]):
        raise HTTPException(status_code=503, detail="El modelo o sus artefactos no están listos.")

    try:
        input_data = data.model_dump() if hasattr(data, 'model_dump') else data.dict()

        votante_df = pd.DataFrame([input_data])
        
        votante_processed_arr = preprocessor.transform(votante_df)

        # Obtener vector 1D para la predicción
        votante_np = np.asarray(votante_processed_arr).reshape(votante_processed_arr.shape[0], -1)[0]

        print(f"Datos del votante procesados para predicción: {votante_np}")

        prediction_class, prediction_confidence = predecir_votante(
            X_entrenamiento=X_train,
            y_entrenamiento=y_train,
            nuevo_votante=votante_np,
            k=19
        )
        
        prediction_label = target_map.get(prediction_class, "Candidato Desconocido")


        processed_row_df = pd.DataFrame([votante_np], columns=feature_cols)
        processed_row_df["intended_vote"] = prediction_label

        actual_data = pd.read_csv("data/voter_intentions_COMPLETED.csv")
        actual_data = pd.concat([actual_data, processed_row_df], ignore_index=True)
        actual_data.to_csv("data/voter_intentions_COMPLETED.csv", index=False)

        model_artifacts["X_train"] = np.vstack([X_train, votante_processed_arr])
        model_artifacts["y_train"] = np.append(y_train, prediction_class)

        return PredictionOutput(
            predicted_class=int(prediction_class),
            predicted_candidate=prediction_label,
            confidence=round(prediction_confidence, 4)
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error durante la predicción: {str(e)}")


@app.get("/candidates", response_model=CandidatesList, tags=["Información del Modelo"])
async def get_available_candidates():
    """
    Devuelve la lista de todos los candidatos que el modelo puede predecir.
    """
    if "error" in model_artifacts:
        raise HTTPException(status_code=503, detail=f"El modelo no pudo cargar: {model_artifacts['error']}")

    target_map = model_artifacts.get("target_map")
    
    if not target_map:
        raise HTTPException(status_code=404, detail="No se encontró la lista de candidatos en el modelo cargado.")

    list_of_candidates = [
        {"code": code, "name": name} for code, name in target_map.items()
    ]
    list_of_candidates.sort(key=lambda x: x["code"])

    return {"candidates": list_of_candidates}