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
    # Asumo que tu script les puso prefijo 'ord__'
    ord__has_children: float = Field(..., example=1.0)
    ord__home_owner: float = Field(..., example=0.0)
    ord__public_sector: float = Field(..., example=1.0)
    ord__union_member: float = Field(..., example=0.0)
    
    # Asumo prefijo 'nom__' y 10 candidatos para 'primary_choice'
    nom__primary_choice_CAND_Azon: float = Field(..., example=1.0)
    nom__primary_choice_CAND_Boreal: float = Field(..., example=0.0)
    nom__primary_choice_CAND_Civico: float = Field(..., example=0.0)
    nom__primary_choice_CAND_Demetra: float = Field(..., example=0.0)
    nom__primary_choice_CAND_Electra: float = Field(..., example=0.0)
    nom__primary_choice_CAND_Frontera: float = Field(..., example=0.0)
    nom__primary_choice_CAND_Gaia: float = Field(..., example=0.0)
    nom__primary_choice_CAND_Halley: float = Field(..., example=0.0)
    nom__primary_choice_CAND_Icaro: float = Field(..., example=0.0)
    nom__primary_choice_CAND_Jade: float = Field(..., example=0.0)
        
    class Config:
        schema_extra = {
            "example": {
                "ord__has_children": 1.0,
                "ord__home_owner": 0.0,
                "ord__public_sector": 1.0,
                "ord__union_member": 0.0,
                "nom__primary_choice_CAND_Azon": 0.0,
                "nom__primary_choice_CAND_Boreal": 0.0,
                "nom__primary_choice_CAND_Civico": 0.0,
                "nom__primary_choice_CAND_Demetra": 0.0,
                "nom__primary_choice_CAND_Electra": 0.0,
                "nom__primary_choice_CAND_Frontera": 0.0,
                "nom__primary_choice_CAND_Gaia": 1.0,
                "nom__primary_choice_CAND_Halley": 0.0,
                "nom__primary_choice_CAND_Icaro": 0.0,
                "nom__primary_choice_CAND_Jade": 0.0
            }
        }

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
    
    # 1. Validar que los artefactos NUEVOS estén cargados
    X_train = model_artifacts.get("X_train")
    y_train = model_artifacts.get("y_train")
    target_map = model_artifacts.get("target_map")
    
    feature_cols = model_artifacts.get("feature_cols") # La lista de columnas procesadas
    
    if not all([X_train is not None, y_train is not None, target_map, feature_cols]):
         raise HTTPException(status_code=503, detail="El modelo o sus artefactos no están listos.")

    try:
        # --- Preparar datos de entrada ---
        
        input_data = data.model_dump() if hasattr(data, 'model_dump') else data.dict()
        votante_df = pd.DataFrame([input_data])
        votante_df_ordered = votante_df[feature_cols]
        

        votante_np = votante_df_ordered.values[0]



        prediction_class, prediction_confidence = predecir_votante(
            X_entrenamiento=X_train,
            y_entrenamiento=y_train,
            nuevo_votante=votante_np,
            k=8 
        )
        
        prediction_label = target_map.get(prediction_class, "Candidato Desconocido")

        data_to_save = input_data.copy()
        data_to_save["intended_vote"] = prediction_label

        actual_data = pd.read_csv("data/voter_intentions_COMPLETED.csv")
        actual_data = pd.concat(
            [actual_data, pd.DataFrame([data_to_save])], ignore_index=True
        )
        
        actual_data.to_csv("data/voter_intentions_COMPLETED.csv", index=False)

        # 5. Actualizar modelo en MEMORIA (para la siguiente petición)
        model_artifacts["X_train"] = np.vstack([X_train, votante_np])
        model_artifacts["y_train"] = np.append(y_train, prediction_class)
        print(f"Modelo actualizado. Nuevo tamaño: {len(model_artifacts['y_train'])}")

        # --- 5. Devolver la respuesta ---
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