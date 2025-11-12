import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score

def calcular_distancia_euclidiana(punto_a, punto_b):
    """Calcula la distancia euclidiana entre dos puntos (vectores NumPy)."""
    return np.sqrt(np.sum((punto_a - punto_b)**2))

def predecir_votante(X_entrenamiento, y_entrenamiento, nuevo_votante, k=8):
    """
    Predice la clase de un nuevo votante usando k-NN manual.
    Espera que X_entrenamiento y nuevo_votante sean arrays de NumPy.
    """
    # Calcula distancia euclidiana
    distancias = [
        (calcular_distancia_euclidiana(votante_entrenamiento, nuevo_votante), y_entrenamiento[i])
        for i, votante_entrenamiento in enumerate(X_entrenamiento)
    ]
    
    # Ordena y obtiene los k vecinos más cercanos
    distancias_ordenadas = sorted(distancias, key=lambda x: x[0])
    k_vecinos = [et for _, et in distancias_ordenadas[:k]]
    
    # Votación: devuelve la clase más común y su "confianza"
    pred, conteo = Counter(k_vecinos).most_common(1)[0]
    return pred, conteo / k


# --- Entrenamiento del Modelo al Inicio ---
def entrenar_modelo_al_inicio(file_path: str) -> dict:
    """
    Carga los datos YA PROCESADOS, los separa y "entrena" el modelo k-NN.
    "Entrenar" es solo guardar los datos de entrenamiento.
    """
    print(f"Iniciando carga desde el dataset PROCESADO: {file_path}...")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"ERROR: No se encontró el archivo {file_path}")
        return {"error": f"No se encontró {file_path}"}
    
    # 1) Definir X (features) e y (target)
    target_col = "intended_vote"
    
    if target_col not in df.columns:
        print(f"ERROR: El target '{target_col}' no está en el CSV.")
        return {"error": f"El target '{target_col}' no está en el CSV."}

    # X es TODO menos el target. Ya está limpio y procesado.
    X_df = df.drop(columns=[target_col])
    
    # Guardamos los nombres de las columnas procesadas (ej. 'ord__has_children', 'nom__primary_choice_CAND_A')
    feature_cols = list(X_df.columns)
    
    # 2) Hallar y (y convertir a códigos numéricos)
    y_labels = df[target_col].astype("category")
    target_map = dict(enumerate(y_labels.cat.categories))
    y = y_labels.cat.codes.values # y (array de numpy con los códigos: 0, 1, 2...)

    # 3) Separar los datos para evaluación
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4) Convertir a NumPy para la función manual
    # ESTOS SON LOS DATOS QUE EL MODELO USARÁ
    X_train_np = X_train_df.values
    X_test_np = X_test_df.values

    # 5) Evaluación del KNN (Usando los datos de Test)
    print("Iniciando evaluación del modelo KNN manual sobre datos procesados...")
    preds, confs = [], []
    for fila in X_test_np:
        p, c = predecir_votante(X_train_np, y_train, fila, k=8)
        preds.append(p); confs.append(c)

    # Métricas
    acc = accuracy_score(y_test, preds)
    f1_macro = f1_score(y_test, preds, average='macro', zero_division=0)
    
    print(f"\n--- Reporte de Evaluación (k=8, datos procesados) ---")
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"F1-score (macro): {f1_macro:.4f}")
    
    nombres = list(target_map.values())
    print(classification_report(y_test, preds, target_names=nombres))
    print("---------------------------------------------------------")

    print("¡'Entrenamiento' (carga de datos) completado!")
    
    # 6) Devolver solo lo necesario para predecir
    return {
        "X_train": X_train_np,       # Los datos numéricos para comparar
        "y_train": y_train,          # Las etiquetas de esos datos
        "target_map": target_map,    # El mapa {0: 'CandidatoA', 1: 'CandidatoB'}
        "feature_cols": feature_cols,# La lista de columnas procesadas que espera la API
        "metrics": {
            "accuracy": acc,
            "f1_score_macro": f1_macro,
        }
    }
    """
    Carga los datos, los separa y "entrena" el modelo k-NN.
    En k-NN, "entrenar" es solo guardar los datos de entrenamiento.
    """
    print(f"Iniciando carga y entrenamiento desde {file_path}...")
    df = pd.read_csv(file_path)

    # 1) Definir columnas para X (SOLO NUMÉRICAS) y el target (y)
    target_col = "intended_vote"
    
    # (Estas son tus 25 variables con importancia positiva que ya son numéricas)
    numeric_cols = [
        "undecided", "party_id_strength", "survey_confidence",
        "civic_participation", "region", "marital_status", "public_sector",
        "income_bracket", "urbanicity", "education", "will_turnout",
        "tv_news_hours", "home_owner", "voted_last",
        "employment_sector", "job_tenure_years", "social_media_hours",
        "refused_count", "age", "household_size", "wa_groups",
        "small_biz_owner", "trust_media", "gender", "attention_check",
    ]

    # 2) Hallar X
    # Rellenamos NaN con 0. Es el mínimo procesamiento necesario
    # para que la función 'calcular_distancia_euclidiana' no falle.
    X = df[numeric_cols].fillna(0)

    # 3) Hallar y
    y_labels = df[target_col].astype("category")
    target_map = dict(enumerate(y_labels.cat.categories))
    y = y_labels.cat.codes.values # y (array de numpy con los códigos: 0, 1, 2...)

    # 4) Separar los datos
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5) Convertir a NumPy para la función manual (es más rápido)
    # k-NN "almacena" estos datos para la predicción.
    X_train_np = X_train_df.values
    X_test_np = X_test_df.values

    # -------------------------------
    # 6) Evaluación del KNN (Usando los datos de Test)
    # -------------------------------
    print("Iniciando evaluación del modelo KNN manual...")
    preds, confs = [], []
    for fila in X_test_np:
        # Usamos los datos de "entrenamiento" para predecir cada fila de "test"
        p, c = predecir_votante(X_train_np, y_train, fila, k=8)
        preds.append(p); confs.append(c)

    # Métricas
    acc = accuracy_score(y_test, preds)
    f1_macro = f1_score(y_test, preds, average='macro', zero_division=0)
    
    print(f"\n--- Reporte de Evaluación (k=8, sin preprocesamiento) ---")
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"F1-score (macro): {f1_macro:.4f}")
    
    nombres = list(target_map.values())
    print(classification_report(y_test, preds, target_names=nombres))
    print("---------------------------------------------------------")

    print("¡'Entrenamiento' (carga de datos) completado!")
    
    # 7) Devolver solo lo necesario para predecir
    return {
        "X_train": X_train_np,       # Los datos numéricos para comparar
        "y_train": y_train,          # Las etiquetas de esos datos
        "target_map": target_map,    # El mapa {0: 'CandidatoA', 1: 'CandidatoB'}
        "numeric_cols": numeric_cols,# La lista de columnas que espera la API
        "metrics": {                 # Métricas de evaluación
            "accuracy": acc,
            "f1_score_macro": f1_macro,
        }
    }