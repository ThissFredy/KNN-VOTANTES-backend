import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

# --- Tus Funciones de k-NN (Copiadas de tu script) ---

def calcular_distancia_euclidiana(punto_a, punto_b):
    """
    Calcula la distancia euclidiana entre dos puntos.
    """
    return np.sqrt(np.sum((punto_a - punto_b)**2))


def predecir_votante(X_entrenamiento, y_entrenamiento, nuevo_votante, k=8):
    """
    Predice la clase de un nuevo votante usando KNN (con k=8 por defecto).
    """
    # Calcular distancias a todos los puntos de entrenamiento
    distancias = [
        (calcular_distancia_euclidiana(votante_entrenamiento, nuevo_votante), y_entrenamiento[i])
        for i, votante_entrenamiento in enumerate(X_entrenamiento)
    ]

    # Ordenar las distancias y obtener las k más cercanas
    distancias_ordenadas = sorted(distancias, key=lambda x: x[0])
    k_vecinos_etiquetas = [etiqueta for _, etiqueta in distancias_ordenadas[:k]]

    # Encontrar la clase más común
    votacion = Counter(k_vecinos_etiquetas)
    prediccion = votacion.most_common(1)[0][0]

    return prediccion

# --- Nueva Función de "Entrenamiento" ---

def entrenar_modelo_al_inicio(file_path: str) -> dict:
    """
    Esta función se ejecuta una vez al iniciar la API.
    Carga el CSV, lo pre-procesa, y devuelve los artefactos
    necesarios para la predicción (imputer, scaler, X_train, y_train).
    """
    print(f"Iniciando carga y entrenamiento desde {file_path}...")
    
    # --- 1. Carga y Definición de Features ---
    try:
        df_original = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo en {file_path}")
        raise
        
    feature_cols = [
        'age', 'income_bracket', 'party_id_strength', 'tv_news_hours', 
        'social_media_hours', 'trust_media', 'civic_participation'
    ]
    target_col = 'intended_vote'

    df_model = df_original[feature_cols + [target_col]].copy()

    # --- 2. Pre-procesamiento de Datos ---
    print("Pre-procesando datos...")
    
    # Limpiar el target (y)
    df_model = df_model[df_model[target_col] != 'Undecided']
    df_model = df_model.dropna(subset=[target_col])

    # Convertir el target a números y guardar el mapa
    target_labels = df_model[target_col].astype('category')
    # Convertimos el mapa de categorías a un dict estándar
    target_map = dict(enumerate(target_labels.cat.categories))
    y = target_labels.cat.codes.values

    # Separar Features (X)
    X = df_model[feature_cols]

    # --- 3. División de Datos (Correcta) ---
    # Dividimos ANTES de imputar/escalar para evitar "data leakage"
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- 4. Crear y "Entrenar" Pre-procesadores ---
    # 4.1. Imputador: Se ajusta (fit) SOLO con X_train_raw
    imputer = SimpleImputer(strategy='median')
    imputer.fit(X_train_raw)
    
    # Transformamos X_train (y X_test para evaluación)
    X_train_imputed = imputer.transform(X_train_raw)
    X_test_imputed = imputer.transform(X_test_raw)

    # 4.2. Escalador: Se ajusta (fit) SOLO con X_train_imputed
    scaler = MinMaxScaler()
    scaler.fit(X_train_imputed)
    
    # Transformamos X_train (y X_test para evaluación)
    X_train_scaled = scaler.transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)

    print("Pre-procesadores (imputer, scaler) ajustados.")

    # --- 5. (Opcional) Evaluación del Modelo ---
    print(f"Evaluando el modelo k-NN (k=8) con los datos de test...")
    predicciones = []
    for votante_prueba in X_test_scaled:
        pred = predecir_votante(X_train_scaled, y_train, votante_prueba, k=8)
        predicciones.append(pred)

    aciertos = np.sum(predicciones == y_test)
    precision = aciertos / len(y_test)
    print(f"Precisión (Accuracy) del modelo: {precision * 100:.2f}%")

    # --- 6. Devolver Artefactos ---
    # Devolvemos todo lo que la API necesita para predecir
    print("¡Entrenamiento completado! Modelo listo.")
    return {
        "imputer": imputer,
        "scaler": scaler,
        "X_train": X_train_scaled, # El X_train escalado
        "y_train": y_train,        # El y_train
        "target_map": target_map,
        "feature_cols": feature_cols
    }