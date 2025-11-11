import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

# --- Tus Funciones de k-NN (sin cambios) ---
def calcular_distancia_euclidiana(punto_a, punto_b):
    return np.sqrt(np.sum((punto_a - punto_b)**2))

def predecir_votante(X_entrenamiento, y_entrenamiento, nuevo_votante, k=8):
    distancias = [
        (calcular_distancia_euclidiana(votante_entrenamiento, nuevo_votante), y_entrenamiento[i])
        for i, votante_entrenamiento in enumerate(X_entrenamiento)
    ]
    distancias_ordenadas = sorted(distancias, key=lambda x: x[0])
    k_vecinos_etiquetas = [etiqueta for _, etiqueta in distancias_ordenadas[:k]]
    votacion = Counter(k_vecinos_etiquetas)
    prediccion, conteo = votacion.most_common(1)[0]
    
    confianza = conteo / k
    
    return prediccion, confianza

# --- Función de "Entrenamiento" CORREGIDA ---
def entrenar_modelo_al_inicio(file_path: str) -> dict:
    """
    Esta función se ejecuta una vez al iniciar la API.
    Carga el CSV, lo pre-procesa, y devuelve los artefactos necesarios.
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
        'social_media_hours', 'trust_media', 'civic_participation', 'primary_choice',
        'secondary_choice'
    ]
    target_col = 'intended_vote'

    df_model = df_original[feature_cols + [target_col]].copy()

    # --- 2. Pre-procesamiento de Datos ---
    print("Pre-procesando datos...")
    
    # Limpiar el target (y)
    df_model = df_model[df_model[target_col] != 'Undecided']
    df_model = df_model.dropna(subset=[target_col])

    # Convertir el target a números y guardar el mapa (como ya lo tenías)
    target_labels = df_model[target_col].astype('category')
    target_map = dict(enumerate(target_labels.cat.categories))
    y = target_labels.cat.codes.values

    # --- NUEVO PASO: MANEJO DE CATEGÓRICAS EN FEATURES ---
    # Identificar cuáles son las columnas categóricas
    categorical_cols = ['primary_choice', 'secondary_choice']
    numerical_cols = [col for col in feature_cols if col not in categorical_cols]
    
    # Crear un diccionario para guardar los mapas de conversión de cada feature categórico
    feature_category_maps = {}

    # Convertir cada columna categórica a números y guardar su mapa
    for col in categorical_cols:
        df_model[col] = df_model[col].astype('category')
        # Guardamos el mapa de categoría -> código
        feature_category_maps[col] = {category: code for code, category in enumerate(df_model[col].cat.categories)}
        # Reemplazamos la columna original por sus códigos numéricos
        df_model[col] = df_model[col].cat.codes

    X = df_model[feature_cols]

    # --- 3. División de Datos (Correcta) ---
    # Ahora sí, podemos dividir porque X es numérico
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- 4. Crear y "Entrenar" Pre-procesadores ---
    # 4.1. Imputador: Ahora funciona porque los datos son numéricos
    imputer = SimpleImputer(strategy='median')
    imputer.fit(X_train_raw)
    
    X_train_imputed = imputer.transform(X_train_raw)
    X_test_imputed = imputer.transform(X_test_raw)

    # 4.2. Escalador
    scaler = MinMaxScaler()
    scaler.fit(X_train_imputed)
    
    X_train_scaled = scaler.transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)

    print("Pre-procesadores (imputer, scaler) ajustados.")

    # --- 5. Evaluación del Modelo ---
    print(f"Evaluando el modelo k-NN (k=8) con los datos de test...")
    predicciones = []
    confianzas = []
    for votante_prueba in X_test_scaled:
        pred, confianza = predecir_votante(X_train_scaled, y_train, votante_prueba, k=8)
        predicciones.append(pred)
        confianzas.append(confianza)

    #accuracy
    accuracy = accuracy_score(y_test, predicciones)
    print(f"Precisión (Accuracy) del modelo: {accuracy * 100:.2f}%")
    #F1 Score y Reporte
    print("\n--- Reporte de Clasificación ---")
    
    target_names_list = list(target_map.values())
    report = classification_report(
        y_test, 
        predicciones, 
        target_names=target_names_list
    )
    print(report)
    
    confianza_promedio = np.mean(confianzas)
    print(f"\nConfianza promedio del modelo en las predicciones de test: {confianza_promedio:.2f}")


    # --- 6. Devolver Artefactos ---
    # Devolvemos todo lo que la API necesita para predecir, INCLUYENDO los mapas de categorías
    print("¡Entrenamiento completado! Modelo listo.")
    return {
        "imputer": imputer,
        "scaler": scaler,
        "X_train": X_train_scaled,
        "y_train": y_train,
        "target_map": target_map,
        "feature_cols": feature_cols,
        "feature_category_maps": feature_category_maps 
    }