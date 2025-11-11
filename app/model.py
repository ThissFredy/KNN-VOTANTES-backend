import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def calcular_distancia_euclidiana(punto_a, punto_b):
    return np.sqrt(np.sum((punto_a - punto_b)**2))

def predecir_votante(X_entrenamiento, y_entrenamiento, nuevo_votante, k=8):
    distancias = [
        (calcular_distancia_euclidiana(votante_entrenamiento, nuevo_votante), y_entrenamiento[i])
        for i, votante_entrenamiento in enumerate(X_entrenamiento)
    ]
    distancias_ordenadas = sorted(distancias, key=lambda x: x[0])
    k_vecinos = [et for _, et in distancias_ordenadas[:k]]
    pred, conteo = Counter(k_vecinos).most_common(1)[0]
    return pred, conteo / k

# --- Entrenamiento con preprocesamiento ---
def entrenar_modelo_al_inicio(file_path: str) -> dict:
    print(f"Iniciando carga y entrenamiento desde {file_path}...")
    df = pd.read_csv(file_path)

    # 1) Definir columnas base y el target
    target_col = "intended_vote"
    base_cols = [
        "age",                      # continua
        "job_tenure_years",         # continua
        "social_media_hours",       # continua
        "public_sector",            # ordinal
        "gender",                   # ordinal 
        "trust_media",              # ordinal 
        "civic_participation",      # ordinal/frecuencia
        "primary_choice",           # nominal (texto)
        "secondary_choice",         # nominal (texto)
    ]

    # 2) Filtramos el target utilizable
    df = df.dropna(subset=[target_col])
    df = df[df[target_col] != "Undecided"]

    # 3) Definir tipos por bloque para preprocesamiento
    continuas = ["age", "job_tenure_years", "social_media_hours"]
    ordinales = ["public_sector", "gender", "trust_media", "civic_participation"]
    nominales_texto = ["primary_choice", "secondary_choice"]

    # 4) Construir X/y
    X = df[base_cols].copy()
    y_labels = df[target_col].astype("category")
    target_map = dict(enumerate(y_labels.cat.categories))
    y = y_labels.cat.codes.values

    # -------------------------------
    # 5) Pipelines por tipo de variable
    # -------------------------------
    # Continuas: imputar mediana + escalar [0,1]
    pipe_cont = Pipeline(steps=[
        ("imp", SimpleImputer(strategy="median")),
        ("sc", MinMaxScaler())
    ])

    # Ordinales: imputar mediana + escalar [0,1]
    pipe_ord = Pipeline(steps=[
        ("imp", SimpleImputer(strategy="median")),
        ("sc", MinMaxScaler())
    ])

    # Nominales texto: imputar moda con OneHotEncoding 
    pipe_nom = Pipeline(steps=[
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    pre = ColumnTransformer(
        transformers=[
            ("cont", pipe_cont, continuas),
            ("ord",  pipe_ord, ordinales),
            ("nom",  pipe_nom, nominales_texto),
        ],
        remainder="drop"
    )

    # Ajustar preprocesamiento con TRAIN solamente
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pre.fit(X_train_raw)

    # Transformar
    X_train_prep = pre.transform(X_train_raw)
    X_test_prep  = pre.transform(X_test_raw)

    # -------------------------------
    # 6) Evaluación del KNN 
    # -------------------------------
    preds, confs = [], []
    for fila in X_test_prep:
        p, c = predecir_votante(X_train_prep, y_train, fila, k=8)
        preds.append(p); confs.append(c)

    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc*100:.2f}%")

    nombres = list(target_map.values())
    print(classification_report(y_test, preds, target_names=nombres))
    print(f"Confianza promedio: {np.mean(confs):.2f}")

    # Para depurar o servir: nombres de columnas finales
    try:
        feature_names = pre.get_feature_names_out()
    except:
        feature_names = None

    # Mapas de categorías para nominales guardados
    ohe = pre.named_transformers_["nom"].named_steps["ohe"]
    ohe_categories = {col: list(cats) for col, cats in zip(nominales_texto, ohe.categories_)}

    print("¡Entrenamiento completado! Preprocesamiento listo.")
    return {
        "preprocessor": pre,                
        "X_train": X_train_prep,            
        "y_train": y_train,
        "target_map": target_map,
        "feature_names_out": feature_names,
        "ohe_categories": ohe_categories,
        "config": {
            "continuas": continuas,
            "ordinales": ordinales,
            "nominales_texto": nominales_texto,
            "k": 8
        }
    }
