import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score

class Counter:
    """Contador mínimo con un método most_common(n) similar al de collections.Counter."""
    def __init__(self, iterable=None):
        self.counts = {}
        self._order = []
        if iterable:
            for item in iterable:
                if item not in self.counts:
                    self.counts[item] = 0
                    self._order.append(item)
                self.counts[item] += 1

    def most_common(self, n=None):
        # Ordena por frecuencia descendente; en empates respeta el orden de primera aparición.
        order_index = {v: i for i, v in enumerate(self._order)}
        items = list(self.counts.items())
        items.sort(key=lambda x: (-x[1], order_index.get(x[0], 0)))
        if n is None:
            return items
        return items[:n]

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

    # X sin target
    X_df = df.drop(columns=[target_col])
    
    # Guardamos los nombres de las columnas procesadas
    feature_cols = list(X_df.columns)
    
    # Hallar y (y convertir a códigos numéricos)
    y_labels = df[target_col].astype("category")
    target_map = dict(enumerate(y_labels.cat.categories))
    y = y_labels.cat.codes.values # y (array de numpy con los códigos: 0, 1, 2...)

    # Separar los datos para evaluación
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df, y, test_size=0.2, random_state=42, stratify=y
    )

    # Convertir a NumPy para la función manual
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
        "X_train": X_train_np,
        "y_train": y_train,
        "target_map": target_map,
        "feature_cols": feature_cols,
        "metrics": {
            "accuracy": acc,
            "f1_score_macro": f1_macro,
        }
    }