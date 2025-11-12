import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import permutation_importance

print("--- INICIANDO PROCESO DE PROPAGACIÓN DE ETIQUETAS ---")

# === 1) Carga y Definición de Columnas ===
print("\n[Paso 1/8] Cargando datos y definiendo tipos de columnas...")
try:
    df = pd.read_csv("data/voter_intentions_3000.csv")
except FileNotFoundError:
    print("Error: No se encontró 'data/voter_intentions_3000.csv'")
    exit()

target_col = "intended_vote"

# Basado en tu primer análisis de importancia (Script 1)
# Definimos TODAS las columnas que podrían ser útiles
continuas = [
    "age", "household_size", "refused_count", "tv_news_hours", 
    "social_media_hours", "job_tenure_years"
]
ordinales = [
    "gender", "education", "employment_status", "employment_sector", 
    "income_bracket", "marital_status", "has_children", "urbanicity", 
    "region", "voted_last", "party_id_strength", "union_member", 
    "public_sector", "home_owner", "small_biz_owner", "owns_car", 
    "wa_groups", "attention_check", "will_turnout", 
    "preference_strength", "survey_confidence", "trust_media", 
    "civic_participation"
]
nominales_texto = ["primary_choice", "secondary_choice"]

print(f"\n[Paso 3/8] Dividiendo el dataset (Total: {len(df)} filas)...")
df_known = df[df[target_col] != "Undecided"].copy()
df_unknown = df[df[target_col] == "Undecided"].copy()

print(f"  -> Datos Conocidos (para entrenar): {len(df_known)} filas")
print(f"  -> Datos Indecisos (para predecir): {len(df_unknown)} filas")

# === 4) Definición del Pipeline de Preprocesamiento ===
print("\n[Paso 4/8] Creando pipeline de preprocesamiento (Imputación + Escalado)...")

# Pipeline para Continuas: Imputar con Mediana, Escalar [0,1]
pipe_cont = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("sc",  MinMaxScaler())
])

# Pipeline para Ordinales: Imputar con Mediana, Escalar [0,1]
pipe_ord  = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("sc",  MinMaxScaler())
])

# Pipeline para Nominales: Imputar con Moda, luego One-Hot Encoding
pipe_nom  = Pipeline([
    ("imp", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

# ColumnTransformer une todos los pipelines
preprocessor = ColumnTransformer([
    ("cont", pipe_cont, continuas),
    ("ord",  pipe_ord,  ordinales),
    ("nom",  pipe_nom,  nominales_texto),
], remainder="drop")

# === 5) Entrenamiento de k-NN y Predicción de Indecisos ===
print("\n[Paso 5/8] Entrenando k-NN para asignar etiquetas a los 'Indecisos'...")

# Combinamos el preprocesador y el clasificador en un solo Pipeline
model_pipeline = Pipeline([
    ("pre", preprocessor),
    ("clf", KNeighborsClassifier(n_neighbors=8))
])

# Definimos X e y de los datos CONOCIDOS
X_known = df_known.drop(columns=[target_col])
y_known = df_known[target_col]

# Entrenamos el pipeline COMPLETO con los datos CONOCIDOS
model_pipeline.fit(X_known, y_known)
print("  -> Modelo k-NN entrenado con datos conocidos.")

# Definimos X de los datos INDECISOS
X_unknown = df_unknown.drop(columns=[target_col])

# Predecimos las etiquetas para los indecisos
predicted_labels = model_pipeline.predict(X_unknown)
print(f"  -> Etiquetas predichas para {len(predicted_labels)} votantes indecisos.")

# === 6) Reconstrucción del Dataset ===
print("\n[Paso 6/8] Reconstruyendo el dataset con las nuevas etiquetas...")

# Asignamos las etiquetas predichas a los indecisos
df_unknown_predicted = df_unknown.copy()
df_unknown_predicted[target_col] = predicted_labels

# Unimos los dataframes
df_completed = pd.concat([df_known, df_unknown_predicted]).reset_index(drop=True)
print(f"  -> Dataset completado, total de filas: {len(df_completed)}")
print(f"  -> Nuevas categorías de 'intended_vote':\n{df_completed[target_col].value_counts()}")

# === 7) Permutación por Importancia (En dataset completado) ===
print("\n[Paso 7/8] Calculando importancia de features en el dataset COMPLETADO...")

# Dividimos el dataset COMPLETO para validación
X_final = df_completed.drop(columns=[target_col])
y_final = df_completed[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X_final, y_final, test_size=0.2, random_state=42, stratify=y_final
)

# Re-entrenamos el pipeline con la división de datos COMPLETA
model_pipeline.fit(X_train, y_train)
print("  -> Modelo k-NN final entrenado.")

# Calculamos la importancia en el set de Test
pi = permutation_importance(
    model_pipeline, X_test, y_test,
    n_repeats=10, random_state=42, scoring="f1_macro", n_jobs=-1
)
print("  -> Permutación por importancia completada.")
print("  -> Procesando resultados de importancia...")

# Los nombres de las features son las columnas originales de X_test
original_feature_names = X_test.columns

# Creamos el DataFrame final. Ya está agrupado por la feature original.
pi_group = pd.DataFrame({
    "base": original_feature_names,
    "importance_mean": pi.importances_mean,
}).sort_values("importance_mean", ascending=False)

print("\n=== Importancia de Features (Dataset Final) ===")
print(pi_group.to_string(index=False))

# === 8) Guardado del Dataset Final (PROCESADO) ===
print("\n[Paso 8/8] Procesando y guardando el dataset final...")

# 1. Creamos un NUEVO preprocesador solo con las features positivas
# (Esto es crucial para que el One-Hot Encoding funcione bien)
positive_features = pi_group[pi_group["importance_mean"] > 0]["base"].tolist()

# 2. Separamos las features positivas por tipo
final_cont = [col for col in continuas if col in positive_features]
final_ord  = [col for col in ordinales if col in positive_features]
final_nom  = [col for col in nominales_texto if col in positive_features]

# 3. Creamos el preprocesador final
# Usamos 'passthrough' para las features que no son OHE
# para que conserven sus nombres originales si son numéricas.
# OJO: Usamos los pipelines de imputación que ya definimos
final_preprocessor = ColumnTransformer([
    ("cont", pipe_cont, final_cont),
    ("ord",  pipe_ord,  final_ord),
    ("nom",  pipe_nom,  final_nom),
], remainder="drop") # Descartamos todo lo demás

print(f"Features seleccionadas: {positive_features}")

# 4. Ajustamos el preprocesador final y transformamos TODO el dataset
final_preprocessor.fit(df_completed)
data_processed = final_preprocessor.transform(df_completed)

# 5. Obtenemos los nombres de las columnas procesadas (ej. 'nom__primary_choice_CAND_A')
feature_names_out = final_preprocessor.get_feature_names_out()

# 6. Creamos el DataFrame procesado final (¡Este ya no tendrá NaNs!)
df_processed_clean = pd.DataFrame(
    data_processed, 
    columns=feature_names_out
)

# 7. Añadimos la columna 'target' de vuelta
# (Asegúrate de resetear el índice si hay problemas de alineación)
df_processed_clean[target_col] = df_completed[target_col].values

# 8. Guardamos el archivo final
output_path = "data/voter_intentions_COMPLETED_PROCESSED.csv"
df_processed_clean.to_csv(output_path, index=False)

print(f"\n¡PROCESO COMPLETADO!")
print(f"Dataset final procesado y limpio guardado en: {output_path}")
print(f"El shape del archivo es: {df_processed_clean.shape}")