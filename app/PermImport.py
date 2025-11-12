import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import permutation_importance

# === 1) Preparación  ===
df = pd.read_csv("data/voter_intentions_3000.csv")
target_col = "intended_vote"
df = df.dropna(subset=[target_col])
df = df[df[target_col] != "Undecided"]

X = df.drop(columns=[target_col]).copy()
y = df[target_col].astype("category").cat.codes

continuas = ["age", "household_size", "refused_count", "tv_news_hours", "social_media_hours", "job_tenure_years"]
ordinales = ["gender", "education", "employment_status", "employment_sector", "income_bracket", "marital_status", "has_children", "urbanicity", "region", "voted_last", "party_id_strength", "union_member", "public_sector", "home_owner", "small_biz_owner", "owns_car", "wa_groups", "attention_check", "will_turnout", "undecided", "preference_strength", "survey_confidence", "trust_media", "civic_participation"]
nominales_texto = ["primary_choice","secondary_choice"]

pipe_cont = Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc",  MinMaxScaler())])

pipe_ord  = Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc",  MinMaxScaler())])

pipe_nom  = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

pre = ColumnTransformer([
    ("cont", pipe_cont, continuas),
    ("ord",  pipe_ord,  ordinales),
    ("nom",  pipe_nom,  nominales_texto),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === 2) Ajustamos el preprocesador con Train y transformamos ===
pre.fit(X_train)
X_train_tx = pre.transform(X_train)
X_test_tx  = pre.transform(X_test)

# === 3) Entrena un estimador que reciba matrices transformadas ===
clf = KNeighborsClassifier(n_neighbors=8, weights="distance")
clf.fit(X_train_tx, y_train)

# === 4) Nombres de features transformadas ===
feat_names = pre.get_feature_names_out()

# === 5) Aplicamos Permutation Importance en el ESPACIO TRANSFORMADO ===
pi = permutation_importance(
    clf, X_test_tx, y_test,
    n_repeats=20, random_state=42, scoring="f1_macro", n_jobs=-1
)

pi_df = pd.DataFrame({
    "feature_transformed": feat_names,
    "importance_mean": pi.importances_mean,
    "importance_std":  pi.importances_std
})

# === 6) Agrupar por columna original ===
def base_col(name: str) -> str:
    if "__" in name:
        _, rest = name.split("__", 1)
        # Si viene de OHE tendrá patrón col_categoria: tomamos solo 'col'
        for nom in nominales_texto:
            if rest.startswith(nom + "_"):
                return nom
        return rest  # cont/ord mantienen el nombre de la columna
    return name

pi_df["base"] = pi_df["feature_transformed"].map(base_col)

pi_group = (pi_df.groupby("base", as_index=False)["importance_mean"]
                 .sum()
                 .sort_values("importance_mean", ascending=False))

print("\n=== Permutation Importance (agrupado por columna original) ===")
print(pi_group.to_string(index=False))
