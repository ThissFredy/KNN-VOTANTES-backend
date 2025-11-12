import pandas as pd
import numpy as np
from textwrap import shorten

CATEG_NUM_MAX_UNIQUE = 30  
PRINT_MAX_VALUES = 30  


df = pd.read_csv("data/voter_intentions_3000.csv")
target_col = "intended_vote"

#--- Features del dataset ---
cols = list(df.columns)
print("\n=== LISTADO DE FEATURES ({} columnas) ===".format(len(cols)))
for i, c in enumerate(cols, 1):
    print(f"{i:>3}. {c}")


#--- Opciones por feature ---
print("\n=== OPCIONES DE RESPUESTA POR FEATURE ===")
print(f"{'Feature':<25} | {'Tipo':<12} | {'#Únicos':>8} | Valores (preview)")
print("-" * 120)

for c in cols:
    s = df[c]

    # determinar tipo “categórica” vs “continua” 
    dt = s.dtype
    is_cat_like = (
        pd.api.types.is_bool_dtype(dt)
        or pd.api.types.is_categorical_dtype(dt)
        or pd.api.types.is_object_dtype(dt)
        or ((pd.api.types.is_integer_dtype(dt) or pd.api.types.is_float_dtype(dt)) and s.nunique(dropna=True) <= CATEG_NUM_MAX_UNIQUE)
    )

    # únicos y orden “amigable”
    uniques = pd.unique(s)
    nums = [u for u in uniques if pd.api.types.is_number(u)]
    nonnums = [u for u in uniques if not pd.api.types.is_number(u)]
    try:
        nums_sorted = sorted(nums)
    except Exception:
        nums_sorted = [str(x) for x in nums]
    nonnums_sorted = sorted(nonnums, key=lambda z: str(z))
    ordered = nums_sorted + nonnums_sorted

    total = len(ordered)
    shown = ordered[:PRINT_MAX_VALUES]

    # formatear valores (NaN como <NaN>)
    vals = []
    for v in shown:
        if pd.isna(v):
            vals.append("<NaN>")
        else:
            vals.append(str(v))
    preview = ", ".join(vals)
    if total > PRINT_MAX_VALUES:
        preview += f", … (+{total - PRINT_MAX_VALUES} más)"

    preview_short = shorten(preview, width=400, placeholder=" …")

    print(f"{c:<25} | {'Categórica' if is_cat_like else 'Continua':<12} | {s.nunique(dropna=True):>8} | {preview_short}")

print("\nNota: <MISSING> indica valores faltantes; se muestran hasta", PRINT_MAX_VALUES, "valores únicos por columna.")