# 🚀 Informe de Metodología: Preprocesamiento y Creación del Modelo k-NN

## 1. Introducción: El Problema de los Dos Modelos

Este script es el núcleo de la preparación de datos del proyecto. Su objetivo es tomar el dataset crudo y ruidoso (`voter_intentions_3000.csv`) y convertirlo en un archivo limpio, optimizado y listo para producción (`voter_intentions_COMPLETED_PROCESSED.csv`).

Para lograr esto, el script resuelve **dos problemas de Machine Learning distintos** en secuencia:

1.  **Problema de Propagación:** El dataset original tiene un 75% de etiquetas faltantes ("Undecided"). Debemos usar el 25% de datos "conocidos" (755 filas) para "adivinar" (propagar) las etiquetas del 75% restante.
2.  **Problema de Predicción:** Una vez que tenemos un dataset completo (3000 filas), debemos optimizarlo (seleccionando solo las features útiles) y encontrar el mejor hiperparámetro (`k`) para el modelo final que se usará en la API.

---

## 2. Fase 1: Carga y Definición de Pipelines (Pasos 1-4)

### Qué se hace

Se carga el dataset crudo (`voter_intentions_3000.csv`) y se definen todas las columnas por tipo: `continuas` (ej. `age`), `ordinales` (ej. `education`) y `nominales_texto` (ej. `primary_choice`).

Luego, se crea un `preprocessor` de `sklearn` que define una "receta" de limpieza para cada tipo de dato.

### El Porqué (La Razón)

El dataset crudo no puede ser usado directamente por un modelo k-NN. El modelo fallaría porque:

1.  **Contiene Texto:** k-NN necesita números para calcular distancias. No puede restar `"CAND_Azon"` de `"CAND_Gaia"`.
2.  **Contiene `NaN` (Datos Faltantes):** La matemática de la distancia (ej. `50 - NaN`) falla con valores nulos.
3.  **Tiene Escalas Diferentes:** Una feature como `age` (rango 18-84) dominaría injustamente a una feature como `public_sector` (rango 0-1) en el cálculo de la distancia.

### Conceptos Matemáticos Clave

-   **Imputación (Mediana/Moda):** Se usa la **Mediana** (valor central) para las columnas numéricas porque es robusta a _outliers_ (valores atípicos). Se usa la **Moda** (valor más frecuente) para las columnas de texto.
-   **Escalado (MinMaxScaler):** Normaliza todas las features numéricas al mismo rango [0, 1] para que contribuyan de forma justa a la distancia.
    $X_{\text{scaled}} = \frac{X - X_{\text{min}}}{X_{\text{max}} - X_{\text{min}}}$
-   **One-Hot Encoding (OHE):** Convierte una columna nominal (ej. `primary_choice`) en `N` columnas binarias (0 o 1). Esto es crucial porque evita crear un orden numérico falso (ej. `Azon=1`, `Gaia=2` implicaría falsamente que `Gaia` es "mayor" que `Azon`).

---

## 3. Fase 2: Hallar K para Propagación y Etiquetado (Paso 5)

### Qué se hace

Se busca el valor `k` óptimo para la **tarea de propagación**.

1.  Se divide el 25% de datos "conocidos" (`df_known`, 755 filas) en un set de entrenamiento (`X_train_known`) y uno de validación (`X_test_known`).
2.  Se aplica el `preprocessor` (Imputación, Escalado, OHE) a ambos sets.
3.  Se itera `k` de 1 a 20, entrenando un `KNeighborsClassifier` en el set de entrenamiento y midiendo su **F1-Score (Macro)** en el set de validación.
4.  El `k` con el F1-Score más alto (`k=19`) se usa para entrenar un modelo final con _todos_ los datos conocidos (`X_known`, 755 filas).
5.  Este modelo final predice las etiquetas para las 2245 filas "desconocidas" (`X_unknown`).

### El Porqué (La Razón)

No podemos asumir que un `k` arbitrario (como `k=8`) es el mejor. Esta tarea es difícil (pocos datos) y crítica. Encontrar el `k` óptimo nos da la mayor confianza de que las etiquetas que estamos "inventando" para los indecisos son lo más precisas posible.

-   **Resultado de Consola:** `Mejor K (para propagación) encontrado: K=19 con F1-macro=0.8931`

### Conceptos Matemáticos Clave

-   **Métrica (F1-Score Macro):** Se elige esta métrica sobre la "Accuracy" porque nuestro dataset de 755 filas es **desbalanceado** (algunos candidatos tienen más votantes que otros).
    -   $Precisión = \frac{\text{Verdaderos Positivos}}{\text{Todos los Positivos Predichos}}$
    -   $Recall = \frac{\text{Verdaderos Positivos}}{\text{Todos los Positivos Reales}}$
    -   $F1 = 2 \times \frac{\text{Precisión} \times \text{Recall}}{\text{Precisión} + \text{Recall}}$
    -   El **"Macro"** calcula el F1-Score para cada candidato por separado y luego toma el promedio simple. Esto asegura que el rendimiento en candidatos minoritarios es tan importante como en los mayoritarios.

---

## 4. Fase 3: Selección de Features Relevantes (Pasos 6-7)

### Qué se hace

Se unen los datos "conocidos" (755) y los "predichos" (2245) para crear `df_completed` (3000 filas). Luego, se ejecuta una **Permutación por Importancia** sobre este dataset completo para descubrir qué features son _realmente_ útiles.

### El Porqué (La Razón)

El `df_completed` es una simulación de nuestro dataset de producción. Ahora el problema ha cambiado: ya no es predecir "Indeciso", sino predecir _entre 10 candidatos_. Necesitamos saber qué features contienen "señal" (información útil) y cuáles son "ruido" (información inútil o perjudicial).

### Conceptos Matemáticos Clave

-   **Permutation Importance:** Un método robusto para medir la utilidad de una feature.
    1.  El modelo calcula el F1-Score base (ej. 0.90) en el set de prueba.
    2.  Luego, "baraja" (permuta) aleatoriamente solo una columna (ej. `age`) y vuelve a calcular el F1-Score (ej. 0.88).
    3.  $Importancia_{\text{age}} = 0.90 - 0.88 = 0.02$ (Es útil).
    4.  Si el score _empeora_ (ej. 0.87), la importancia es positiva.
    5.  Si el score _mejora_ (ej. 0.91), la feature es _perjudicial_ (importancia negativa).
-   **Resultado de Consola:** El análisis (`=== Importancia de Features ===`) mostró que **28 de las 31** features tenían un impacto positivo. `primary_choice` fue la más importante (0.806), mientras que 3 features (`urbanicity`, `will_turnout`, `region`) resultaron ser perjudiciales (negativas) y, por lo tanto, se descartan.

---

## 5. Fase 4: Creación del Dataset Final (Paso 8)

### Qué se hace

Se toma la lista de 28 features positivas. Se crea un `final_preprocessor` que solo procesa esas 28 features. Este preprocesador se usa para transformar las 3000 filas del `df_completed` y el resultado se guarda como `voter_intentions_COMPLETED.csv`.

### El Porqué (La Razón)

Este es el **artefacto de producción**. Es el archivo final, 100% limpio (sin NaNs), procesado (OHE y escalado) y optimizado (solo features útiles) que nuestra API cargará en memoria. Esto hace que la API sea extremadamente rápida, ya que no tiene que hacer ningún preprocesamiento en vivo.

-   **Resultado de Consola:** `El shape del archivo es: (3000, 47)`.
    -   Esto significa que nuestras 28 features "crudas" seleccionadas se convirtieron en 46 columnas "procesadas" (después de OHE) + 1 columna de target.

---

## 6. Fase 5: Hallar K Óptimo para Producción (Paso 9)

### Qué se hace

Ahora que tenemos el dataset de producción (`...COMPLETED.csv`), cargamos _ese_ archivo y ejecutamos un segundo "Método del Codo" (probar k de 1 a 20) sobre él.

### El Porqué (La Razón)

El "terreno" ha cambiado. El primer `k` (k=19) se encontró en un dataset _diferente_ (755 filas, 30+ features). Debemos encontrar el `k` que sea óptimo para los **datos exactos que usará la API** (3000 filas, 46 features). Este es el hiperparámetro final y validado para nuestro modelo.

### Conceptos Matemáticos Clave

-   **Método del Codo / Meseta:** Al graficar el F1-Score contra `k`, buscamos el "codo" (la última subida significativa) o el inicio de la "meseta" (donde el score se estabiliza).
-   **Resultado de Consola:** `Mejor K encontrado: K=19 con F1-score=0.9120`.
    -   Este resultado confirma que `k=19` es el valor más robusto y preciso para el modelo final, logrando un F1-Score (macro) extremadamente alto de **91.2%**. Este será el `k` que usaremos en producción.
