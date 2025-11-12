# Informe de Práctica: Predicción de voto con K Vecinos Más Cercanos (KNN)

Autores: Fredy Alejandro Zarate - Juan David Rodriguez
Asignatura: Machine Learning

## 1. Resumen Ejecutivo

Este informe detalla el diseño, implementación y despliegue de un modelo de machine learning para predecir la intención de voto de electores. Ante el requerimiento de una campaña política de clasificar a nuevos votantes, se utilizó un conjunto de datos de 3000 encuestas previas. El desafío principal se centró en un desbalance extremo de la variable objetivo, donde el 74.83% de las etiquetas eran "Undecided".

Para resolver esto, se implementó un enfoque de propagación de etiquetas: primero, se entrenó un modelo KNN (de sklearn) utilizando el 25% de los datos "conocidos" para imputar las etiquetas del 75% "desconocido". Con un dataset completo de 3000 filas, se realizó un análisis de Permutation Importance para identificar las 5 características más predictivas. Finalmente, se entrenó y evaluó el modelo KNN de predicción final sobre este dataset limpio y filtrado, que luego fue contenedorizado (Docker) como un servicio API para el consumo de un frontend.

## 2. Introducción

En el contexto de las campañas políticas modernas, la capacidad de analizar datos para entender y predecir el comportamiento de los votantes es un activo estratégico fundamental. Las campañas buscan optimizar recursos y dirigir sus mensajes de manera más efectiva. Conocer de antemano la "intención de voto" de un electorado específico permite personalizar la comunicación y enfocar los esfuerzos en segmentos de la población (indecisos, afines, opositores) que maximicen el impacto.

Este proyecto surge de la necesidad de un aspirante a un cargo de elección popular de clasificar a nuevos electores. El equipo de campaña dispone de un dataset histórico y desea un sistema que, basado en las características personales y afinidades políticas de un individuo, pueda predecir su probable intención de voto.

La principal restricción técnica del proyecto es el uso obligatorio del algoritmo K-Vecinos Más Cercanos (KNN) para la clasificación.

## 3. Objetivos

### 3.1. Objetivo General

Diseñar e implementar un modelo predictivo de intención de voto basado en el algoritmo K-Vecinos Más Cercanos (KNN), desplegado como un servicio web escalable para el consumo de una aplicación frontend.

### 3.2. Objetivos Específicos

* Analizar y preprocesar el dataset voter_intentions_3000.csv para asegurar la calidad y compatibilidad de los datos con el modelo KNN.

* Implementar una estrategia de propagación de etiquetas usando un clasificador KNN (de sklearn) para imputar las 2245 etiquetas "Undecided".

* Imputar los valores NaN de las features usando la Mediana (para numéricas) y la Moda (para categóricas).

* Codificar variables categóricas de texto (como primary_choice) usando One-Hot Encoding (OHE).

* Realizar una selección de características post-imputación usando Permutation Importance para identificar las variables más relevantes.

* Entrenar, optimizar y evaluar un clasificador KNN final sobre el dataset de 3000 filas completado y filtrado.

* Desarrollar un servicio (API) en Python y una aplicación frontend, desplegando ambos en contenedores Docker independientes.

## 4. Contexto y Escenario

El proyecto se enmarca en una "Actividad Democrática" simulada.

Cliente: La campaña de un aspirante a un cargo de elección popular.

Problema: Identificar la afinidad de votación de nuevos electores.

Datos Disponibles: Un dataset (voter_intentions_3000.csv) con 3000 registros de votantes probados, incluyendo características demográficas, socioeconómicas, afinidades políticas y, crucialmente, su intención de voto (la variable objetivo).

Restricción Principal: El modelo de predicción final debe ser KNN.

Concesión: Se permite el uso de Regresión Lineal/Logística y Bosques Aleatorios únicamente para completar datos desconocidos (imputación) dentro del cuerpo de entrenamiento.

Requerimiento de Despliegue: Una arquitectura de microservicios, con el backend de ML y el frontend en contenedores separados.

Bibliotecas Permitidas: Pandas, Sklearn, Pyplot.

## 5. Metodología y Herramientas

### 5.1. Dataset

Se utilizó el archivo voter_intentions_3000.csv. Este conjunto de datos contiene 33 variables clsificadas y distribuidas de la siguiente forma:

#### Features Numéricas

| Variable           | NaN | Únicos | Rango         | Media | Mediana |
| ------------------ | --: | -----: | ------------- | ----: | ------: |
| age                | 154 |     67 | [18.0 – 84.0] | 51.04 |   51.00 |
| household_size     | 162 |      7 | [1.0 – 7.0]   |  3.96 |    4.00 |
| refused_count      | 154 |      6 | [0.0 – 5.0]   |  2.50 |    2.50 |
| tv_news_hours      | 141 |      6 | [0.0 – 5.0]   |  2.51 |    3.00 |
| social_media_hours | 152 |      6 | [0.0 – 5.0]   |  2.47 |    2.00 |
| job_tenure_years   | 151 |     41 | [0.0 – 40.0]  | 20.14 |   20.00 |

#### Features Categoricas

| Variable            | NaN | Únicos | Frecuencias (valor: conteo, %)                                                                                                                                                                    |
| ------------------- | --: | -----: | ---------------------------------------------------- |
| gender              | 155 |      3 | 1: 1408 (46.93%), 0: 1384 (46.13%), 2: 53 (1.77%), nan: 155 (5.17%)                                                                                                                               |
| education           | 156 |      6 | 0: 498 (16.60%), 1: 480 (16.00%), 2: 478 (15.93%), 5: 473 (15.77%), 3: 469 (15.63%), 4: 446 (14.87%), nan: 156 (5.20%)                                                                            |
| employment_status   | 160 |      6 | 3: 494 (16.47%), 0: 485 (16.17%), 4: 474 (15.80%), 1: 473 (15.77%), 2: 464 (15.47%), 5: 450 (15.00%), nan: 160 (5.33%)                                                                            |
| employment_sector   | 152 |      6 | 3: 514 (17.13%), 1: 484 (16.13%), 5: 474 (15.80%), 4: 467 (15.57%), 0: 462 (15.40%), 2: 447 (14.90%), nan: 152 (5.07%)                                                                            |
| income_bracket      | 148 |      5 | 0: 602 (20.07%), 1: 583 (19.43%), 3: 573 (19.10%), 2: 549 (18.30%), 4: 545 (18.17%), nan: 148 (4.93%)                                                                                             |
| marital_status      | 143 |      5 | 4: 584 (19.47%), 2: 575 (19.17%), 3: 568 (18.93%), 1: 567 (18.90%), 0: 563 (18.77%), nan: 143 (4.77%)                                                                                             |
| has_children        | 164 |      2 | 1: 1554 (51.80%), 0: 1282 (42.73%), nan: 164 (5.47%)                                                                                                                                              |
| urbanicity          | 153 |      3 | 1: 965 (32.17%), 0: 951 (31.70%), 2: 931 (31.03%), nan: 153 (5.10%)                                                                                                                               |
| region              | 153 |      5 | 0: 587 (19.57%), 1: 570 (19.00%), 2: 568 (18.93%), 4: 567 (18.90%), 3: 555 (18.50%), nan: 153 (5.10%)                                                                                             |
| voted_last          | 152 |      2 | 1: 1850 (61.67%), 0: 998 (33.27%), nan: 152 (5.07%)                                                                                                                                               |
| party_id_strength   | 147 |      5 | 3: 583 (19.43%), 1: 574 (19.13%), 0: 570 (19.00%), 4: 564 (18.80%), 2: 562 (18.73%), nan: 147 (4.90%)                                                                                             |
| union_member        | 154 |      2 | 0: 2412 (80.40%), 1: 434 (14.47%), nan: 154 (5.13%)                                                                                                                                               |
| public_sector       | 155 |      2 | 0: 2200 (73.33%), 1: 645 (21.50%), nan: 155 (5.17%)                                                                                                                                               |
| home_owner          | 143 |      2 | 0: 1580 (52.67%), 1: 1277 (42.57%), nan: 143 (4.77%)                                                                                                                                              |
| small_biz_owner     | 157 |      2 | 0: 2363 (78.77%), 1: 480 (16.00%), nan: 157 (5.23%)                                                                                                                                               |
| owns_car            | 157 |      2 | 0: 1491 (49.70%), 1: 1352 (45.07%), nan: 157 (5.23%)                                                                                                                                              |
| wa_groups           | 159 |      9 | 2: 759 (25.30%), 1: 749 (24.97%), 3: 483 (16.10%), 0: 408 (13.60%), 4: 270 (9.00%), 5: 122 (4.07%), 6: 38 (1.27%), 7: 9 (0.30%), 8: 3 (0.10%), nan: 159 (5.30%)                                   |
| attention_check     | 163 |      2 | 1: 2580 (86.00%), 0: 257 (8.57%), nan: 163 (5.43%)                                                                                                                                                |
| will_turnout        | 150 |      2 | 1: 1724 (57.47%), 0: 1126 (37.53%), nan: 150 (5.00%)                                                                                                                                              |
| undecided           | 135 |      2 | 1: 2152 (71.73%), 0: 713 (23.77%), nan: 135 (4.50%)                                                                                                                                               |
| preference_strength | 117 |     11 | 7: 282 (9.40%), 5: 280 (9.33%), 10: 277 (9.23%), 9: 275 (9.17%), 6: 270 (9.00%), 1: 265 (8.83%), 8: 255 (8.50%), 4: 251 (8.37%), 3: 246 (8.20%), 2: 242 (8.07%), 0: 240 (8.00%), nan: 117 (3.90%) |
| survey_confidence   | 157 |     11 | 9: 284 (9.47%), 10: 268 (8.93%), 6: 267 (8.90%), 0: 262 (8.73%), 2: 262 (8.73%), 1: 257 (8.57%), 3: 252 (8.40%), 8: 251 (8.37%), 5: 248 (8.27%), 7: 247 (8.23%), 4: 245 (8.17%), nan: 157 (5.23%) |
| trust_media         | 152 |     11 | 9: 288 (9.60%), 3: 270 (9.00%), 10: 264 (8.80%), 2: 259 (8.63%), 6: 259 (8.63%), 8: 258 (8.60%), 0: 257 (8.57%), 4: 256 (8.53%), 5: 254 (8.47%), 1: 250 (8.33%), 7: 233 (7.77%), nan: 152 (5.07%) |
| civic_participation | 159 |     11 | 4: 280 (9.33%), 2: 272 (9.07%), 6: 269 (8.97%), 3: 264 (8.80%), 9: 259 (8.63%), 0: 257 (8.57%), 5: 255 (8.50%), 7: 255 (8.50%), 8: 253 (8.43%), 1: 240 (8.00%), 10: 237 (7.90%), nan: 159 (5.30%) |

#### Features Nominales

| Variable         | NaN | Únicos | Frecuencias (categoría: conteo, %)                                                                                                                                                                                                                                                    |
| ---------------- | --: | -----: | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| primary_choice   |   0 |     10 | CAND_Gaia: 619 (20.63%), CAND_Azon: 496 (16.53%), CAND_Demetra: 434 (14.47%), CAND_Civico: 306 (10.20%), CAND_Electra: 255 (8.50%), CAND_Jade: 213 (7.10%), CAND_Icaro: 203 (6.77%), CAND_Frontera: 161 (5.37%), CAND_Boreal: 160 (5.33%), CAND_Halley: 153 (5.10%)                   |
| secondary_choice | 131 |     10 | CAND_Demetra: 350 (11.67%), CAND_Gaia: 336 (11.20%), CAND_Azon: 323 (10.77%), CAND_Civico: 302 (10.07%), CAND_Halley: 298 (9.93%), CAND_Electra: 289 (9.63%), CAND_Boreal: 260 (8.67%), CAND_Icaro: 252 (8.40%), CAND_Frontera: 251 (8.37%), CAND_Jade: 208 (6.93%), nan: 131 (4.37%) |
| intended_vote    |   0 |     11 | Undecided: 2245 (74.83%), CAND_Gaia: 160 (5.33%), CAND_Azon: 130 (4.33%), CAND_Demetra: 119 (3.97%), CAND_Civico: 77 (2.57%), CAND_Electra: 68 (2.27%), CAND_Jade: 47 (1.57%), CAND_Halley: 41 (1.37%), CAND_Icaro: 39 (1.30%), CAND_Frontera: 38 (1.27%), CAND_Boreal: 36 (1.20%)    |

Estas Features representan los siguientes tipos de datos

Datos Demográficos: age, gender, education, etc.

Datos Socioeconómicos: employment_status, income_bracket, home_owner, etc.

Afinidades Políticas: party_id_strength, voted_last, etc.

Variable Objetivo: intended_vote (con las clases: [CAND_A, CAND_B, Undecided, etc.]).

### 5.2. Herramientas de Software

Lenguaje: Python 3.x

Entorno de Desarrollo: Google Colab (para modelado) y VS Code (para servicios).

Análisis y Modelado:

Pandas: Para la carga, manipulación y limpieza de datos.

Sklearn (scikit-learn): Para el preprocesamiento (escalado, codificación), imputación (usando sus módulos de regresión/bosques), implementación de KNN (KNeighborsClassifier) y evaluación de métricas.

Pyplot (Matplotlib): Para la visualización de datos durante el análisis exploratorio y la evaluación del modelo.

Despliegue:

Flask / FastAPI: (O la herramienta seleccionada) para crear el servicio API del modelo.

Docker: Para la contenedorización del servicio de ML y del frontend.

Frontend: [Lenguaje y Framework seleccionados, ej: HTML/CSS/JavaScript, React, Angular]

6. Desarrollo y Análisis de Código

Esta sección presenta los fragmentos de código clave del proyecto, explicando su propósito y la justificación de su uso, como se solicitó.