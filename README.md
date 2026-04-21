# Predicción de Riesgo Metabólico — Brigada Médica

## Contexto general

Este notebook forma parte de un proyecto de análisis clínico derivado de una brigada médica comunitaria. Durante la brigada se recopilaron datos de 91 pacientes mediante el sistema ClinicaApp, incluyendo antecedentes patológicos, medicamentos activos y, en tres casos, signos vitales medidos directamente. El objetivo es identificar qué pacientes tienen mayor probabilidad de presentar síndrome metabólico o riesgo cardiovascular elevado, de modo que el equipo médico pueda priorizar el seguimiento y la intervención preventiva.

El modelo no realiza un diagnóstico clínico. Produce una probabilidad de riesgo que debe interpretarse como una herramienta de apoyo a la decisión médica, no como un sustituto del juicio clínico.

---

## Definición del riesgo metabólico

La etiqueta de riesgo alto se asigna cuando un paciente cumple cuatro o más de los siguientes criterios clínicos, derivados de los estándares internacionales IDF y ATP-III para síndrome metabólico:

- Glucosa en ayunas mayor o igual a 100 mg/dL
- Hemoglobina glucosilada (HbA1c) mayor o igual a 5.7%
- Presión sistólica mayor o igual a 130 mmHg o diastólica mayor o igual a 85 mmHg
- Índice de masa corporal (IMC) mayor o igual a 30 kg/m²
- Diagnóstico previo de diabetes mellitus
- Diagnóstico previo de hipertensión arterial
- Antecedente familiar de diabetes o cardiopatía
- Actividad física menor a 30 minutos por semana

Un paciente que cumple cuatro o más de estos criterios recibe la etiqueta `target = 1` (riesgo alto). Quien cumple tres o menos recibe `target = 0` (riesgo bajo o moderado). Esta clasificación binaria es la variable que el modelo aprende a predecir.

---

## Datos utilizados para el entrenamiento

El modelo se entrena exclusivamente sobre datos públicos y sintéticos. Los 91 pacientes de la brigada se mantienen completamente fuera del entrenamiento y se usan únicamente en la fase de predicción final. Esto evita que el modelo "memorice" a los pacientes de la brigada y genere probabilidades artificialmente optimistas.

| Conjunto de datos | Pacientes | Descripción |
|---|---|---|
| Sintético ENSANUT México | 5,000 | Generado con distribuciones epidemiológicas reales de la Encuesta Nacional de Salud y Nutrición de México. Incluye prevalencias de DM, HTA y obesidad propias de la población mexicana adulta. |
| PIMA Indians Diabetes (UCI) | 768 | Dataset real del repositorio de Machine Learning de la Universidad de California. Contiene mujeres indígenas Pima con seguimiento longitudinal de diabetes. |
| Diabetes Prediction Dataset (Kaggle, 100k) | hasta 8,000 | Dataset real con 100,000 pacientes. Si los servidores de descarga están disponibles, se carga una muestra de 8,000. Es el conjunto más importante para mejorar la precisión del modelo. |
| CDC BRFSS 2015 | hasta 10,000 | Sistema de vigilancia de factores de riesgo conductuales del Centro para el Control de Enfermedades de Estados Unidos. Datos reales con diagnósticos confirmados. |
| Brigada médica (solo inferencia) | 91 | Pacientes reales. Se usan únicamente para obtener predicciones finales, nunca para entrenar. |

**Nota sobre las variables continuas de la brigada**: solo tres pacientes de la brigada tienen glucosa, presión arterial e IMC medidos directamente. Para los 88 restantes, estas variables se imputan con la mediana del conjunto de entrenamiento, que es la práctica estándar en modelos clínicos cuando no se dispone de la medición. Las variables binarias (diagnóstico de DM, HTA, tabaquismo, antecedentes familiares) sí se extraen directamente del texto del expediente y de los medicamentos registrados.

---

## Arquitectura del modelo

El sistema utiliza tres modelos base combinados en un ensemble final.

**XGBoost** es un algoritmo de árboles de decisión con gradiente potenciado. Construye de forma secuencial un conjunto de árboles donde cada árbol nuevo corrige los errores del anterior. Es considerado el modelo de referencia para datos tabulares de tamaño mediano. Sus hiperparámetros se optimizan mediante Optuna, una biblioteca de optimización bayesiana que explora el espacio de configuraciones de forma inteligente.

**LightGBM** es una variante de gradient boosting que crece los árboles hoja por hoja en lugar de nivel por nivel, lo que le permite encontrar relaciones más específicas en los datos con menor tiempo de cómputo. En datos médicos tabulares suele obtener resultados competitivos con XGBoost.

**Red Neuronal** es una red densa con tres capas ocultas, normalización por lotes y regularización L2. Capta relaciones no lineales complejas que los árboles de decisión pueden pasar por alto. Es el modelo de apoyo dentro del ensemble.

**Stacking con meta-learner** es una técnica de combinación de segundo nivel. En lugar de promediar las predicciones de los tres modelos con pesos fijos, se entrena una regresión logística que aprende cuánto peso darle a cada modelo en función de sus predicciones. El entrenamiento del meta-learner se hace con validación cruzada de cinco particiones (out-of-fold) para evitar que el modelo de segundo nivel vea los datos con los que los modelos base fueron entrenados.

**Balanceo con SMOTE** (Synthetic Minority Oversampling Technique): como la clase de alto riesgo tiene menos pacientes que la clase de bajo riesgo, se generan ejemplos sintéticos de la clase minoritaria interpolando entre ejemplos reales cercanos en el espacio de características. SMOTE se aplica únicamente al conjunto de entrenamiento, nunca al de prueba.

---

## Fórmulas del modelo

**Probabilidad de riesgo (salida de la red neuronal)**

La red neuronal produce una probabilidad mediante la función sigmoide aplicada a la combinación lineal de la última capa oculta:

$$P(\text{riesgo alto}) = \sigma(z) = \frac{1}{1 + e^{-z}}$$

donde $z = W_n \cdot a_{n-1} + b_n$ es la salida de la última capa antes de la activación.

La función sigmoide garantiza que la salida esté en el intervalo (0, 1), interpretable directamente como probabilidad.

**Función de pérdida (binary cross-entropy)**

El entrenamiento minimiza la diferencia entre las probabilidades predichas y las etiquetas reales:

$$L = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{p}_i) + (1 - y_i) \log(1 - \hat{p}_i) \right]$$

donde $y_i$ es la etiqueta real (0 o 1) y $\hat{p}_i$ es la probabilidad predicha para el paciente $i$.

**Ensemble ponderado por AUC**

La probabilidad final del ensemble se calcula como una combinación lineal de las probabilidades de cada modelo, ponderada por su AUC individual:

$$P_{\text{final}} = w_{\text{XGB}} \cdot P_{\text{XGB}} + w_{\text{LGB}} \cdot P_{\text{LGB}} + w_{\text{NN}} \cdot P_{\text{NN}} + w_{\text{STK}} \cdot P_{\text{STK}}$$

$$w_i = \frac{\text{AUC}_i}{\sum_j \text{AUC}_j}$$

Un modelo con mayor AUC recibe mayor peso en la predicción final.

---

## Variables de entrada al modelo

| Variable | Tipo | Descripción |
|---|---|---|
| age | Numérica continua | Edad del paciente en años |
| gender | Binaria | 1 = masculino, 0 = femenino |
| glucose | Numérica continua | Glucosa en ayunas en mg/dL. Rango esperado: 60–400. |
| HbA1c | Numérica continua | Hemoglobina glucosilada en porcentaje. Rango esperado: 4–14. |
| systolic | Numérica continua | Presión arterial sistólica en mmHg |
| diastolic | Numérica continua | Presión arterial diastólica en mmHg |
| bmi | Numérica continua | Índice de masa corporal en kg/m² |
| hypertension | Binaria | 1 si existe diagnóstico previo de hipertensión arterial |
| diabetes_history | Binaria | 1 si existe diagnóstico previo de diabetes mellitus |
| family_history | Binaria | 1 si existe antecedente familiar de cardiopatía o DM |
| smoking | Binaria | 1 si el paciente es o fue fumador |
| physical_activity | Numérica continua | Minutos de actividad física por semana |

Adicionalmente se calculan variables derivadas (feature engineering) que capturan interacciones clínicas relevantes:

| Variable derivada | Descripción |
|---|---|
| glucemia_combinada | Producto de glucosa normalizada por HbA1c. Combina ambos marcadores glucémicos en una sola señal. |
| carga_cardiometabolica | Producto de presión sistólica normalizada por IMC normalizado. Refleja la carga combinada sobre el sistema cardiovascular. |
| edad_grupo | Categoría de riesgo por edad: 0 si menor de 45 años, 1 entre 45 y 59, 2 si mayor o igual a 60. El riesgo metabólico sube en escalones, no de forma lineal. |
| condiciones_confirmadas | Suma de los cuatro diagnósticos binarios: HTA + DM + antecedente familiar + tabaquismo. Un valor de 3 o 4 indica acumulación de factores de riesgo. |
| dm_hta_combo | Producto de diabetes e hipertensión. Vale 1 solo cuando ambas condiciones están presentes simultáneamente, que es la combinación de mayor riesgo cardiovascular. |
| edad_dm | Producto de la edad por el indicador de DM, normalizado. Captura que la diabetes en un paciente mayor de 55 años conlleva mayor riesgo que en uno joven. |
| obesidad_severa | Indicador binario de IMC mayor o igual a 35, umbral de obesidad clase II donde el riesgo metabólico aumenta de forma no lineal. |

---

## Métricas de evaluación

Todas las métricas se calculan sobre el conjunto de prueba, que contiene el 20% de los datos y que el modelo no vio durante el entrenamiento.

**AUC-ROC (Área bajo la curva ROC)**

Mide la capacidad del modelo para distinguir entre pacientes de alto y bajo riesgo independientemente del umbral de decisión. Un AUC de 0.50 equivale a una clasificación aleatoria. Un AUC de 1.00 equivale a clasificación perfecta. Para modelos clínicos de predicción de riesgo se considera aceptable por encima de 0.80 y bueno por encima de 0.90.

**Accuracy (Exactitud)**

Proporción de pacientes clasificados correctamente sobre el total. Es la métrica más intuitiva pero puede ser engañosa cuando las clases están desbalanceadas. Por ejemplo, si el 70% de los pacientes son de bajo riesgo, un modelo que siempre predice bajo riesgo tendría 70% de accuracy sin haber aprendido nada.

**Precision (Precisión)**

De todos los pacientes que el modelo clasifica como alto riesgo, qué proporción realmente lo es. Una precisión baja implica muchas falsas alarmas.

$$\text{Precision} = \frac{TP}{TP + FP}$$

**Recall (Sensibilidad)**

De todos los pacientes que realmente son de alto riesgo, qué proporción el modelo identifica correctamente. Un recall bajo implica que el modelo está dejando pasar casos que deberían atenderse.

$$\text{Recall} = \frac{TP}{TP + FN}$$

**F1-Score**

Media armónica de precisión y recall. Equilibra ambas métricas y es preferible a la accuracy cuando las clases están desbalanceadas.

$$F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

En el contexto médico, el recall suele ser más importante que la precisión, porque identificar a un paciente de alto riesgo que el modelo pasa por alto (falso negativo) tiene consecuencias más graves que generar una consulta adicional innecesaria (falso positivo).

**Umbral de decisión**

El modelo no produce directamente una clase (alto o bajo riesgo), sino una probabilidad continua entre 0 y 1. El umbral es el valor a partir del cual se considera que la predicción es "alto riesgo". El umbral por defecto es 0.5, pero el notebook calcula el umbral que maximiza la accuracy y el que maximiza el F1-Score en el conjunto de prueba. Para uso médico se recomienda ajustar el umbral según el objetivo: un umbral más bajo aumenta el recall (se detectan más casos reales) a costa de más falsas alarmas; un umbral más alto reduce las falsas alarmas a costa de pasar por alto algunos casos.

---

## Interpretación de la tabla de comparación de modelos

La tabla que aparece al final del entrenamiento tiene la siguiente estructura:

```
Modelo               |    AUC |   Acc% |   Prec |    Rec |     F1
XGBoost (Optuna)     | 0.XXXX |  XX.X% | 0.XXXX | 0.XXXX | 0.XXXX
LightGBM             | 0.XXXX |  XX.X% | 0.XXXX | 0.XXXX | 0.XXXX
Red Neuronal         | 0.XXXX |  XX.X% | 0.XXXX | 0.XXXX | 0.XXXX
Stacking             | 0.XXXX |  XX.X% | 0.XXXX | 0.XXXX | 0.XXXX
Ensemble final       | 0.XXXX |  XX.X% | 0.XXXX | 0.XXXX | 0.XXXX
```

La fila de "Ensemble final" siempre contendrá el mejor o igual resultado que cualquier modelo individual, ya que el ensemble solo se selecciona si supera al stacking. La columna más relevante para evaluar la calidad global del modelo es el AUC. La accuracy y el F1 dependen del umbral elegido, mientras que el AUC es independiente del umbral y mide la capacidad discriminativa del modelo en su conjunto.

---

## Interpretación de los niveles de riesgo asignados a la brigada

| Nivel | Rango de probabilidad | Interpretación clínica sugerida |
|---|---|---|
| Bajo | Menor al 30% | Paciente sin acumulación significativa de factores de riesgo. Recomendación de mantener hábitos saludables y revisión anual. |
| Moderado | Entre 30% y 50% | Presencia de uno o dos factores de riesgo. Indicado intervención preventiva: orientación nutricional, actividad física, control en tres meses. |
| Moderado-Alto | Entre 50% y 75% | Varios factores de riesgo activos. Seguimiento estrecho, optimización del tratamiento farmacológico si existe, y evaluación de referencia a segundo nivel. |
| Alto | Mayor al 75% | Acumulación de factores de riesgo de alta severidad. Derivación prioritaria a especialista, control glucémico y cardiovascular urgente. |

Estos niveles son orientativos. El médico tratante debe considerar la información clínica completa del paciente, el origen de los datos utilizados (medidos directamente o imputados con mediana poblacional) y el contexto individual antes de tomar cualquier decisión clínica.

---

## Limitaciones del modelo

El modelo predice riesgo metabólico agregado, no enfermedades específicas ni eventos futuros. Para la gran mayoría de los pacientes de la brigada, las variables continuas como glucosa, HbA1c e IMC no fueron medidas y se imputaron con valores poblacionales promedio. Esto significa que las probabilidades predichas para esos pacientes reflejan principalmente la información contenida en sus diagnósticos binarios (DM, HTA, tabaquismo, antecedentes), no mediciones directas. La confiabilidad del resultado es mayor para los tres pacientes con signos vitales medidos y para cualquier paciente cuyo expediente tenga información clínica detallada.
