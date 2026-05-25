<h1 align="center">
  Procesamiento de Alto Volumen de Datos <br>
  Pontificia Universidad Javeriana <br>
  Facultad de Ingeniería de Sistemas <br>
  <img
    src="https://upload.wikimedia.org/wikipedia/commons/6/6c/Javeriana.svg"
    width="200"
    style="display:block; margin:auto; margin-top:12px;"
    alt="Logo Pontificia Universidad Javeriana">
</h1>

<h2 align="center">
  Juan Camilo Torres Meza <br>
</h2>

# Procesamiento de Alto Volumen de Datos



# Contenido del Laboratorio



# 1. Preparación del Entorno

En esta primera etapa se importan las librerías necesarias para el análisis de datos, visualización y construcción de modelos de clasificación sobre Apache Spark. Además, se inicializa la sesión de Spark, la cual permite ejecutar procesamiento distribuido sobre grandes volúmenes de datos.

Este paso es indispensable porque PySpark funciona sobre una arquitectura distribuida y requiere una sesión activa para poder leer datos, ejecutar transformaciones y entrenar modelos de Machine Learning.

### Actividades realizadas

* Importación de bibliotecas de análisis y visualización:

  * `pandas`
  * `numpy`
  * `matplotlib`
  * `seaborn`

* Importación de bibliotecas de PySpark:

  * `SparkSession`
  * `pyspark.sql.functions`
  * `pyspark.ml`
  * `pyspark.ml.classification`
  * `pyspark.ml.feature`
  * `pyspark.ml.evaluation`

* Inicialización de la sesión Spark para el laboratorio.

---

# 2. Carga y Comprensión del Dataset

Se trabaja con un dataset bancario orientado a problemas de clasificación binaria. El objetivo del laboratorio es predecir si un cliente aceptará o no un depósito a plazo fijo, representado mediante la variable objetivo `y`.

El dataset contiene variables demográficas, financieras y relacionadas con campañas de marketing telefónico.

## Variables principales analizadas

* AGE: Edad del cliente.
* JOB: Tipo de trabajo.
* MARITAL: Estado civil.
* EDUCATION: Nivel educativo.
* DEFAULT: Historial de incumplimiento crediticio.
* BALANCE: Balance promedio anual.
* HOUSING: Crédito hipotecario.
* LOAN: Crédito personal.
* CONTACT: Tipo de contacto utilizado.
* DAY: Día del último contacto.
* MONTH: Mes del último contacto.
* DURATION: Duración de la llamada.
* CAMPAIGN: Número de contactos realizados.
* PDAYS: Días desde el último contacto.
* PREVIOUS: Número de contactos previos.
* POUTCOME: Resultado de campañas anteriores.
* Y: Etiqueta objetivo (yes/no).

### Objetivo del análisis

El propósito principal es construir modelos de clasificación capaces de identificar patrones asociados a clientes con mayor probabilidad de aceptar un depósito a plazo fijo.

---

# 3. Inspección y Preparación de Datos

Antes de entrenar modelos de Machine Learning es necesario comprender la estructura de los datos, validar los tipos de variables y detectar posibles problemas de calidad.

## Actividades realizadas

### Inspección de esquema

Se verifica el tipo de datos de cada columna y se identifica que múltiples variables fueron cargadas inicialmente como tipo `String`.

Esto obliga a realizar conversiones de tipos para garantizar compatibilidad con las operaciones matemáticas y algoritmos de Machine Learning.

### Conversión de variables

Se transforman las columnas numéricas al tipo adecuado (`IntegerType`, `DoubleType` o equivalentes en Spark).

### Identificación de la variable objetivo

Se identifica la columna `y` como etiqueta binaria del problema:

* `yes` → Cliente acepta el depósito.
* `no` → Cliente no acepta el depósito.

### Balance de clases

Se analiza la distribución de la variable objetivo y se observa un fuerte desbalance:

* Clase minoritaria: `yes` (~11.7%)
* Clase mayoritaria: `no` (~88.3%)

Este comportamiento es importante porque puede generar sesgo en los modelos de clasificación, favoreciendo la predicción de la clase mayoritaria.

---

# 4. Análisis Exploratorio de Variables Numéricas

Se realiza un análisis exploratorio profundo sobre las variables numéricas mediante histogramas y boxplots.

El objetivo es comprender:

* Distribución de los datos.
* Asimetrías.
* Presencia de outliers.
* Concentraciones de frecuencia.
* Diferencias entre clases.

---

## 4.1 Histogramas

Los histogramas permiten observar cómo se distribuyen las variables numéricas a lo largo de distintos intervalos.

### Variables analizadas

* AGE
* BALANCE
* DAY
* DURATION
* CAMPAIGN
* PDAYS
* PREVIOUS

### Hallazgos principales

#### AGE

* Mayor concentración entre 30 y 50 años.
* Poca representación en edades extremas.
* Distribución relativamente estable.

#### BALANCE

* Fuerte asimetría positiva.
* Alta concentración en balances bajos.
* Presencia de outliers extremos.

#### DURATION

* La mayoría de llamadas tienen duraciones cortas.
* Distribución altamente sesgada hacia la derecha.

#### CAMPAIGN

* La mayoría de clientes reciben pocos contactos.
* Existen pocos casos con cantidades elevadas de llamadas.

#### PDAYS y PREVIOUS

* Gran concentración de datos en valores cercanos a cero.
* Amplia dispersión en valores altos.

---

## 4.2 Boxplots y Análisis por Clase

Se utilizan boxplots para comparar las distribuciones de variables numéricas entre las clases `yes` y `no`.

Los boxplots permiten identificar:

* Mediana.
* Cuartiles.
* Rango intercuartílico.
* Valores atípicos.
* Diferencias entre categorías.

### Hallazgos importantes

#### AGE

* Ambas clases tienen medianas similares.
* El grupo `yes` presenta mayor dispersión hacia edades altas.

#### BALANCE

* Los clientes con balances altos aparecen principalmente en la clase `no`.
* Existen múltiples outliers financieros.

#### DURATION

* Es una de las variables más discriminantes.
* Clientes que aceptan el depósito presentan llamadas significativamente más largas.

#### PDAYS

* Clientes previamente contactados muestran mayor probabilidad de aceptar.

#### PREVIOUS

* La mayoría de clientes tienen pocos contactos previos.
* Existen outliers extremos en campañas anteriores.

---

# 5. Preparación de Datos para Machine Learning

Antes de entrenar modelos supervisados es necesario transformar las variables categóricas y construir el conjunto final de características.

## Actividades realizadas

### Indexación de variables categóricas

Se utiliza `StringIndexer` para convertir variables categóricas en índices numéricos compatibles con Spark MLlib.

### One Hot Encoding

Las variables categóricas indexadas son transformadas usando `OneHotEncoder`.

Esto evita introducir relaciones ordinales artificiales entre categorías.

### Construcción del vector de características

Se emplea `VectorAssembler` para combinar todas las variables en una sola columna de features.

### División del dataset

Se divide el dataset en:

* 80% entrenamiento.
* 20% prueba.

La división permite evaluar la capacidad de generalización de los modelos.


# 6. Métricas de Evaluación

Se describen y utilizan múltiples métricas para evaluar el rendimiento de los modelos de clasificación.

## Accuracy

Mide la proporción total de predicciones correctas.

## Precision

Evalúa qué porcentaje de predicciones positivas realmente pertenece a la clase positiva.

## Recall

Mide la capacidad del modelo para detectar correctamente los casos positivos.

## F1-Score

Representa el equilibrio entre precisión y recall.

## AUC-ROC

Mide la capacidad discriminativa del modelo entre clases.

## Matriz de Confusión

Permite visualizar:

* Verdaderos positivos.
* Verdaderos negativos.
* Falsos positivos.
* Falsos negativos.


# 7. Construcción y Evaluación de Modelos

Se implementan múltiples algoritmos de clasificación usando PySpark MLlib con el objetivo de comparar su desempeño sobre el mismo conjunto de datos.


# 7.1 Logistic Regression

Se construye un modelo de regresión logística para clasificación binaria.

## Objetivo

Modelar la probabilidad de que un cliente acepte el depósito a plazo fijo.

## Evaluación realizada

* Accuracy
* Matriz de confusión
* Curva ROC
* AUC

### Observaciones

* El modelo funciona adecuadamente sobre la clase mayoritaria.
* Se observan limitaciones para detectar correctamente la clase minoritaria debido al desbalance.


# 7.2 Random Forest

Se implementa un modelo basado en múltiples árboles de decisión.

## Objetivo

Mejorar la capacidad de generalización y reducir el overfitting mediante ensambles.

## Evaluación realizada

* Accuracy
* Matriz de confusión
* Curva ROC
* AUC

### Observaciones

* Mayor robustez frente a ruido.
* Mejor capacidad de capturar relaciones no lineales.
* Buen desempeño general.


# 7.3 Gradient Boosted Trees

Se utiliza un modelo de boosting basado en árboles secuenciales.

## Objetivo

Optimizar el rendimiento corrigiendo errores iterativamente.

## Evaluación realizada

* Accuracy
* Matriz de confusión
* Curva ROC
* AUC

### Observaciones

* Alto poder predictivo.
* Mejor discriminación de patrones complejos.
* Sensible al ajuste de hiperparámetros.


# 7.4 Decision Tree

Se implementa un árbol de decisión individual.

## Objetivo

Construir un modelo interpretable basado en reglas.

## Evaluación realizada

* Accuracy
* Matriz de confusión
* Curva ROC
* AUC

### Observaciones

* Fácil interpretación.
* Mayor riesgo de overfitting frente a modelos ensemble.


# 7.5 Linear Support Vector Classifier (Linear SVC)

Se implementa un clasificador lineal basado en máquinas de soporte vectorial.

## Objetivo

Separar las clases mediante un hiperplano óptimo.

## Evaluación realizada

* Accuracy
* Matriz de confusión
* Curva ROC
* AUC

### Observaciones

* Buen comportamiento en espacios de alta dimensionalidad.
* Sensible al desbalance de clases.


# 7.6 Multilayer Perceptron

Se construye una red neuronal multicapa utilizando las capacidades de clasificación de Spark MLlib.

## Objetivo

Capturar relaciones complejas y no lineales entre las variables.

## Evaluación realizada

* Accuracy
* Matriz de confusión
* Curva ROC
* AUC

### Observaciones

* Capacidad de aprendizaje más flexible.
* Mayor complejidad computacional.
* Dependencia de arquitectura y parámetros.


# 8. Comparación de Modelos

Finalmente se comparan todos los algoritmos entrenados usando las métricas obtenidas.

## Aspectos comparados

* Accuracy
* AUC
* Rendimiento sobre clase minoritaria
* Robustez
* Capacidad de generalización
* Interpretabilidad

El objetivo de esta etapa es identificar qué algoritmo presenta el mejor equilibrio entre rendimiento predictivo y estabilidad.


# 9. Conclusiones Finales

## Hallazgos generales

* El dataset presenta un fuerte desbalance de clases.
* Variables como `DURATION` tienen alta capacidad discriminativa.
* Los modelos ensemble muestran mejor rendimiento general.
* La evaluación mediante AUC-ROC y matriz de confusión resulta fundamental en problemas desbalanceados.
* Los modelos simples son más interpretables, mientras que modelos complejos capturan mejor relaciones no lineales.

## Aprendizajes del laboratorio

* Uso de PySpark para clasificación distribuida.
* Preparación de variables categóricas.
* Construcción de pipelines de Machine Learning.
* Evaluación comparativa de algoritmos.
* Interpretación de métricas de clasificación.


# Tecnologías Utilizadas

* Apache Spark
* PySpark MLlib
* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn


# Referencias

* Apache Software Foundation. (2024). *Apache Spark Documentation.* Recuperado de: [https://spark.apache.org/docs/latest/](https://spark.apache.org/docs/latest/)

* Apache Software Foundation. (2024). *PySpark MLlib Classification Documentation.* Recuperado de: [https://spark.apache.org/docs/latest/ml-classification-regression.html](https://spark.apache.org/docs/latest/ml-classification-regression.html)

* Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python.* Journal of Machine Learning Research, 12, 2825-2830.

* Han, J., Kamber, M., & Pei, J. (2012). *Data Mining: Concepts and Techniques.* Morgan Kaufmann.

* Géron, A. (2022). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow.* O'Reilly Media.

* Dataset Bank Marketing — UCI Machine Learning Repository. Recuperado de: [https://archive.ics.uci.edu/ml/datasets/bank+marketing](https://archive.ics.uci.edu/ml/datasets/bank+marketing)
