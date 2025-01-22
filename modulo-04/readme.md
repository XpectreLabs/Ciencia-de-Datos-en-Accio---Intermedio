# Módulo 4: Fundamentos de Machine Learning 

## 🎯Objetivo del Módulo
Al finalizar este módulo, los participantes comprenderán los conceptos clave de Machine Learning, los diferentes tipos de aprendizaje, y serán capaces de construir y evaluar modelos supervisados y no supervisados utilizando **Scikit-learn**.

---

## Tema: Introducción a Machine Learning

### Tema 4.1: Supervisado vs No Supervisado


#### ¿Qué es Machine Learning?  
- Machine Learning es un campo de la informática que se enfoca en desarrollar algoritmos y modelos que permiten a las máquinas aprender patrones y a partir de esos patrones, realizar predicciones o tomar decisiones automáticamente. A diferencia de los enfoques tradicionales de programación, donde los resultados son definidos explícitamente, el Machine Learning permite que los sistemas aprendan de los datos y mejoren su rendimiento sin la necesidad de programación explícita para cada caso.

- **Aplicaciones comunes**:
  1. **Predicción de precios de bienes o servicios**:
     - Los modelos de Machine Learning, como los algoritmos de regresión, se utilizan para predecir el precio de bienes, como casas, acciones bursátiles o productos. Estos modelos pueden tener en cuenta múltiples variables como la ubicación, el tamaño, las tasas de interés, el historial económico, etc.

  2. **Clasificación de correos electrónicos como spam o no spam**:
     - Un ejemplo clásico de Machine Learning supervisado es el filtrado de correos electrónicos. Los modelos, como los basados en **Machine Learning** supervisado (como la Regresión Logística o Máquinas de Soporte Vectorial), aprenden a distinguir entre correos legítimos y mensajes de **spam** mediante el análisis de características del contenido, el remitente y el asunto.

  3. **Agrupamiento de clientes según sus características de compra**:
     - Utilizando técnicas como el Clustering, se pueden agrupar clientes en segmentos basados en su comportamiento de compra, intereses, geolocalización, patrones de gasto, etc. Esto permite a las empresas personalizar sus estrategias de marketing y ofrecer productos que coincidan con los hábitos de consumo de cada segmento.
  
  4. **Recomendación de productos**:
     - A través del aprendizaje automático, se pueden construir sistemas de recomendación que sugieren productos a los usuarios basados en su historial de compras anteriores, visualizaciones, calificaciones o preferencias. Por ejemplo, plataformas como Netflix, Amazon y Spotify utilizan estos sistemas para mejorar la experiencia del usuario.

  5. **Análisis de sentimiento**:
     - El Machine Learning también es empleado para analizar el sentimiento en redes sociales, opiniones de usuarios, o textos, identificando si un mensaje tiene un tono positivo, negativo o neutro. Los modelos de clasificación textuales permiten entender las emociones subyacentes en los datos textuales.

  6. **Reconocimiento de voz y procesamiento de lenguaje natural**:
     - Los sistemas de Machine Learning, como los modelos de **Deep Learning** y redes neuronales recurrentes, se utilizan para convertir el habla en texto (sistemas de reconocimiento de voz) o para analizar y comprender el lenguaje humano en aplicaciones como asistentes virtuales (Siri, Alexa) o traducción automática.

  7. **Diagnóstico en salud**:
     - Los modelos de Machine Learning también se aplican en el sector salud para diagnosticar enfermedades, como la detección temprana de cáncer a través de análisis de imágenes médicas (radiografías, resonancias magnéticas) o para el análisis de datos genéticos.

---


#### Tipos principales de aprendizaje:

### 1. Machine Learning Supervisado:
- Los modelos de Machine Learning supervisado aprenden de datos etiquetados, es decir, cada entrada en el conjunto de datos tiene una salida conocida y predefinida. Este tipo de aprendizaje implica la utilización de ejemplos previamente clasificados para entrenar el modelo, permitiéndole generalizar patrones y realizar predicciones sobre datos nuevos.

- **Ejemplo práctico**:
  - **Predecir el precio de una casa**: Un modelo de Machine Learning supervisado puede predecir el precio de una casa basado en características como el tamaño, la ubicación, la antigüedad, el número de habitaciones, los servicios cercanos, entre otros. El modelo se entrena utilizando un conjunto de datos que incluye precios reales de casas junto con sus respectivas características.

- **Características**:
  - **Entrenamiento**: El modelo se entrena con datos etiquetados, donde se conocen las respuestas correctas.
  - **Salida**: El modelo proporciona una respuesta específica y cuantificada, como un precio o una clasificación (por ejemplo, clasificar un correo como **spam** o **no spam**).
  - **Modelos comunes**:
    - **Regresión lineal**: Se utiliza para hacer predicciones continuas, como el precio de una casa basado en características numéricas.
    - **Regresión logística**: Muy útil para problemas de clasificación binaria, como la clasificación de correos electrónicos.
    - **Máquinas de Soporte Vectorial (SVM)**: Son eficaces cuando los datos están separados por márgenes definidos.
    - **Árboles de Decisión**: Permiten la creación de modelos interpretables que dividen el espacio de entrada en regiones que maximizan la pureza de clasificación.

---

### 2. Machine Learning No Supervisado:
- Los modelos de Machine Learning no supervisado no aprenden con datos etiquetados, sino que buscan patrones, agrupamientos o relaciones en los datos sin una guía explícita sobre las salidas. Este tipo de aprendizaje permite descubrir estructuras en los datos que son útiles para la interpretación y segmentación.

- **Ejemplo práctico**:
  - **Agrupar clientes en segmentos según su comportamiento de compra**: Un modelo de Machine Learning no supervisado puede utilizar algoritmos como K-Means para identificar clusters de clientes con características similares en términos de sus patrones de compra, edad, ingresos, o hábitos.

- **Características**:
  - **Entrenamiento**: El modelo es entrenado con datos no etiquetados, donde la información no incluye resultados previamente definidos.
  - **Salida**: Los resultados son estructuras como clusters, agrupamientos o características de similitud que ayudan a entender los datos desde una perspectiva nueva.
  - **Modelos comunes**:
    - **Algoritmos de agrupamiento (Clustering)**:
      - **K-Means**: Divide los datos en `k` clusters de acuerdo con la proximidad.
      - **Clustering jerárquico**: Construye una jerarquía de clusters mediante divisiones sucesivas.
      - **Aprendizaje Basado en Vecinos Cercanos**: Agrupa los puntos en función de la proximidad a otros puntos en el espacio de características.


#### Ciclo de vida de un modelo de Machine Learning:
1. **Recopilación de datos**: 
   - Obtener datos relevantes y estructurados para el problema que se desea resolver. 
   - Ejemplo: Datos de clientes, histórico de ventas, etc.

2. **Preprocesamiento**:
   - Limpiar y transformar los datos para que puedan ser utilizados en el modelo.
   - Los pasos incluyen: corrección de valores nulos, escalado de variables, normalización, codificación de características categóricas, etc.

3. **División de datos**:
   - **Conjunto de entrenamiento**: Se usa para ajustar el modelo.
   - **Conjunto de validación**: Se usa para ajustar y validar el modelo durante el entrenamiento.
   - **Conjunto de prueba**: Se usa para evaluar el modelo una vez entrenado, para conocer su desempeño en datos que no ha visto.

4. **Entrenamiento**:
   - Ajustar los parámetros del modelo utilizando el conjunto de entrenamiento.
   - Proceso iterativo en el cual el modelo busca encontrar las mejores funciones o pesos que mejoran su capacidad para hacer predicciones.

5. **Evaluación**:
   - Medir el rendimiento del modelo utilizando métricas adecuadas en el conjunto de prueba.
   - Las métricas comunes incluyen:
     - **Precisión**: La fracción de aciertos en relación con todos los casos evaluados.
     - **Recall**: La capacidad del modelo de encontrar todos los casos positivos.
     - **F1-score**: Combinación entre precisión y recall.
     - **Root Mean Square Error (RMSE)**: Para modelos de regresión.
     - **Accuracy**: Porcentaje de aciertos en las predicciones.

#### Diagrama del proceso de Machine Learning:
- <img src="https://rchavarria.github.io/notes/assets/images/2018/machine-learning-process.png" alt="Descripción de la imagen" width="700">

#### Ejemplo
  -[**`Ejemplo 4.1`**](ejemplo4_1/Ejemplo4_1.ipynb)

#### Ejercicio para alumnos 
- Clasificar diferentes casos de uso de Machine Learning como supervisados o no supervisados.

---

## Tema: División de Datos

### Tema 4.2: Entrenamiento, Validación y Prueba


**¿Por qué dividir los datos?**
- La división de los datos es crucial para **evitar el sobreajuste** al modelo y para **evaluar su rendimiento** en situaciones reales. El entrenamiento solo con un conjunto puede llevar a una evaluación optimista (desempeño demasiado bueno en datos vistos).
- Al dividir los datos en conjuntos separados para **entrenamiento**, **validación** y **prueba**, se asegura que el modelo generalice bien a nuevos datos y que el rendimiento del modelo sea medible de manera precisa.

---

#### Divisiones comunes de los datos:

1. **Conjunto de Entrenamiento (60-80%)**:
   - **Propósito**: Se utiliza para ajustar el modelo, aprendiendo los patrones y relaciones existentes en los datos.
   - Este conjunto es donde se entrena el modelo aplicando los algoritmos hasta que se ajuste correctamente al problema.
   - **Ventajas**: 
     - Amplía la capacidad del modelo para generalizar a datos nuevos.
     - Permite ajustar los parámetros internos del modelo a partir del conjunto de entrada.
   - **Ejemplo**: Cuando se trabaja con datos de ventas, los datos de entrenamiento serán utilizados para ajustar las características del modelo que predicen los resultados de ventas futuras.

2. **Conjunto de Validación (10-20%)**:
   - **Propósito**: Este conjunto se usa para ajustar los **hiperparámetros** del modelo, como la tasa de aprendizaje, el número de vecinos en **KNN**, o el **threshold** en un clasificador.
   - El modelo puede ajustar el rendimiento según la validación antes de su implementación final.
   - **Ventajas**: Permite comparar el desempeño de varias configuraciones del modelo.
   - **Ejemplo**: Ajustar el número de épocas en un modelo de red neuronal para evitar el **overfitting** o el **underfitting**.

3. **Conjunto de Prueba (10-20%)**:
   - **Propósito**: Se usa para evaluar el rendimiento final del modelo, proporcionando una métrica objetiva de su capacidad para hacer predicciones en datos nuevos que no ha visto.
   - Es vital que este conjunto no se haya utilizado previamente en ninguna etapa de entrenamiento o validación.
   - **Ventajas**:
      - **Evaluación Objetiva**: Permite medir el rendimiento del modelo en datos que nunca antes ha visto, lo que proporciona una estimación más realista de su capacidad para generalizar a situaciones nuevas.
      - **Verificación Final**: Sirve como un paso final para validar si el modelo es capaz de realizar predicciones precisas en escenarios reales, antes de su implementación comercial o en producción.
      - **Minimiza el Overfitting**: Al utilizar datos completamente nuevos, evita el sobreajuste al garantizar que el modelo no ha aprendido patrones específicos de los datos utilizados en el entrenamiento o validación.
   - **Ejemplo**: Evaluar el desempeño de un modelo clasificador en una muestra de datos completamente nueva y separada, como la predicción de enfermedades en una población distinta.

- <img src="https://keepcoding.io/wp-content/uploads/2022/08/image-202-1024x565.png" alt="Descripción de la imagen" width="700">
---

#### Técnicas de validación cruzada:

1. **K-Fold Cross Validation**:
   - **Descripción**: Divide el conjunto de datos en `k` subconjuntos iguales o aproximadamente iguales. 
   - Se entrena y evalúa el modelo `k` veces, utilizando diferentes subconjuntos para el entrenamiento cada vez, dejando un subconjunto como prueba.
   - **Ventajas**:
     - Proporciona una estimación más robusta del rendimiento del modelo al entrenar y evaluar múltiples veces.
     - Minimiza el sesgo que podría surgir por una partición accidentalmente desbalanceada.
   - **Ejemplo**: Si `k=5`, el conjunto de datos se divide en cinco partes iguales y el modelo es entrenado y validado cinco veces usando diferentes combinaciones.

2. **Leave-One-Out Cross Validation (LOOCV)**:
   - **Descripción**: Usa un dato como conjunto de prueba y el resto como conjunto de entrenamiento. 
   - Este proceso se repite hasta que todos los datos han servido como conjunto de prueba una vez.
   - **Ventajas**:
     - Minimiza el sesgo, ya que utiliza todos los datos en la fase de entrenamiento.
   - **Ejemplo**: Ideal para pequeñas bases de datos donde el número total de datos es pequeño.

3. **Stratified K-Fold Cross Validation**:
   - **Descripción**: Similar a K-Fold, pero asegura que las clases estén equilibradas en cada división. 
   - Muy útil cuando los datos están desbalanceados, como en problemas de clasificación binaria.
   - **Ventajas**:
     - Mantiene la proporción de clases en cada subsampleo, lo que evita que las clases minoritarias influyan de manera excesiva en el entrenamiento.
   - **Ejemplo**: En una base de datos con 90% de datos de clase A y 10% de clase B, el **Stratified K-Fold** asegura que cada división mantenga esa proporción.

---

#### Otras técnicas de validación:

1. **Bootstrap**:
   - **Descripción**: Genera múltiples conjuntos de entrenamiento a partir del conjunto original mediante muestreo con reemplazo.
   - Esto permite evaluar la estabilidad y la varianza del modelo al generar múltiples subconjuntos.
   - **Ventajas**:
     - Proporciona una estimación más robusta del rendimiento del modelo.
     - Permite identificar la incertidumbre asociada con el desempeño del modelo.
   - **Ejemplo**: Ideal para realizar inferencias sobre el desempeño del modelo cuando se tienen muestras pequeñas.

2. **Under-Sampling**:
   - **Descripción**: Consiste en eliminar muestras de la mayoría de clases para igualar el número de instancias de las clases minoritarias.
   - **Ventajas**:
     - Ayuda a mitigar el sesgo cuando los datos están desbalanceados.
   - **Ejemplo**: En conjuntos de datos con más casos negativos que positivos (clasificación binaria), el **under-sampling** permite reducir el número de negativos.

3. **Over-Sampling**:
   - **Descripción**: Aumenta las muestras de las clases minoritarias mediante técnicas como **SMOTE** (Synthetic Minority Over-sampling Technique).
   - **Ventajas**:
     - Reduce el problema del sesgo al aumentar las instancias de clases minoritarias.
   - **Ejemplo**: En conjuntos de datos con pocos casos positivos en una clasificación binaria, el **over-sampling** ayuda a equilibrar las clases.

---

#### Otras técnicas importantes para la validación de modelos:

- **Grid Search**:
  - **Descripción**: Técnica que busca encontrar los mejores hiperparámetros en un espacio definido. 
  - Se realiza explorando exhaustivamente todas las combinaciones posibles.
  - **Ventajas**:
    - Garantiza encontrar un conjunto óptimo de hiperparámetros.
  - **Desventajas**:
    - Computacionalmente costoso para grandes conjuntos de parámetros.

- **Random Search**:
  - **Descripción**: Similar al **Grid Search**, pero explora de manera aleatoria las configuraciones posibles de hiperparámetros.
  - **Ventajas**:
    - Más eficiente computacionalmente que el **Grid Search** y proporciona buenos resultados en muchos casos.
  - **Ejemplo**: Cuando se buscan las mejores combinaciones de **learning rate**, **número de capas** o **número de vecinos**.

- **Early Stopping**:
  - **Descripción**: Esta técnica se usa para detener el entrenamiento cuando el desempeño en la validación deja de mejorar. 
  - **Ventajas**:
    - Evita el sobreajuste al detener el proceso antes de que el modelo aprenda patrones innecesarios que no contribuyen a una buena generalización.
  - **Ejemplo**: Se detiene el entrenamiento cuando la pérdida en el conjunto de validación no mejora durante varias épocas consecutivas.

#### Ejemplo 
   -[**`Ejemplo 4.2`**](ejemplo4.2/Ejemplo4_2.ipynb)
   
---

## Tema: Modelos Supervisados

### Tema 4.3: Regresión Lineal

- La **regresión lineal** es un modelo supervisado que establece una relación lineal entre las variables independientes (features) y una variable dependiente (target).
- Es uno de los modelos más básicos en Machine Learning, utilizado comúnmente para **predecir** resultados continuos, como precios, ingresos o temperaturas.
- **Ejemplo práctico**: Predecir el precio de una casa basado en características como su tamaño, ubicación, antigüedad, número de habitaciones, etc.

**Ecuación del modelo**:
\[y = mx + b\]
- **m** es la pendiente, que representa la inclinación de la línea, es decir, cuánto cambia la variable dependiente (`y`) cuando la variable independiente (`x`) aumenta en una unidad.
- **b** es el intercepto, es decir, el punto en el eje `y` donde la línea cruza al no tener ninguna influencia de la variable independiente.

**Entrenamiento del modelo**:
- La regresión lineal se entrena **minimizando el error cuadrático medio (MSE)**, que mide la diferencia entre los valores reales y los valores predichos.
- El objetivo es encontrar los valores óptimos de `m` (pendiente) y `b` (intercepto) que minimizan el MSE. Esto asegura que la línea sea la más ajustada a los datos.

**Formas adicionales de entrenamiento**:
- **Gradient Descent**: Técnica comúnmente utilizada para encontrar el conjunto óptimo de `m` y `b` a través de iteraciones que ajustan los parámetros siguiendo el descenso del gradiente.
- **Normal Equation**: Proporciona una fórmula algebraica directa para encontrar `m` y `b`, ideal cuando el conjunto de datos es pequeño y no hay colinealidad entre las variables.

**Ventajas de la Regresión Lineal**:
- **Fácil de interpretar**: Los coeficientes (`m` y `b`) tienen un significado claro en términos del impacto de las variables independientes en la variable dependiente.
- **Rápido de implementar**: Se puede entrenar rápidamente y obtener predicciones basadas en datos relativamente simples.
- **Escalabilidad**: Es fácil extenderla a múltiples variables y se puede aplicar tanto a problemas con una como con múltiples variables.

**Desventajas de la Regresión Lineal**:
- **Asumida Linealidad**: Asume que la relación entre las variables es estrictamente lineal, lo que puede llevar a malos resultados si los datos tienen relaciones no lineales.
- **Sensibilidad a Outliers**: La regresión lineal es afectada negativamente por **outliers**, ya que estos distorsionan los cálculos del MSE.
- **Restricción de normalidad**: Asume que los errores siguen una distribución normal, lo cual puede ser inapropiado para ciertos conjuntos de datos.

- <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Linear_regression.svg/1200px-Linear_regression.svg.png" alt="Descripción de la imagen" width="500">

#### Ejemplo 
  -[**`Ejemplo 4.3`**](ejemplo4.3/Ejemplo4_3.ipynb)

#### Ejercicio 4.1 (Alumnos)
 -[**`Ejercicio 4.1`**](ejercicio4.1/Ejercicio4_1.ipynb)

---

### Tema 3.4: Clasificación

- La **clasificación** es un tipo de problema supervisado donde el objetivo es predecir **etiquetas categóricas** o clases de salida. Las clases pueden ser binarias (spam/no spam) o multiclasificación (categorías como documentos de texto, tipos de flores, clases de imágenes, etc.).
- Ejemplos comunes de clasificación incluyen **detectar si un correo es spam o no spam**, **clasificar pacientes con base en diagnóstico** o **predecir si un cliente realiza una compra o no**.

---

### Algoritmo k-Nearest Neighbors (k-NN):

- **k-Nearest Neighbors (k-NN)** es uno de los algoritmos más sencillos y efectivos para problemas de clasificación. 
- Utiliza la **distancia** entre puntos en el espacio de características para asignar una clase a un nuevo dato basado en los `k` vecinos más cercanos.
  
1. **Seleccionar** un nuevo dato o punto de interés.
2. **Calcular** la distancia entre este nuevo punto y todos los puntos en el conjunto de entrenamiento. Las distancias se miden comúnmente utilizando:
   - **Distancia Euclidiana**:
   - <img src="https://upload.wikimedia.org/wikipedia/commons/6/67/Distance_Formula.svg" alt="Descripción de la imagen" width="500">
   - **Distancia Manhattan**:
   - <img src="https://medidassimdist.wordpress.com/wp-content/uploads/2019/04/image-11.png" alt="Descripción de la imagen" width="500">
3. **Seleccionar** los `k` vecinos más cercanos en función de la distancia calculada.
4. **Asignar** la clase basada en el modo (mayoría) de las etiquetas de estos `k` vecinos.
   
**Ventajas del k-NN**:
- **Sencillo y fácil de entender**: No requiere entrenamiento complejo; simplemente almacena los datos y los compara con nuevos puntos.
- **Escalabilidad**: Es flexible y puede ser utilizado tanto con datos pequeños como grandes.
- **No necesita hiperparámetros complejos**: Solo se elige el valor de `k`.

**Desventajas del k-NN**:
- **Sensibilidad a los outliers**: Los puntos atípicos (outliers) pueden afectar la clasificación debido a su influencia en las distancias.
- **Requiere mucho espacio de almacenamiento**: A medida que los datos aumentan, el algoritmo necesita más memoria para almacenar los puntos de entrenamiento.
- **Complejidad computacional**: En datos grandes, el cálculo de las distancias puede ser costoso en términos de tiempo.

- <img src="https://db0dce98.rocketcdn.me/es/files/2020/11/Illu-2-KNN-1024x492.jpg" alt="Descripción de la imagen" width="500">

#### Ejemplo 
  -[**`Ejemplo 4.4`**](ejemplo4.4/Ejemplo4_4.ipynb)

#### Ejercicio 4.2 (Alumnos)
  --[**`Ejercicio 4.2`**](ejemplo4.2/Ejemplo4_2.ipynb)

---

## Tema: Modelos No Supervisados

### Tema 3.5: Clustering con K-Means

---

### **Definición de Clustering**:
- **Clustering** es una técnica de aprendizaje no supervisado que **agrupa datos** en conjuntos (clústeres) basándose en sus **similitudes** o características compartidas. 
- Los puntos en el mismo clúster son **más similares** entre sí que con los puntos de otros clústeres.
- **Ejemplos de aplicación**:
  - Agrupamiento de clientes según su comportamiento de compra (marketing).
  - Identificación de patrones en datos genómicos (bioinformática).
  - Segmentación de imágenes en visión por computadora.

---

### **Funcionamiento de K-Means**:

**Definición**:
- El algoritmo **K-Means** es uno de los métodos de clustering más populares, que organiza los datos en **k** clústeres basándose en la proximidad a centroides.
- <img src="https://images.javatpoint.com/tutorial/machine-learning/images/k-means-clustering-algorithm-in-machine-learning.png" alt="Descripción de la imagen" width="500">

---

### **Ventajas del K-Means**:
- **Simplicidad**: Fácil de entender e implementar.
- **Escalabilidad**: Puede manejar grandes volúmenes de datos de manera eficiente.
- **Flexibilidad**: Funciona bien con datos esféricos y distribuciones homogéneas.

---

### **Desventajas del K-Means**:
- **Dependencia de `k`**:
  - Requiere predefinir el número de clústeres (`k`), lo cual puede ser difícil sin conocimiento previo.
  - Técnicas como el **Método del Codo** ayudan a seleccionar un valor óptimo de `k`.

- **Sensibilidad a Outliers**:
  - Los datos atípicos pueden distorsionar los centroides y la asignación de clústeres.

- **Forma de los Clústeres**:
  - Funciona mejor con clústeres esféricos y uniformemente distribuidos; no maneja bien formas irregulares.

---

### **Aplicaciones del K-Means**:
- **Segmentación de Clientes**:
  - Agrupa consumidores según sus patrones de compra para personalizar estrategias de marketing.
- **Compresión de Imágenes**:
  - Reduce el número de colores en una imagen, agrupando píxeles con colores similares.
- **Análisis de Datos Genéticos**:
  - Clasifica genes o muestras según patrones de expresión similares.

---

### **Visualización del Proceso**:
- <img src="https://blog.thedigitalgroup.com/assets/uploads/k-means3.jpg" alt="Descripción de la imagen" width="500">

#### Ejemplo 
  --[**`Ejemplo 4.5`**](ejemplo4.5/Ejemplo4_5.ipynb)

---

## Tema: Evaluación de Modelos

### Tema 4.6: Métricas de Evaluación

---

### **Evaluación para Clasificación**

Las métricas de evaluación para problemas de clasificación miden qué tan bien un modelo predice las clases correctas. Entre las más comunes se encuentran:

---

#### **1. Accuracy (Precisión Global):**
- Mide la proporción de predicciones correctas en relación con el total de predicciones realizadas.
- <img src="https://latex.codecogs.com/png.latex?%5Ctext%7BAccuracy%7D%20%3D%20%5Cfrac%7B%5Ctext%7BPredicciones%20Correctas%7D%7D%7B%5Ctext%7BTotal%20de%20Predicciones%7D%7D" alt="Descripción de la imagen" width="300">
- **Ventajas**:
  - Fácil de interpretar.
  - Útil cuando las clases están equilibradas.
- **Desventajas**:
  - No es fiable si las clases están desbalanceadas (por ejemplo, 95% de una clase y 5% de otra).

---

#### **2. Precisión (Precision):**
- Indica qué proporción de las predicciones positivas son realmente positivas.
- <img src="https://latex.codecogs.com/png.latex?%5Ctext%7BPrecision%7D%20%3D%20%5Cfrac%7B%5Ctext%7BTrue%20Positives%20%28TP%29%7D%7D%7B%5Ctext%7BTrue%20Positives%20%28TP%29%20%2B%20False%20Positives%20%28FP%29%7D%7D" alt="Descripción de la imagen" width="300">
- **Aplicación**: Útil en casos donde es crucial minimizar falsos positivos (por ejemplo, detección de spam)
- **Ventajas**:
  - Es ideal en problemas donde las consecuencias de los falsos positivos son graves, como en la clasificación de correos no deseados o detección de fraudes.
  - Ofrece una métrica clara cuando el interés principal está en la calidad de las predicciones positivas.
- **Desventajas**:
  - No considera los falsos negativos, por lo que puede ser engañosa en problemas donde estos tienen un impacto significativo (por ejemplo, diagnóstico médico).
  - En datasets desbalanceados, puede ofrecer resultados sesgados si una clase domina las predicciones positivas.

---

#### **3. Recall (Sensibilidad o Tasa de Verdaderos Positivos):**
- Mide la proporción de positivos reales que fueron correctamente identificados.
- <img src="https://latex.codecogs.com/png.latex?%5Ctext%7BRecall%7D%20%3D%20%5Cfrac%7B%5Ctext%7BTrue%20Positives%20%28TP%29%7D%7D%7B%5Ctext%7BTrue%20Positives%20%28TP%29%20%2B%20False%20Negatives%20%28FN%29%7D%7D" alt="Descripción de la imagen" width="300">
- **Aplicación**: Importante en problemas donde los falsos negativos tienen un alto costo (por ejemplo, diagnóstico médico).
- **Ventajas**:
  - Es útil cuando los falsos negativos deben minimizarse al máximo, como en problemas críticos de salud o seguridad.
  - Proporciona una visión completa de cuántos positivos reales son correctamente identificados.
- **Desventajas**:
  - No considera los falsos positivos, lo que puede llevar a sobreestimar el desempeño en problemas donde estos tienen impacto.
  - Si se optimiza únicamente el recall, pueden aumentar los falsos positivos.


---

#### **4. F1-Score:**
- Es la media armónica entre precisión y recall, proporcionando un equilibrio entre ambas métricas.
- <img src="https://latex.codecogs.com/png.latex?%5Ctext%7BF1-Score%7D%20%3D%202%20%5Ccdot%20%5Cfrac%7B%5Ctext%7BPrecision%7D%20%5Ccdot%20%5Ctext%7BRecall%7D%7D%7B%5Ctext%7BPrecision%7D%20%2B%20%5Ctext%7BRecall%7D%7D" alt="Descripción de la imagen" width="300">
- **Aplicación**: Útil cuando hay un desbalance entre las clases y se requiere un balance entre precisión y recall.
- **Ventajas**:
  - Equilibra precisión y recall, siendo ideal para problemas con clases desbalanceadas donde ambos errores (falsos positivos y negativos) son relevantes.
  - Proporciona una única métrica que resume el desempeño general del modelo.
- **Desventajas**:
  - No ofrece un análisis detallado de precisión y recall por separado, lo que puede ocultar problemas específicos en el modelo.
  - No es fácil de interpretar directamente si el contexto del problema no está claro.


---

#### **5. Matriz de Confusión:**
- Es una tabla que describe el desempeño del modelo, mostrando las predicciones correctas e incorrectas para cada clase.
  
  |                | Predicción Positiva | Predicción Negativa |
  |----------------|---------------------|---------------------|
  | **Clase Positiva** | True Positive (TP)    | False Negative (FN)   |
  | **Clase Negativa** | False Positive (FP)   | True Negative (TN)    |

- **Ventajas**:
  - Permite calcular múltiples métricas (precisión, recall, etc.).
  - Proporciona una visión detallada del desempeño del modelo.

---

### **Evaluación para Regresión**

En problemas de regresión, las métricas evalúan qué tan cerca están las predicciones del modelo de los valores reales.

---

#### **1. Error Absoluto Medio (MAE):**
- Promedio de los valores absolutos de los errores entre predicciones y valores reales.
- <img src="https://latex.codecogs.com/png.latex?%5Ctext%7BMAE%7D%20%3D%20%5Cfrac%7B1%7D%7Bn%7D%20%5Csum_%7Bi%3D1%7D%5En%20%7Cy_i%20-%20%5Chat%7By%7D_i%7C" alt="Descripción de la imagen" width="300">
- **Ventajas**:
  - Fácil de interpretar.
  - No amplifica grandes errores como el MSE.
- **Desventajas**:
  - No penaliza errores grandes.

---

#### **2. Error Cuadrático Medio (MSE):**
- Promedio de los cuadrados de los errores entre predicciones y valores reales.
- <img src="https://latex.codecogs.com/png.latex?%5Ctext%7BMSE%7D%20%3D%20%5Cfrac%7B1%7D%7Bn%7D%20%5Csum_%7Bi%3D1%7D%5En%20%28y_i%20-%20%5Chat%7By%7D_i%29%5E2" alt="Descripción de la imagen" width="300">
- **Ventajas**:
- **Ventajas**:
  - Penaliza los errores grandes, siendo útil para modelos donde estos son críticos.
- **Desventajas**:
  - Puede ser influido en exceso por outliers.

---

#### **3. Coeficiente de Determinación ($R^2$):**
- Mide qué proporción de la variación en los datos es explicada por el modelo.
- <img src="https://latex.codecogs.com/png.latex?R%5E2%20%3D%201%20-%20%5Cfrac%7B%5Ctext%7BSSE%7D%7D%7B%5Ctext%7BSST%7D%7D" alt="Descripción de la imagen" width="300">
- **Ventajas**:
- **Interpretación**:
  - $R^2 = 1$: El modelo explica perfectamente los datos.
  - $R^2 = 0$: El modelo no explica ninguna variación.
  - $R^2 < 0$: El modelo es peor que simplemente usar la media como predicción.

---

#### Ejemplo 
  --[**`Ejemplo 4.6`**](ejemplo4.6/ejemplo4_6.ipynb)
  
#### Ejercicio 4.3 (Alumnos)
  -[**`Ejercicio 4.3`**](ejercicio4.3/ejercicio4_3.ipynb)

#### Ejercicio Unificado
  -[**`Ejercicio Unificado`**](Ejercicio_Unificado_Modulo_4_Alumno.ipynb)

#### Ejercicio Final
  -[**`Ejercicio Final`**](Ejercicio_Final_Modelo_Machine_Learning_Student.ipynb)
