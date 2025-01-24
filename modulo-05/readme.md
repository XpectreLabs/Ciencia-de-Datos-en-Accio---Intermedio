# **Módulo 5: Feature Engineering y Optimización de Modelos (4 horas)**

## 🎯**Objetivos Específicos**
1. **Feature Engineering**: Crear y transformar características (variables) de manera eficiente para mejorar el rendimiento de los modelos.
2. **Selección y Reducción de Dimensionalidad**: Seleccionar las variables más relevantes y/o reducir la dimensionalidad utilizando técnicas como PCA u otros métodos avanzados.
3. **Optimización de Hiperparámetros**: Ajustar y optimizar los hiperparámetros de un modelo (e.g., Grid Search, Random Search) para obtener la mejor configuración posible.
4. **Evaluación de Modelos**: Evaluar y comparar distintas configuraciones de modelos para seleccionar la más adecuada en función del problema.

---

## **Tema 5.1: Feature Engineering**

### **Concepto y Ubicación en el Flujo de Trabajo**

**¿Qué es Feature Engineering?**  
Feature Engineering es el proceso de crear, transformar y seleccionar características o variables dentro de un dataset con el fin de mejorar la capacidad predictiva y el rendimiento general de los modelos de Machine Learning. Es una de las etapas más críticas en el flujo de trabajo de Machine Learning, ya que influye directamente en la calidad de los datos utilizados para entrenar los modelos.

Este proceso se lleva a cabo dentro del **preprocesamiento de datos**, antes de entrenar el modelo, y puede involucrar diversas tareas como la creación de nuevas variables, la transformación de las existentes o la eliminación de aquellas que no contribuyen al modelo.

### **Importancia de Feature Engineering**
- **Mejora significativa del rendimiento**: Las variables bien diseñadas pueden proporcionar al modelo información más relevante y útil, lo que se traduce en un mejor rendimiento, con métricas más altas como la precisión y el recall.
  
- **Captura de relaciones complejas**: Muchas veces, las relaciones entre las variables y la variable objetivo son complejas o no lineales. Feature Engineering permite representar estas relaciones de una manera que los modelos pueden entender.

- **Reducción de ruido**: Al seleccionar o transformar las variables adecuadas, se puede reducir el impacto de ruido o información irrelevante en el modelo, ayudando a evitar el sobreajuste (overfitting) y mejorando la generalización del modelo.

- **Mejora de la interpretabilidad del modelo**: Las variables transformadas o creadas correctamente pueden hacer que el modelo sea más fácil de interpretar, ayudando a entender los factores más importantes que afectan las predicciones.

#### **Ejemplo de impacto**:  
Supongamos que tenemos un dataset de ventas con variables como "ventas" y "precio_unitario". Si creamos una nueva variable "ingreso_total" multiplicando `ventas` por `precio_unitario`, esta nueva característica puede ofrecer una representación más precisa del rendimiento de las ventas, ayudando al modelo a identificar patrones de ingresos que no eran evidentes cuando tratábamos a `ventas` y `precio_unitario` por separado.

---

### **Otras Razones para Realizar Feature Engineering**
- **Preparación para Modelos Complejos**: En el caso de algoritmos que no manejan bien características categóricas o no lineales, el proceso de ingeniería de características puede preparar el dataset para obtener mejores resultados.

- **Adaptabilidad a diferentes tipos de datos**: Feature Engineering también permite adaptar los modelos a diferentes tipos de datos, como datos temporales, geoespaciales, o de texto, transformándolos en un formato adecuado para que los modelos puedan aprender de ellos.

- **Reducción de dimensionalidad**: A través de la creación de nuevas características o la selección de las más relevantes, es posible reducir la dimensionalidad de los datos, lo que ayuda a mejorar el rendimiento de los modelos, especialmente en situaciones con grandes volúmenes de datos.

### **Áreas Comunes de Feature Engineering**
1. **Transformación de Datos**: Involucra aplicar funciones matemáticas a las características, como la normalización, la estandarización, o la creación de logaritmos de las variables.
   
2. **Creación de Características Derivadas**: Se generan nuevas variables a partir de las existentes, como combinaciones entre variables o variables agregadas que capturan información relevante.

3. **Manejo de Variables Categóricas**: Convertir las variables categóricas a un formato adecuado (por ejemplo, One-Hot Encoding, Label Encoding) para que puedan ser utilizadas por modelos que no manejan directamente datos no numéricos.

4. **Manejo de Valores Faltantes**: Implica estrategias como la imputación de valores faltantes para asegurar que el modelo no se vea afectado por datos incompletos.

5. **Detección y Manejo de Outliers**: Identificar y tratar los valores atípicos (outliers) para evitar que distorsionen las predicciones del modelo.

6. **Variables Temporales**: Descomponer y extraer información útil de las variables temporales, como fechas, horas o ciclos estacionales.

---

### **Creación de Nuevas Variables**

#### **1. Transformaciones Matemáticas**
- Aplicar transformaciones para estabilizar la varianza o capturar relaciones no lineales.
- Ayudar a que los datos sigan una distribución más adecuada para los modelos.
- Mejorar el rendimiento de los modelos lineales en situaciones de relaciones no lineales.
- Minimizar el impacto de los valores atípicos o extremos.
- Facilitar la convergencia de algunos algoritmos de Machine Learning.
- Adaptar los datos a una escala más uniforme, mejorando la estabilidad numérica.
- Permitir la mejor representación de los datos cuando hay variaciones grandes en los valores.
- **¿Cuándo Aplicar Transformaciones Matemáticas?**
  - **Sesgo de Distribución**: Cuando las variables tienen distribuciones sesgadas que afectan negativamente el modelo.
  - **Relaciones No Lineales**: Cuando se quiere transformar la relación entre las características y la variable objetivo de no lineal a lineal.
  - **Escalado de Datos**: Cuando las características tienen diferentes rangos de magnitud y deben ser normalizadas o estandarizadas.

- **Ejemplo**:
```python
import numpy as np
df['log_ventas'] = np.log(df['ventas'] + 1)  # Evitar log(0)

```

#### **2. Interacciones entre Variables**
- Las interacciones entre variables consisten en crear nuevas columnas combinando variables existentes mediante operaciones matemáticas como la multiplicación, división, resta o suma. Esto ayuda a capturar relaciones complejas entre las características y puede mejorar significativamente el rendimiento del modelo.
- Permiten identificar patrones ocultos que no son evidentes a simple vista y que pueden influir de manera importante en la predicción.
- Es útil cuando se sospecha que el efecto de una variable depende del valor de otra variable.
- Ayuda a mejorar la precisión de modelos que no logran captar la relación entre variables de manera independiente.
- Ejemplo práctico: Si tienes una variable `edad` y `ingreso`, la interacción de ambas podría representar una nueva variable como `edad_por_ingreso`, lo que podría reflejar mejor ciertos patrones.
- También puede evitar la multicolinealidad entre variables independientes, si la combinación de estas da lugar a nuevas variables con menos redundancia.

- **Ejemplo:**
```python
df['ventas_x_precio'] = df['ventas'] * df['precio_unitario']
```
#### **3. Variables Categóricas**

**One-Hot Encoding**
- Este método convierte cada categoría en una columna binaria. Es útil cuando las categorías no tienen un orden inherente, ya que evita introducir un sesgo en el modelo. Cada columna resultante tiene un valor de 0 o 1 dependiendo de si el registro pertenece a esa categoría.
- One-Hot Encoding es adecuado para modelos que no pueden manejar datos categóricos directamente, como los modelos lineales o los árboles de decisión.
- La principal ventaja es que no impone ninguna relación ordinal entre las categorías, lo cual es esencial para evitar suposiciones incorrectas en el modelo.
- Sin embargo, puede aumentar significativamente la dimensionalidad del dataset cuando hay muchas categorías, lo que puede hacer que el modelo sea más costoso computacionalmente y más propenso al sobreajuste.
- **Ejemplo**: Si tienes una columna `color` con valores ["rojo", "verde", "azul"], One-Hot Encoding generará tres columnas: `color_rojo`, `color_verde` y `color_azul`. Un valor "rojo" se representaría como [1, 0, 0], "verde" como [0, 1, 0] y "azul" como [0, 0, 1].
- **Ventajas de One-Hot Encoding**:
  - Evita introducir sesgos en el modelo relacionados con el orden de las categorías.
  - Es útil para categorías nominales sin ninguna relación jerárquica o secuencial.
- **Desventajas de One-Hot Encoding**:
  - Aumenta la dimensionalidad del conjunto de datos, lo que puede llevar a una mayor complejidad computacional.
  - Puede causar problemas con modelos que no manejan bien la alta dimensionalidad.
- **Ejemplo de código:**
```python
# Convierte una columna categórica en múltiples columnas binarias
pd.get_dummies(df['region'], prefix='region')
```

**Label Encoding**
- En este enfoque, cada categoría se asigna un valor numérico único. Es adecuado cuando las categorías tienen un orden inherente o se desea reducir la dimensionalidad.
- A diferencia de One-Hot Encoding, Label Encoding introduce una relación ordinal entre las categorías, lo que puede ser útil si las categorías tienen un orden natural (por ejemplo, "bajo", "medio", "alto").
- **Desventaja**: Si las categorías no tienen un orden lógico, Label Encoding puede introducir sesgos en el modelo, ya que los valores numéricos asignados podrían ser interpretados como representaciones de magnitudes, lo cual no tiene sentido en algunas situaciones.
- **Ejemplo**: Si tienes una columna `tamaño` con valores ["pequeño", "mediano", "grande"], Label Encoding los transformará en [0, 1, 2], representando el orden implícito.
- **Ventajas de Label Encoding**:
  - No aumenta la dimensionalidad del conjunto de datos.
  - Es más eficiente computacionalmente para un número grande de categorías.
- **Desventajas de Label Encoding**:
  - Introduce un orden entre las categorías que puede ser inapropiado cuando no existe una relación ordinal entre ellas.
```python
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df['region_encoded'] = encoder.fit_transform(df['region'])
```

#### **4. Variables Temporales**

- **Extracción de componentes temporales**: Las variables temporales derivadas de fechas y horas pueden proporcionar información valiosa en muchos modelos predictivos. Al extraer componentes clave como el "mes", "día de la semana", "hora" o "año", se puede identificar patrones que están relacionados con el ciclo temporal.
  
- **Comportamiento estacional y tendencias**: El comportamiento de los datos puede cambiar según la temporada o la hora del día. Por ejemplo, el comportamiento de los consumidores puede ser diferente durante los días laborales frente a los fines de semana, o durante los meses festivos frente a los meses normales. Las variables temporales ayudan a capturar estos efectos.

- **Tendencias a largo plazo**: Además de los patrones estacionales y semanales, las variables temporales también pueden ayudar a identificar tendencias a largo plazo. Por ejemplo, la venta de ciertos productos podría seguir una tendencia creciente o decreciente con el paso de los años.

- **Variables temporales y machine learning**: Las características derivadas de fechas y tiempos son esenciales en áreas como la predicción de series de tiempo, la planificación de la demanda, y el análisis de eventos. Permiten que los modelos aprendan no solo de las relaciones directas entre las características y la variable objetivo, sino también de los patrones temporales que pueden influir en la predicción.

**Ventajas de las Variables Temporales:**
  - **Captura de patrones estacionales**: Ayuda a detectar comportamientos periódicos en los datos, como variaciones mensuales o semanales, lo que puede mejorar las predicciones en muchos tipos de modelos.
  - **Mejora en el rendimiento predictivo**: La inclusión de información temporal relevante puede mejorar la capacidad predictiva de los modelos, ya que muchas veces los comportamientos están influenciados por la hora del día, el mes del año o eventos recurrentes como los festivos.
    
**Desventajas de las Variables Temporales:**
  - **Dimensionalidad adicional**: La extracción de múltiples componentes temporales puede aumentar la dimensionalidad de los datos, lo que podría afectar el rendimiento del modelo si no se manejan adecuadamente.
  - **Complejidad en la interpretación**: La relación entre las variables temporales y la variable objetivo no siempre es lineal, lo que podría hacer que la interpretación de los resultados sea más compleja y requiera técnicas de modelado más sofisticadas.
  - **Necesidad de cuidado en el preprocesamiento**: Las fechas pueden requerir un manejo especial, especialmente si hay datos faltantes, inconsistencias o errores en las fechas. Esto puede requerir un preprocesamiento adicional antes de utilizarlas en el modelo.
  
**Ejemplo:**
```python
df['mes'] = df['fecha'].dt.month
df['es_festivo'] = df['fecha'].apply(lambda x: x in lista_festivos)
```

#### Ejemplo 5.1
-[**`Ejemplo 5.1`**](Ejemplo5_1.ipynb)

#### Ejercicio 5.1
-[**`Ejercicio Feature Engineering`**](Ejercicio5_1.ipynb)

---

## Tema 5.2: Selección de Características y Reducción de Dimensionalidad

### **Selección de Características (Feature Selection)**

La **selección de características** es un proceso esencial en el preprocesamiento de datos que busca identificar las características más relevantes para un modelo, eliminando aquellas que son redundantes o irrelevantes. Este paso puede mejorar el rendimiento del modelo, reducir la complejidad computacional y facilitar la interpretación de los resultados.

#### **Problema de Alta Dimensionalidad**
- **Rendimiento computacional reducido**: A medida que el número de características aumenta, los algoritmos de aprendizaje pueden volverse más lentos y requerir mayores recursos computacionales. Esto también puede aumentar el tiempo de entrenamiento y el riesgo de sobreajuste.
  
- **Dificultad para interpretar los modelos**: Los modelos con muchas variables son más difíciles de interpretar, lo que puede dificultar la comprensión de cómo las características están influyendo en las predicciones. En modelos complejos como las redes neuronales, esto se vuelve especialmente problemático.
  
- **Aumento del riesgo de sobreajuste (overfitting)**: Con demasiadas características, el modelo puede ajustarse demasiado a los datos de entrenamiento, perdiendo capacidad para generalizar a nuevos datos.

#### **Métodos de Selección de Características**
Existen varios enfoques para la selección de características, que se pueden agrupar en tres categorías: **basados en filtros**, **basados en envolventes (wrappers)** e **integrados**.

---

### **Métodos Basados en Filtros**
Estos métodos evalúan cada característica de forma independiente y la seleccionan o rechazan en función de su relevancia para el problema. No requieren un modelo de aprendizaje para ser aplicados.

- **Correlación**: Este método evalúa la relación entre las variables y la variable objetivo utilizando métricas como el coeficiente de correlación de Pearson para características numéricas. Se eliminan características altamente correlacionadas entre sí para reducir redundancia.
  
- **Test chi-cuadrado para variables categóricas**: Este test estadístico evalúa la dependencia entre dos variables categóricas. Si una variable categórica no tiene una relación significativa con la variable objetivo, puede ser descartada.

#### **Ventajas**:
- Rápido y computacionalmente eficiente.
- No requiere entrenamiento de un modelo.
- Facilita la identificación de características irrelevantes rápidamente.

#### **Desventajas**:
- No tiene en cuenta las interacciones entre las características.
- Puede no ser tan preciso en problemas complejos donde las interacciones entre variables son importantes.

#### **Ejemplo:**
```python
from sklearn.feature_selection import chi2
chi_scores = chi2(X, y)
```

---

### **Métodos Basados en Wrappers**
Los métodos de selección basada en envolventes usan un modelo de aprendizaje para evaluar la importancia de las características. Se seleccionan las características según su rendimiento en un modelo y se repite el proceso iterativamente.

- **Recursive Feature Elimination (RFE)**: RFE es un enfoque iterativo que elimina las características menos importantes según el rendimiento del modelo. En cada iteración, el modelo se entrena con las características restantes, y se elimina la característica menos relevante. Este proceso se repite hasta que se obtiene el conjunto óptimo de características.

#### **Ventajas**:
- Toma en cuenta la interacción entre las características.
- A menudo produce mejores resultados, ya que considera el rendimiento del modelo directamente.

#### **Desventajas**:
- Computacionalmente costoso, especialmente con muchos datos y características.
- Requiere entrenamiento de un modelo, lo que puede ser lento.

#### **Ejemplo:**
```python
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
selector = RFE(model, n_features_to_select=5)
selector.fit(X, y)
```


---

### **Métodos Integrados (Embedded)**
Los métodos integrados seleccionan características durante el proceso de entrenamiento del modelo. Esto significa que la selección de características se realiza automáticamente al mismo tiempo que el modelo aprende. Algunos algoritmos de machine learning tienen mecanismos integrados de selección de características.

- **Regularización (Lasso y Ridge)**:  
  - **Lasso (L1 regularization)**: Penaliza los coeficientes de las características para reducir algunos de ellos a cero, eliminando características irrelevantes. Es particularmente útil cuando hay muchas variables correlacionadas.
  - **Ridge (L2 regularization)**: Penaliza los coeficientes, pero no los elimina completamente, lo que permite manejar multicolinealidad sin reducir las características a cero.
  
- **Modelos de Árboles (como Random Forest y Gradient Boosting)**:  
  Los modelos basados en árboles tienen la capacidad de asignar una "importancia" a cada característica, basada en la mejora que produce en el rendimiento del modelo al dividir los datos. Las características menos importantes pueden ser eliminadas según esta medida de importancia.

#### **Ventajas**:
- Tienen en cuenta las interacciones entre características de manera eficiente.
- Suelen ser más eficientes computacionalmente que los métodos de envolvente.

#### **Desventajas**:
- Puede ser más difícil de interpretar en comparación con los métodos basados en filtros.
- Dependiente del modelo utilizado, puede no ser aplicable a todos los algoritmos.

```python
from sklearn.linear_model import Lasso
model = Lasso(alpha=0.1)
model.fit(X, y)
print(model.coef_)
```

---

### **Reducción de Dimensionalidad**

La **reducción de dimensionalidad** es el proceso de reducir el número de características (variables) en un conjunto de datos mientras se preserva la mayor cantidad de información posible. Este proceso es esencial para simplificar modelos, mejorar la eficiencia computacional y reducir el riesgo de sobreajuste.

#### **¿Por qué es importante?**
- **Reduce la complejidad computacional**: Menos características implican menor tiempo de procesamiento y menor memoria requerida.
- **Mejora la visualización**: Al reducir las dimensiones a 2 o 3, es posible visualizar los datos, lo que facilita la interpretación y el análisis.
- **Reduce el riesgo de sobreajuste**: Menos características pueden ayudar a evitar que el modelo se ajuste demasiado a los datos de entrenamiento.
- **Mejora la interpretabilidad**: Un modelo con menos características es más fácil de entender y analizar.

#### **PCA (Principal Component Analysis)**

**PCA** es uno de los métodos más comunes para la reducción de dimensionalidad. Su objetivo es transformar un conjunto de variables correlacionadas en un conjunto de **componentes principales** que son lineales, no correlacionados y que capturan la mayor parte de la varianza de los datos.

##### **¿Cómo funciona PCA?**
1. **Estandarización**: Los datos deben ser estandarizados para que todas las variables tengan la misma escala. Esto es crucial cuando las características tienen diferentes unidades de medida (por ejemplo, peso en kilogramos y altura en metros).
  
2. **Cálculo de la matriz de covarianza**: Se calcula la matriz de covarianza entre las variables, que describe cómo se relacionan entre sí.

3. **Obtención de los vectores propios y valores propios**: Los vectores propios representan las direcciones de mayor varianza en los datos, y los valores propios indican la cantidad de varianza que cada componente captura.

4. **Selección de los componentes principales**: Se seleccionan los componentes principales según la varianza que explican. Generalmente, se seleccionan los primeros componentes que explican un porcentaje significativo de la varianza, como el 80% o más, dependiendo de los requisitos del modelo.

5. **Proyección de los datos en el nuevo espacio**: Los datos originales se proyectan sobre los componentes principales seleccionados, reduciendo la dimensionalidad.

##### **Ventajas de PCA**:
- **Reducción de dimensionalidad**: Disminuye el número de variables manteniendo la mayor parte de la información.
- **Mejora del rendimiento del modelo**: Reduce la complejidad, lo que puede mejorar la precisión y reducir el tiempo de entrenamiento del modelo.
- **Mejora la visualización**: Al reducir las dimensiones a 2 o 3, es más fácil visualizar los datos.
  
##### **Desventajas de PCA**:
- **Dificultad de interpretación**: Los componentes principales son combinaciones lineales de las variables originales, lo que puede hacer que sea difícil interpretar los resultados directamente.
- **Pérdida de información**: Aunque PCA trata de retener la mayor varianza, siempre hay una pequeña pérdida de información al reducir las dimensiones.

##### **¿Cuándo usar PCA?**
- Cuando tienes un conjunto de datos con muchas características y deseas reducir la complejidad.
- Cuando las características están altamente correlacionadas.
- En modelos que pueden beneficiarse de la reducción de la dimensionalidad sin perder mucha información.

#### **Otras Técnicas de Reducción de Dimensionalidad**
- **t-SNE (t-Distributed Stochastic Neighbor Embedding)**: Es una técnica no lineal que es útil para la visualización de datos de alta dimensionalidad, pero no necesariamente preserva la varianza global.
- **LDA (Linear Discriminant Analysis)**: A diferencia de PCA, que es una técnica no supervisada, LDA se utiliza principalmente para tareas de clasificación, buscando reducir la dimensionalidad mientras mantiene la información que ayuda a separar las clases.

#### **Ejemplo**
```python
  from sklearn.decomposition import PCA
  pca = PCA(n_components=2)
  X_pca = pca.fit_transform(X)
  print("Varianza explicada:", pca.explained_variance_ratio_)
  ```

#### **Visualización de PCA:**
```python
import matplotlib.pyplot as plt
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
plt.title("Visualización PCA")
plt.show()
```

#### Ejemplo 5.2
-[**`Ejemplo 5.2`**](ejemplo5.2/Ejemplo5_2.ipynb)

#### Ejercicio 5.2
-[**`Ejercicio Selección de Características y Reducción de Dimensionalidad`**](ejercicio5.2/Ejercicio5_2.ipynb)

---

### **Tema 5.3: Optimización de Modelos (Hyperparameter Tuning)**

La optimización de modelos, también conocida como **ajuste de hiperparámetros**, es un proceso crucial para mejorar el rendimiento de un modelo de machine learning. Los hiperparámetros son configuraciones que determinan la estructura del modelo y cómo se entrena, pero no se ajustan durante el entrenamiento en sí.

#### **Introducción a la Optimización de Hiperparámetros**

Los **hiperparámetros** pueden tener un gran impacto en el rendimiento de un modelo. Ajustarlos adecuadamente permite que el modelo se adapte mejor a los datos y, por lo tanto, generalice mejor a datos no vistos. El ajuste de hiperparámetros se realiza antes del entrenamiento y afecta aspectos como la complejidad del modelo, la velocidad de convergencia y la capacidad de generalización.

##### **Parámetros vs Hiperparámetros**  
- **Parámetros**: Son los valores que se ajustan durante el entrenamiento del modelo. En un modelo de regresión, por ejemplo, los **coeficientes** de las variables son parámetros. El modelo aprende estos valores a partir de los datos.
  
- **Hiperparámetros**: Son valores configurados antes de que comience el entrenamiento y determinan el comportamiento del modelo. Ejemplos incluyen:
  - **Profundidad de un árbol** en un modelo de árbol de decisión.
  - **Número de vecinos (k)** en un clasificador K-Nearest Neighbors (K-NN).
  - **Tasa de aprendizaje** en algoritmos de optimización como el descenso por gradiente.
  
#### **Métodos de Búsqueda de Hiperparámetros**

Existen diferentes técnicas para encontrar los mejores hiperparámetros. Las dos más comunes son **Grid Search** y **Random Search**.

##### **Grid Search (Búsqueda en Rejilla)**

**Grid Search** realiza una búsqueda exhaustiva, probando todas las combinaciones posibles de hiperparámetros dentro de un espacio predefinido. Esta técnica garantiza que se consideren todas las combinaciones posibles, lo que puede ser muy útil, pero también puede ser **computacionalmente costosa**.

###### **Ventajas**:
- Garantiza encontrar la mejor combinación de hiperparámetros dentro del espacio definido.
- Método sistemático y exhaustivo.

###### **Desventajas**:
- **Costoso en tiempo**: Si el número de hiperparámetros y sus valores es grande, la búsqueda puede llevar mucho tiempo y recursos computacionales.
- No es práctico para modelos con muchos hiperparámetros o para grandes datasets.

###### **Ejemplo**:
Supón que tienes dos hiperparámetros: la profundidad de un árbol (`max_depth`) y el número de árboles (`n_estimators`). Si `max_depth` puede tomar 3 valores (1, 2, 3) y `n_estimators` 2 valores (50, 100), Grid Search probaría todas las combinaciones posibles (1x2 = 6 combinaciones).

###### **Ejemplo código**:
```python
from sklearn.model_selection import GridSearchCV
param_grid = {'max_depth': [3, 5, 10], 'n_estimators': [50, 100]}
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid_search.fit(X, y)
```

##### **Random Search (Búsqueda Aleatoria)**

**Random Search** selecciona aleatoriamente un subconjunto de combinaciones de hiperparámetros dentro de un rango definido. Aunque no garantiza que se encuentren las mejores combinaciones, puede ser **mucho más eficiente** en términos de tiempo computacional cuando se exploran muchos hiperparámetros.

###### **Ventajas**:
- **Menor tiempo de computación**: Al probar aleatoriamente solo un subconjunto de combinaciones, suele ser más rápido que Grid Search.
- **Menos costoso computacionalmente**: Requiere menos evaluaciones, por lo que es más eficiente cuando los hiperparámetros tienen un espacio de búsqueda grande.

###### **Desventajas**:
- **No garantiza encontrar la mejor combinación**: Puede que no se exploren todas las opciones posibles.
- **Dependencia del número de iteraciones**: La calidad de los resultados depende de cuántas iteraciones se realicen.

###### **Ejemplo**:
Si usas Random Search para los mismos hiperparámetros que en el ejemplo anterior, podrías seleccionar aleatoriamente 3 combinaciones de `max_depth` y `n_estimators`, en lugar de probar todas las combinaciones posibles.

###### **Ejemplo código**:
```python
from sklearn.model_selection import RandomizedSearchCV
random_search = RandomizedSearchCV(RandomForestClassifier(), param_distributions=param_grid, n_iter=10, cv=5)
random_search.fit(X, y)
```

#### **Técnicas Avanzadas de Optimización**

Existen métodos más avanzados para la optimización de hiperparámetros que son más eficientes que Grid y Random Search:

##### **1. Bayesian Optimization**
La **optimización bayesiana** utiliza un enfoque probabilístico para explorar los hiperparámetros, modelando la función de pérdida y optimizándola de manera iterativa. Esta técnica es más eficiente, ya que no prueba todas las combinaciones, sino que prioriza las combinaciones que tienen más probabilidades de mejorar el rendimiento.

###### **Ventajas**:
- Menos iteraciones que Grid y Random Search.
- Más eficiente y rápido en términos de búsqueda de hiperparámetros.

##### **2. Optimización por Enjambre de Partículas (PSO)**
El algoritmo de **optimización por enjambre de partículas** es una técnica inspirada en la naturaleza (específicamente en el comportamiento de los enjambres de aves) que busca la mejor solución en el espacio de hiperparámetros a través de un proceso de búsqueda distribuido.

###### **Ventajas**:
- Puede encontrar soluciones globales de manera eficiente en problemas complejos.
- Es especialmente útil en problemas con muchas variables no lineales.

##### **3. Algoritmos Evolutivos**
Los **algoritmos evolutivos** son técnicas inspiradas en la evolución biológica, donde se realizan mutaciones y cruces entre un conjunto de soluciones para generar nuevas combinaciones de hiperparámetros. Esto permite explorar el espacio de soluciones de manera efectiva.

###### **Ventajas**:
- Capaz de manejar grandes espacios de búsqueda de hiperparámetros.
- Buena capacidad para evitar quedar atrapado en mínimos locales.

#### Ejemplo 5.3
-[**`Ejemplo 5.3`**](ejemplo5.3/Ejemplo5_3.ipynb)

#### Ejercicio 5.3
-[**`Ejercicio Optimización de Hiperparámetros con Random Forest`**](ejercicio5.3/Ejercicio5_3.ipynb)

---

### **Sección 5.4: Evaluación, Comparación y Selección de Modelos**

La evaluación, comparación y selección de modelos son pasos fundamentales para asegurar que el modelo final sea el más adecuado para el problema en cuestión. Esto implica no solo evaluar el rendimiento del modelo, sino también analizar su capacidad para generalizar a nuevos datos, su interpretabilidad y la importancia de las características utilizadas en la predicción.

#### **Comparación de Distintas Estrategias**
La comparación de diferentes estrategias y enfoques es esencial para determinar la efectividad de un modelo y sus ajustes. Algunas estrategias clave incluyen:

- **Evaluar el modelo antes y después de la ingeniería de características**:
  - La ingeniería de características puede mejorar significativamente el rendimiento de un modelo. Evaluar cómo cambia el rendimiento antes y después de realizar transformaciones o combinaciones de características (como la creación de nuevas variables, normalización o eliminación de características irrelevantes) permite observar el impacto directo en la precisión del modelo.
  
- **Comparar resultados del modelo sin optimización vs con optimización de hiperparámetros**:
  - A menudo, un modelo inicial puede ser útil, pero no alcanza su máximo potencial hasta que se optimizan los hiperparámetros. Comparar el rendimiento de un modelo antes de aplicar técnicas de optimización de hiperparámetros (como Grid Search o Random Search) y después de aplicar esas técnicas puede proporcionar información importante sobre la efectividad de la búsqueda de hiperparámetros y el ajuste de los parámetros del modelo.

#### **Curva de Aprendizaje (Learning Curve)**
La curva de aprendizaje es una herramienta valiosa para evaluar el rendimiento de un modelo mientras se entrena con diferentes tamaños de datos. Este análisis es crucial para entender cómo un modelo puede mejorar con más datos o ajustes en el entrenamiento.

- **Análisis de sesgo y varianza con relación al tamaño del conjunto de entrenamiento**:
  - **Sesgo**: Si la curva muestra un alto error en los datos de entrenamiento y prueba, esto sugiere un sesgo alto, lo que significa que el modelo está subajustado (underfitting).
  - **Varianza**: Si el modelo tiene un buen rendimiento en los datos de entrenamiento pero un bajo rendimiento en los datos de prueba, indica alta varianza, lo que sugiere sobreajuste (overfitting).
  - El análisis de la curva de aprendizaje permite identificar si el modelo está aprendiendo correctamente y si se necesita ajustar el tamaño del conjunto de datos o utilizar más regularización.

- **Interpretación de resultados: ¿El modelo necesita más datos, más regularización?**:
  - **Si el modelo muestra sobreajuste (overfitting)**: Se pueden añadir más datos de entrenamiento, aplicar técnicas de regularización (como **Lasso** o **Ridge**) o utilizar métodos como el **dropout** en redes neuronales.
  - **Si el modelo muestra subajuste (underfitting)**: Se pueden incluir más características, ajustar los hiperparámetros o cambiar a un modelo más complejo.

#### **Interpretabilidad de Modelos**
La interpretabilidad es crucial cuando se quiere entender cómo un modelo toma decisiones, especialmente en aplicaciones sensibles como la medicina, la finanza o el derecho.

- **Importancia de Variables (Feature Importances) en Modelos Tipo Árbol**:
  - En modelos basados en árboles como **Random Forest** o **XGBoost**, la importancia de las características se calcula evaluando cuánto contribuye cada variable a la reducción de la impureza en cada división del árbol. Esto ayuda a identificar qué características tienen más influencia en las predicciones del modelo y puede ser útil para la selección de características o para la interpretación del modelo.
  
- **Métodos como SHAP o LIME para Modelos Más Complejos (Opcional)**:
  - **SHAP (SHapley Additive exPlanations)**: Proporciona explicaciones detalladas sobre cómo cada característica contribuye a una predicción específica. Es una extensión de la teoría de juegos y proporciona una forma robusta y precisa de descomponer las predicciones de un modelo en términos de sus características.
  
  - **LIME (Local Interpretable Model-Agnostic Explanations)**: Permite interpretar modelos complejos (como redes neuronales) de manera local, analizando el modelo alrededor de una instancia específica. LIME genera un modelo más simple y comprensible para explicar cómo el modelo complejo llega a su decisión en ese caso particular.

  Estos métodos permiten que incluso los modelos más complejos, como redes neuronales profundas o modelos de ensemble, sean más comprensibles y accesibles para los usuarios, mejorando la confianza en las predicciones del modelo.

###### **Ejemplo código**:
```python
from sklearn.model_selection import learning_curve
train_sizes, train_scores, valid_scores = learning_curve(RandomForestClassifier(), X, y, cv=5)
plt.plot(train_sizes, train_scores.mean(axis=1), label='Entrenamiento')
plt.plot(train_sizes, valid_scores.mean(axis=1), label='Validación')
plt.legend()
plt.show()
```

#### Ejemplo 5.4
-[**`Ejemplo 5.4`**](ejemplo5.4/Ejemplo5_4.ipynb)

#### Ejercicio 5.4
-[**`Ejercicio Evaluación y Comparación de Modelos`**](ejercicio5.4/Ejercicio5_4.ipynb)

#### Actividad Final
-[**`Actividad Final`**](actividad_final/Actividad_Final5.ipynb)
