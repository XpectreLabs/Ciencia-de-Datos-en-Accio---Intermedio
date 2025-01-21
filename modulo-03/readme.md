# Módulo 3: Análisis Exploratorio de Datos (EDA)

¡Bienvenido/a a este fascinante módulo sobre el Análisis Exploratorio de Datos (EDA)! Aquí aprenderás a mirar datos con ojos de detective, encontrando pistas y patrones que nos ayuden a entender mejor cualquier información que tengamos. Piensa que los datos son como un tesoro escondido y, con la ayuda de algunas herramientas, podemos descubrir sus secretos.

---

## 🎯Objetivo del Módulo

Al finalizar este módulo:

- Serás capaz de examinar los datos que te entreguen y encontrar patrones, tendencias o relaciones interesantes.
- Podrás usar gráficos y estadísticas para contar la historia que esconden tus datos.
- Podrás hacer preguntas más inteligentes sobre tu información y comunicar esos descubrimientos a otras personas.

Este módulo está dividido en dos partes principales (Hora 1 y Hora 2), para que puedas ir aprendiendo poco a poco y practicando cada paso.

---

## Tema 3: Introducción al Análisis Exploratorio

### Tema 3.1: ¿Qué es el Análisis Exploratorio de Datos (EDA)?

#### 1. Definición de EDA

Imagina que tienes una gran caja llena de Legos de diferentes colores, formas y tamaños. Antes de ponerte a construir algo, querrás revisar cuántas piezas tienes de cada color, cuáles son las más grandes, cuáles están rotas, cuáles encajan perfectamente, etc. El Análisis Exploratorio de Datos (EDA) es ese primer vistazo que le damos a nuestros datos para comprender su estructura, identificar problemas y determinar qué tan "limpios" o "desordenados" están. Además, el EDA permite conocer la naturaleza de los datos antes de aplicar modelos predictivos o análisis más profundos.

- El EDA es el conjunto de técnicas y métodos utilizados para explorar y entender los datos de forma preliminar, antes de proceder con el modelado. Incluye la inspección visual, el uso de estadísticas descriptivas y las herramientas gráficas para identificar patrones, errores, y relaciones entre las variables.
- **Objetivo fundamental**: Observar y descubrir la naturaleza de los datos, su distribución, y las posibles tendencias antes de cualquier análisis más complejo.

- **Principales actividades del EDA**:
  - **Revisión inicial**: Examinar las características generales de los datos, incluyendo el tipo de variables, la calidad, la presencia de datos faltantes o duplicados.
  - **Exploración estadística**: Calcula medidas como la media, mediana, desviación estándar, cuartiles y otros estadísticos para entender el comportamiento de las variables.
  - **Visualización gráfica**: Utiliza herramientas gráficas para representar los datos, como histogramas, diagramas de dispersión, cajas de bigote (boxplots), heatmaps, etc.
  - **Identificación de valores atípicos**: Detecta datos que no se ajustan al patrón general, utilizando métodos como los Z-scores o los IQR (Intervalo Intercuartílico).
  - **Relaciones entre variables**: Estudia la relación entre diferentes atributos mediante correlaciones estadísticas, regresiones preliminares y gráficos de dispersión.
  - <img src="https://gravitar.biz/wp-content/uploads/2024/02/8-1.png" alt="Descripción de la imagen" width="700">
      

#### 2. Objetivos del EDA

El Análisis Exploratorio de Datos tiene múltiples objetivos que permiten obtener una comprensión profunda antes de realizar análisis más avanzados o modelado.

- **Detectar tendencias generales**: Este paso implica identificar cómo varían las variables con el tiempo o con el cambio en otras condiciones. Por ejemplo, observar si las ventas de un producto están en aumento o en descenso en función de un periodo determinado (mensual, anual).
- **Identificar valores atípicos**: Los valores atípicos o anomalías pueden surgir debido a errores de entrada, registros falsos o simplemente debido a diferencias significativas en los datos. Detectar estos valores es crucial para limpiar los datos antes de cualquier análisis.
- **Comprender relaciones**: Un aspecto clave del EDA es el análisis de cómo diferentes variables están relacionadas entre sí. Por ejemplo, puede analizarse si existe una relación significativa entre el nivel de exposición al sol y las ventas de helados. Esto ayuda a prever posibles asociaciones antes de realizar modelado.

#### 3. Pasos comunes en el EDA

El EDA se compone de varios pasos secuenciales que ayudan a desentrañar la información contenida en los datos:

1. **Exploración inicial**:
   - Consiste en realizar una revisión rápida de los datos para identificar errores evidentes, registros duplicados, datos faltantes, o columnas que parecen irrelevantes.
   - Incluye inspeccionar el tamaño del dataset, la distribución de las variables, la presencia de datos desbalanceados o datos que puedan sesgar el análisis.

2. **Resumen estadístico**:
   - Calcula medidas descriptivas como la media (promedio), la mediana, la desviación estándar, el mínimo, el máximo, los cuartiles, etc.
   - Estos estadísticos ayudan a tener una primera aproximación al comportamiento de las variables y su distribución.

3. **Visualización de datos**:
   - Es uno de los pilares más fuertes del EDA, ya que ayuda a identificar patrones o relaciones visuales que son difíciles de observar solo con estadísticas.
   - Se usan gráficos como histogramas, diagramas de dispersión, cajas de bigote (boxplots), gráficos de líneas, heatmaps, etc., para representar visualmente los datos.

4. **Detección de anomalías**:
   - A través del uso de estadísticos como los Z-scores, o mediante el análisis del IQR (Intervalo Intercuartílico), se identifican los valores atípicos que no se ajustan a la distribución general de los datos.

5. **Análisis de correlaciones**:
   - Mediante el cálculo de coeficientes de correlación, como Pearson o Spearman, se estudian las dependencias entre las variables. Esto permite identificar posibles relaciones lineales o no lineales antes de construir modelos.

6. **Análisis de componentes**:
   - Utilización de técnicas como PCA (Análisis de Componentes Principales) para reducir la dimensionalidad del espacio de datos y visualizar la estructura subyacente de los datos.

#### 4. Herramientas comunes para EDA

Las herramientas son esenciales para realizar EDA de manera eficiente. A continuación se describen algunas de las más utilizadas:

- **Python**:
  - Librerías como Pandas para manipular datos, Numpy para cálculos numéricos, Matplotlib y Seaborn para visualización gráfica, y Plotly para gráficos interactivos.
  
- **R**:
  - R es uno de los lenguajes más populares para realizar EDA, con paquetes como ggplot2 para gráficos, dplyr y tidyr para manipulación de datos, y H2O para análisis avanzados.

- **Excel**:
  - Aunque es más básico, Excel permite realizar un análisis inicial con tablas dinámicas, gráficos básicos y funciones estadísticas.

- **Power BI** y **Tableau**:
  - Herramientas avanzadas que permiten realizar análisis interactivo y visualizaciones avanzadas con capacidad para procesar grandes volúmenes de datos.

#### 5. Casos de Uso del EDA

El EDA es aplicable en múltiples dominios y casos de uso:

- **Ciencia de Datos**:
  - Se utiliza para preprocesar los datos antes de cualquier análisis predictivo o modelado. Permite limpiar los datos, entender su naturaleza, y seleccionar las variables más relevantes para el análisis.
  
- **Análisis de Ventas**:
  - Permite conocer el comportamiento de los clientes, la demanda de productos, y las tendencias del mercado para tomar decisiones informadas en estrategias de marketing y ventas.

- **Análisis financiero**:
  - El EDA es fundamental para identificar patrones en los datos financieros, detectar anomalías en transacciones, y evaluar el riesgo crediticio antes de realizar análisis profundos.

- **Investigación académica**:
  - Ayuda a los investigadores a explorar datos recopilados en estudios de campo, experimentos sociales o científicos, y a generar hipótesis basadas en observaciones iniciales.

#### 6. Tipos de análisis en EDA

El Análisis Exploratorio de Datos (EDA) abarca varios enfoques que permiten explorar los datos desde diferentes perspectivas, ayudando a identificar patrones, anomalías y relaciones antes de proceder con modelados más complejos. A continuación se describen los tipos principales:

- **Estadístico**:
  - **Media (promedio)**:
    - Es la suma total de todos los valores dividida entre el número de valores. Nos proporciona una medida central que indica el punto medio de una distribución. Por ejemplo, si tienes los valores 5, 7 y 9, su promedio se calcula como (5+7+9)/3 = 7.
  - **Mediana**:
    - Es el número que queda en el centro cuando los valores están ordenados de menor a mayor. Si tienes los valores 3, 7, 9, 15, su mediana será 7, ya que es el número que se encuentra en el medio.
  - **Moda**:
    - Es el valor que más se repite en un conjunto de datos. Por ejemplo, en el conjunto {3, 7, 7, 9, 9, 9}, la moda sería 9.
  - **Desviación estándar**:
    - Indica qué tan dispersos están los datos alrededor del promedio. Un alto valor de desviación estándar sugiere que los datos están más alejados del promedio, mientras que un bajo valor indica que los datos están más concentrados en torno al promedio. Se calcula mediante la raíz cuadrada de la varianza.

- **Visual**:
  - **Gráfico de barras (bar plot)**:
    - Se utiliza para comparar las frecuencias de categorías o grupos diferentes, representando las cantidades mediante barras.
    - <img src="https://www.jmp.com/es_mx/statistics-knowledge-portal/exploratory-data-analysis/bar-chart/_jcr_content/par/styledcontainer_2069/par/image_1203777138.img.png/1594745267192.png" alt="Descripción de la imagen" width="700">
  - **Histograma**:
    - Es un tipo especial de gráfico que muestra la distribución de los valores en un conjunto de datos, ayudando a visualizar la frecuencia con la que ocurren ciertos valores. Ideal para ver cómo se distribuyen los datos en rangos o intervalos.
    - <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/Histogram_example.svg/1200px-Histogram_example.svg.png" alt="Descripción de la imagen" width="700">
  - **Gráfico de dispersión (scatter plot)**:
    - Es útil para mostrar la relación entre dos variables continuas. Cada punto en el gráfico representa una combinación de valores, facilitando la observación de patrones o tendencias.
    - <img src="https://aprendiendocalidadyadr.com/wp-content/uploads/2017/05/Dispersion-con-regresion.png" alt="Descripción de la imagen" width="700">
  - **Boxplot (diagrama de cajas)**:
    - Este gráfico muestra la distribución de los datos a través de sus cuartiles, ayudando a visualizar valores atípicos, el rango intercuartílico y la dispersión de los datos. Es especialmente útil para identificar si existen valores atípicos.
    - <img src="https://i0.wp.com/statisticsbyjim.com/wp-content/uploads/2019/01/boxplot_teaching.png?resize=576%2C384" alt="Descripción de la imagen" width="700">

Además de estos, en EDA se pueden realizar otros análisis visuales como los heatmaps (para identificar correlaciones mediante colores), los gráficos de líneas para observar tendencias a lo largo del tiempo, y los dendrogramas para ver agrupaciones jerárquicas.

#### 7. Tipos de datos en EDA

En el EDA es esencial diferenciar entre los tipos de datos presentes en los conjuntos para realizar un análisis adecuado:

- **Datos categóricos**:
  - Aquellos datos que pueden clasificarse en categorías, como el color de un producto (rojo, azul, verde), el género de una persona (hombre, mujer) o el tipo de producto (comida, ropa, tecnología).
- **Datos continuos**:
  - Aquellos datos que pueden tomar cualquier valor dentro de un rango, como la edad, la temperatura, las ventas diarias o las horas trabajadas.

- **Datos numéricos**:
  - **Discretos**:
    - Aquellos que solo pueden tomar ciertos valores enteros, como el número de estudiantes en un salón de clases o el número de productos vendidos.
  - **Continuos**:
    - Aquellos que pueden tomar cualquier valor dentro de un intervalo, como el peso, la altura o el tiempo.

- **Datos ordinales**:
  - Son aquellos datos que poseen un orden, pero la diferencia entre los valores no es uniforme, como las calificaciones del nivel de satisfacción (bueno, regular, malo).

- **Datos nominales**:
  - Son aquellos datos que simplemente categorizan sin implicar un orden, como los colores, los géneros o los tipos de productos.

Este entendimiento profundo de los tipos de datos es crucial para aplicar los análisis correctos y evitar errores en la interpretación de los resultados.

---

#### 4. Ejemplo 1.1
-[**`Ejemplo 1.1`**](ejercicio1.1/Ejercicio1_1_modulo3.ipynb)

#### 5. Ejemplo 1.2
-[**`Ejemplo 1.2`**](ejercicio1.2/Ejercicio1_2_modulo3.ipynb)

#### 6. Ejemplo 1.3
-[**`Ejemplo 1.3`**](ejercicio1.3/Ejercicio1_3_modulo3.ipynb)

#### 7. Ejercicio 1.4 (Alumnos)
-[**`Ejercicio 1.4`**](ejercicio1.4/Ejercicio_1_4_modulo3.ipynb)

---

### Tema 3.2: Herramientas de Visualización

#### 1. Librerías principales para EDA en Python

- **Matplotlib**:
  - **Descripción**:  
    Matplotlib es una de las librerías más fundamentales para crear visualizaciones en Python. Ofrece una amplia gama de funcionalidades para generar gráficos estáticos, interactivos y animados. A pesar de ser una librería básica, cuenta con un alto nivel de personalización, lo que la hace muy versátil para diferentes tipos de análisis visuales.  
  - **Funciones clave**:  
    - **plt.plot()**: Crea gráficos de líneas para mostrar la relación entre dos variables continuas.
    - **plt.bar()**: Permite la creación de gráficos de barras para comparar frecuencias o cantidades.
    - **plt.hist()**: Crea histogramas para visualizar la distribución de datos.
    - **plt.scatter()**: Gráficos de dispersión para observar la relación entre dos variables continuas.
    - **plt.subplot()**: Permite dividir una figura en múltiples subgráficos para comparar varias visualizaciones.
  - **Ventajas**:  
    - Gran flexibilidad en la personalización de gráficos mediante opciones de estilo, color y formato.
    - Puede adaptarse fácilmente para generar cualquier tipo de gráfico mediante el uso de funciones subyacentes.
  - **Casos de uso**:  
    - Visualización básica de datos como tendencias, distribuciones y relaciones entre variables.
    - Creación de gráficos para reportes, publicaciones científicas o presentaciones.
  - <img src="https://coding-blocks.github.io/DS-NOTES/_images/matplotlib1.png" alt="Descripción de la imagen" width="700">

- **Seaborn**:
  - **Descripción**:  
    Seaborn es una librería avanzada basada en Matplotlib, diseñada para facilitar la creación de gráficos estadísticos atractivos visualmente. Proporciona herramientas de visualización que son fáciles de usar, optimizadas para explorar datos de manera estética, y para identificar patrones y relaciones entre variables.
  - **Funciones clave**:  
    - **sns.heatmap()**: Crea mapas de calor que muestran correlaciones entre variables. Ideal para identificar asociaciones entre atributos.
    - **sns.violinplot()**: Visualiza la distribución de los datos a través de los "violines", permitiendo ver la densidad y dispersión de los datos.
    - **sns.pairplot()**: Crea múltiples gráficos en pares para comparar varias variables al mismo tiempo, ayudando a visualizar relaciones entre ellas.
    - **sns.regplot()**: Realiza regresiones lineales y otros tipos de ajustes en los gráficos.
    - **sns.boxplot()**: Representa la distribución de los datos a través de cuartiles, ayudando a identificar valores atípicos.
  - **Ventajas**:  
    - Visualizaciones más estéticas que mejoran la interpretación de los datos.
    - Integración con estadísticas avanzadas que facilita la exploración de patrones.
    - Proporciona estilos predefinidos y optimiza las visualizaciones para mostrar distribuciones y relaciones.
  - **Casos de uso**:  
    - Visualización avanzada de distribuciones de datos, correlaciones y agrupamientos.
    - Exploración estética de datos categóricos y continuos para entender patrones ocultos.
  - <img src="https://seaborn.pydata.org/_images/introduction_29_0.png" alt="Descripción de la imagen" width="700">

---

### 2. Gráficos comunes en EDA

- **Histogramas**:
  - **Descripción**:  
    Los histogramas dividen los datos en intervalos o rangos (bines) y muestran la frecuencia de los datos en cada uno de estos rangos. Son una herramienta fundamental para visualizar la distribución de los datos y para detectar patrones como picos, simetrías, o la concentración de datos en ciertos valores.
  - **Ventajas**:
    - Permiten observar la distribución de los datos y detectar si siguen una distribución normal, sesgada, o con picos múltiples.
    - Ayudan a identificar la frecuencia con la que se repiten los valores en diferentes intervalos.
    - Son útiles para visualizar la densidad y la dispersión de datos numéricos.
  - **Casos de uso**:
    - Visualizar la concentración de los datos en ciertos rangos.
    - Identificar distribuciones (normales, sesgadas o con picos múltiples).
    - Detectar la dispersión de los datos y la presencia de agrupamientos.
  - <img src="https://r-charts.com/es/distribucion/histograma-frecuencias_files/figure-html/color-histograma-frecuencias.png" alt="Descripción de la imagen" width="700">

- **Scatter plots**:
  - **Descripción**:  
    Los gráficos de dispersión muestran la relación entre dos variables continuas. Cada punto en el gráfico representa una combinación de valores, facilitando la observación de patrones, tendencias o posibles correlaciones.
  - **Ventajas**:
    - Permiten visualizar de manera gráfica la relación entre dos variables, ayudando a identificar patrones lineales o no lineales.
    - Son útiles para explorar la existencia de dependencias entre dos variables, así como para detectar posibles outliers.
  - **Casos de uso**:
    - Observar la relación entre dos variables continuas, como el ingreso y la edad de una población.
    - Identificar si dos variables están correlacionadas.
    - Detectar posibles valores atípicos que distorsionen la relación entre las variables.
  - <img src="https://www.health.state.mn.us/communities/practice/resources/phqitoolbox/images/scatter_ex_atlanticcities.jpg" alt="Descripción de la imagen" width="700">


- **Boxplots**:
  - **Descripción**:  
    Los boxplots, o diagramas de cajas, muestran la dispersión y la concentración de los datos en relación a su rango, mediana, cuartiles, y valores atípicos. La caja representa el rango intercuartílico (IQR), y los bigotes (o whiskers) indican los valores mínimo y máximo, excluyendo los outliers.
  - **Ventajas**:
    - Ofrecen una visualización compacta que permite identificar la dispersión de los datos, los valores centrales (mediana) y los posibles outliers.
    - Son útiles para comparar la dispersión de los datos entre diferentes grupos o categorías.
  - **Casos de uso**:
    - Visualizar la dispersión de datos numéricos y los valores centrales.
    - Comparar la distribución de los datos entre diferentes categorías o grupos, como por género o por región.
    - Identificar la presencia de valores atípicos en los datos.
  - <img src="https://datatab.es/assets/tutorial/create_box_plot_online.png" alt="Descripción de la imagen" width="700">



#### 3. Ejemplo de uso con Python

    ```python
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Datos de ejemplo
    data = [5, 7, 8, 7, 5, 6, 8, 9, 6, 7]
    
    # Creación de un boxplot con Matplotlib:
    plt.boxplot(data)
    plt.title("Boxplot de Ejemplo")
    plt.show()

#### 4. Ejemplo 2.1
-[**`Ejemplo 2.1`**](ejercicio2.1/Ejercicio2_1_modulo3.ipynb)

#### 5. Ejemplo 2.2
-[**`Ejemplo 2.2`**](ejercicio2.2/Ejercicio2_2_modulo3.ipynb)

#### 6. Ejemplo 2.3
-[**`Ejemplo 2.3`**](ejercicio2.3/Ejercicio2_3_modulo3.ipynb)

#### 7. Ejercicio 2.4 (Alumnos)
-[**`Ejercicio 2.4`**](ejercicio2.4/Ejercicio2_4_modulo3.ipynb)

---

### Tema 3.3: Aplicaciones Prácticas del EDA

Ahora que sabemos qué es el EDA y cómo hacer algunos gráficos, veamos cómo todo esto nos sirve en el mundo real.

#### 1. Análisis de correlación

La correlación nos dice si dos variables se mueven juntas (por ejemplo, temperatura y consumo de helados). Existen varios métodos para medir la relación entre dos variables, siendo el **coeficiente de correlación de Pearson** uno de los más comunes.

- **Coeficiente de correlación de Pearson**:
  - Es un número que varía entre -1 y 1 y nos ayuda a entender la intensidad de la relación entre dos variables.
  - **Cerca de 1**: Relación positiva muy fuerte, lo que indica que a medida que una variable aumenta, la otra también tiende a subir.
  - **Cerca de -1**: Relación negativa muy fuerte, indicando que cuando una variable sube, la otra disminuye.
  - **Cerca de 0**: Indica que no hay relación o que es muy débil entre las variables.

- **Heatmaps (Mapas de calor)**:
  - Los heatmaps muestran visualmente múltiples correlaciones usando colores para indicar la intensidad de la relación. Tonos más intensos reflejan correlaciones fuertes, mientras que tonos más suaves indican correlaciones débiles o inexistentes.
  - Muy útiles para identificar rápidamente cómo diferentes factores influyen en un objetivo común, como las ventas o la rentabilidad.
  - Podemos usar heatmaps (mapas de calor) para ver varias correlaciones a la vez.
  - <img src="https://miro.medium.com/v2/resize:fit:1400/1*POcUcVvwN3okrXt--IJ2Hw.png" alt="Descripción de la imagen" width="700">

     ```python
        import seaborn as sns
        
        # Supongamos que tenemos un DataFrame llamado df
        # con columnas como 'edad', 'ingresos', 'gastos', 'ahorros'
        correlaciones = df.corr()
        
        sns.heatmap(correlaciones, annot=True, cmap="coolwarm")
        plt.title("Mapa de calor de Correlaciones")
        plt.show()

#### 2. Análisis multivariable

Si tenemos muchas columnas en nuestros datos (por ejemplo, edad, ingresos, gastos, ahorros, número de hijos…), el **Análisis multivariable** nos permite entender cómo se relacionan varias variables al mismo tiempo. Para visualizar estas relaciones, podemos utilizar gráficos como **pair plots** y **matrices de correlación**.

- **Pair plots**:
  - Son gráficos de dispersión que muestran todas las posibles combinaciones de pares de variables continuas. Cada gráfico muestra cómo se relacionan dos variables específicas, pero al mismo tiempo, al agrupar varios gráficos, nos da una visión general de todas las relaciones multivariables.
  - Permiten visualizar simultáneamente las correlaciones, tendencias, agrupamientos (clusters) y valores atípicos en los datos.
  - **Ventajas**:
    - Facilitan la identificación de patrones entre múltiples variables.
    - Son útiles para explorar la relación no lineal entre pares de variables.
  - <img src="https://seaborn.pydata.org/_images/pairplot_11_0.png" alt="Descripción de la imagen" width="700">
   ```python
          #  Codigo para hacer un pairplot
          sns.pairplot(df)
          plt.show()
  
- **Matriz de correlación**:
  - Una matriz que muestra todas las correlaciones entre las variables numéricas. Cada celda de la matriz representa la correlación entre dos variables, ya sea positiva o negativa.
  - Este tipo de visualización es muy práctico cuando se trabaja con muchos datos y se quiere entender cómo las variables se relacionan entre sí.
  - **Ventajas**:
    - Permite visualizar rápidamente cómo están relacionadas todas las variables en el conjunto.
    - Es útil para detectar dependencias multivariables y para identificar posibles redundancias o relaciones fuertes entre las variables.
  - <img src="https://r-coder.com/images/posts/correlation_plot/es/funcion-corPlot-r.PNG" alt="Descripción de la imagen" width="700">

- **Heatmaps multivariables**:
  - Estos mapas de calor permiten observar múltiples correlaciones a la vez en un conjunto de datos multivariable. Se usan colores para representar la intensidad de la relación entre las variables, y muestran fácilmente las correlaciones débiles, moderadas o fuertes.
  - Muy útil para identificar patrones y correlaciones generales entre múltiples variables.
  - <img src="https://bookdown.org/brian_nguyen0305/Multivariate_Statistical_Analysis_with_R/_main_files/figure-html/unnamed-chunk-115-1.png" alt="Descripción de la imagen" width="700">

- **Redes de correlación**:
  - Representan visualmente las relaciones entre múltiples variables como nodos y aristas, donde los nodos son las variables y las aristas muestran la intensidad de la correlación.
  - Permiten explorar las relaciones complejas entre varias variables al mismo tiempo.
  - <img src="https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEirUnrEZ_exKwJSRgFX2YCtqoa8o3a1VA-PV0ad_bgGFiYKzrsLK8k0XFdWWxU1qpRC7QdJ9qIu6mX3qhqClBRO_uS3Nyi2vlIb0HlapjRUkcrrzP_RWvpSV5ccjIGQxIbzQPRI-le4An4/s1600/Imagen1.jpg" alt="Descripción de la imagen" width="700">


---

### Tema 3.4: EDA con Datasets Complejos

A veces, nuestros datos son más complicados o muy grandes. Puede que tengamos cientos de columnas o millones de filas. ¿Cómo empezamos a analizar algo tan enorme?

### 1. Preparación de datasets grandes

- **Muestreo**:  
  El muestreo es un proceso fundamental cuando trabajamos con datasets grandes. Permite tomar una parte más pequeña pero representativa del conjunto completo, lo que es especialmente útil para realizar pruebas rápidas, validaciones o análisis preliminares sin sobrecargar los recursos computacionales.  
  - **Muestreo aleatorio**: Se seleccionan aleatoriamente las muestras del dataset, asegurando una representación uniforme del conjunto.
  - **Muestreo estratificado**: Se divide el dataset en diferentes grupos (estratos) según una característica y luego se seleccionan muestras representativas de cada grupo.
  - **Ventajas del muestreo**:
    - Ahorra tiempo y recursos computacionales al trabajar con una muestra en lugar del dataset completo.
    - Mantiene las características estadísticas del dataset original al asegurar representatividad.

- **Selección de columnas relevantes**:  
  Al trabajar con datasets grandes, muchas veces es necesario identificar qué columnas son las más relevantes para el análisis. Esto evita sobrecargar los modelos con datos innecesarios y permite centrar la atención en las variables que realmente aportan valor al estudio.  
  - **Métodos para seleccionar columnas relevantes**:
    - **Análisis de correlación**: Permite identificar las columnas que tienen una relación significativa con la variable objetivo.
    - **Métodos de reducción de dimensionalidad**: Técnicas como PCA (Análisis de Componentes Principales) ayudan a reducir la cantidad de variables al seleccionar las componentes principales que explican la mayor varianza.
    - **Selección por criterio de importancia**: Usar algoritmos como Random Forest o Extra Trees para determinar cuál columna tiene más impacto en el modelo.
  - **Ventajas de seleccionar columnas relevantes**:
    - Reduce el tiempo de procesamiento y el costo computacional.
    - Mejora la precisión del modelo al enfocarse en las variables que realmente importan.
    - Facilita la interpretación de los resultados del análisis.

### 2. Herramientas para manejo de datos complejos

- **groupby() en pandas**:  
  Esta función es fundamental para trabajar con datos agrupados. Permite agrupar los datos por una o varias categorías y realizar operaciones estadísticas sobre esos grupos.  
  - **Uso común**:  
    - **Agrupar por una categoría**: Para ver los datos resumidos por un criterio específico, como agrupar por región, mes, año, o cualquier otra característica.
    - **Operaciones estadísticas**: Se pueden realizar diversas operaciones sobre los grupos, como la suma, la media, la desviación estándar, la cuenta de registros, entre otros.  
  - **Ejemplo**:
    ```python
    import pandas as pd
    
    # Crear un ejemplo de DataFrame
    df = pd.DataFrame({
        'Region': ['North', 'South', 'East', 'West', 'North', 'East'],
        'Sales': [1500, 2000, 1300, 1800, 1600, 1700]
    })
    
    # Agrupar por región y calcular la suma de ventas
    grouped = df.groupby('Region')['Sales'].sum()
    print(grouped)
    ```
  - **Ventajas de groupby()**:
    - Facilita el análisis por categorías, como el análisis regional, temporal o cualquier clasificación de datos.
    - Permite manipular y sumarizar grandes volúmenes de datos de manera eficiente.

- **pivot_table()**:  
  Esta función en pandas es útil para transformar datos complejos en una tabla más estructurada, donde podemos realizar comparaciones más fácilmente entre distintas categorías o dimensiones.  
  - **Uso común**:  
    - **Transformar datos**: Convierte datos en una tabla pivotada, permitiendo la comparación de valores según múltiples niveles, como tabla cruzada de ingresos por producto y región.
    - **Agregación y resumen**: A través de pivot_table(), es posible realizar agregaciones sobre las celdas pivotadas, como sumas, medias o conteos.  
  - **Ejemplo**:
    ```python
    import pandas as pd
    
    # Crear un ejemplo de DataFrame
    df = pd.DataFrame({
        'Region': ['North', 'South', 'East', 'West', 'North', 'East'],
        'Product': ['A', 'B', 'A', 'B', 'A', 'B'],
        'Sales': [1500, 2000, 1300, 1800, 1600, 1700]
    })
    
    # Crear una tabla pivotada para mostrar las ventas por producto y región
    pivot = df.pivot_table(values='Sales', index='Region', columns='Product', aggfunc='sum')
    print(pivot)
    ```
  - **Ventajas de pivot_table()**:
    - Permite organizar los datos de manera más amigable para la comparación, resaltando las relaciones entre categorías.
    - Es ideal para trabajar con tablas cruzadas, facilitando el análisis comparativo de múltiples dimensiones.

#### 3. Ejemplo 3.1
-[**`Ejemplo 3.1`**](ejercicio3.1/Ejercicio3_1_modulo3.ipynb)

#### 4. Ejemplo 3.2
-[**`Ejemplo 3.2`**](ejercicio3.2/Ejercicio3_2_modulo3.ipynb)

#### 5. Ejemplo 3.3
-[**`Ejemplo 3.3`**](ejercicio3.3/Ejercicio3_3_modulo3.ipynb)

#### 6. Ejercicio 3.4 (Alumnos)
-[**`Ejercicio 3.4`**](ejercicio3.4/Ejercicio3_4_modulo3.ipynb)

#### Ejercicio Unificado (Alumnos)
-[**`Ejercicio Unificado`**](ejercicio_unificado/Ejercicio_Unificado_Alumno.ipynb)

