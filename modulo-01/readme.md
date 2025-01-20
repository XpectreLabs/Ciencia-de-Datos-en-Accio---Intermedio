# Módulo 1: Introducción y Fundamentos de Ciencia de Datos

## :dart: **Objetivo del Módulo**
Al finalizar este módulo, los participantes comprenderán qué es la Ciencia de Datos, sus aplicaciones, el flujo de trabajo típico, las herramientas principales y los conceptos básicos de estadística y probabilidad necesarios para analizar datos. Además, habrán realizado ejercicios prácticos que les permitirán afianzar los conocimientos.

---

## **Tema 1: Introducción y Fundamentos de Ciencia de Datos**

#### **Tema 1.1: ¿Qué es la Ciencia de Datos?**

1. **Definición:**
   - La ciencia de datos es el estudio de datos con el fin de extraer información significativa para empresas. Es un enfoque multidisciplinario que combina principios y prácticas del campo de las matemáticas, la estadística, la inteligencia artificial y la ingeniería de computación para analizar grandes cantidades de datos. Este análisis permite que los científicos de datos planteen y respondan a preguntas como “qué pasó”, “por qué pasó”, “qué pasará” y “qué se puede hacer con los resultados”.
     
   -  <img src="https://sp-ao.shortpixel.ai/client/to_auto,q_glossy,ret_img,w_512,h_288/https://datademia.es/wp-content/uploads/2022/09/Captura-de-Pantalla-2022-09-15-a-las-7.51.49-1024x575.png" alt="Descripción de la imagen" width="700">
      
   
2. **Aplicaciones prácticas:**
   - **Predicción de ventas**
      - **Descripción:** Permite estimar las ventas futuras basándose en datos históricos y patrones observados. Esto ayuda a las empresas a planificar estrategia
        de inventario, producción y marketing.
      - **Ejemplo:** Una cadena de supermercados utiliza datos de ventas pasadas, estacionalidad y factores externos como el clima para prever la demanda
        de ciertos productos durante las festividades.
      - **Herramientas utilizadas:** Modelos de series temporales como ARIMA, Prophet, o técnicas de Machine Learning como regresión lineal y árboles de decisión.
   - **Análisis de comportamiento del cliente**
      - **Descripción:** Identifica patrones en el comportamiento del cliente, como preferencias de compra, hábitos de consumo y propensión a abandonar un
        servicio (churn). Esto permite personalizar ofertas, mejorar la experiencia del cliente y aumentar la retención.
      - **Ejemplo práctico:** Una plataforma de streaming analiza las preferencias de visualización de sus usuarios para recomendar contenido relevante y diseñar         campañas de marketing dirigidas.
      - **Herramientas utilizadas:** Clustering (por ejemplo, K-means), análisis de cohortes y modelos de recomendación.
   - **Modelado de riesgo en finanzas**
      - **Descripción:** Evalúa la probabilidad de eventos negativos, como incumplimiento de pagos o fluctuaciones en el mercado, para mitigar riesgos y tomar
        decisiones informadas.
      - **Ejemplo práctico:** Un banco utiliza modelos predictivos para evaluar la probabilidad de que un cliente incumpla el pago de un préstamo, basándose en
        datos como ingresos, historial crediticio y nivel de endeudamiento.
      - **Herramientas utilizadas:** Modelos de clasificación como Random Forest, Logistic Regression, y técnicas de análisis de series temporales.
   - **Detección de fraudes**
      - **Descripción:** Identifica actividades sospechosas o fraudulentas al analizar transacciones y comportamientos en tiempo real. Esto es fundamental para
        prevenir pérdidas financieras y proteger la seguridad de los usuarios.
      - **Ejemplo práctico:** Un sistema de pagos electrónicos monitorea transacciones inusuales, como compras de alto valor desde ubicaciones geográficas
        inusuales, y las marca para su revisión.
      - **Herramientas utilizadas:** Algoritmos de detección de anomalías, redes neuronales, y técnicas de aprendizaje supervisado o no supervisado.
        
   - <img src="https://www.caosyciencia.com/wp-content/uploads/2024/10/ejemplos-de-dashboard-excel-para-mejorar-tu-toma-de-decisiones.jpg" alt="Descripción de la imagen" width="700">
   
4. **Rol de un Científico de Datos:** El científico de datos es un profesional interdisciplinario que combina habilidades técnicas, analíticas y de comunicación para transformar datos en información valiosa para la toma de decisiones. Algunas responsabilidades:
   
      - **Formular preguntas clave:**
         - El punto de partida es entender qué problemas necesitan resolverse. Esto implica traducir problemas del mundo real en preguntas que puedan responderse con datos.
         - Ejemplo:
            - "¿Qué factores aumentan la satisfacción del cliente?"
            - "¿Cómo podemos optimizar nuestro inventario según patrones de compra?"
         - Un buen científico de datos sabe que las preguntas correctas guían todo el análisis.
           
      - **Manejar grandes volúmenes de datos:**
         - Los datos provienen de múltiples fuentes, como bases de datos, redes sociales o sensores IoT. Este paso incluye:
            - Recolectar datos de manera estructurada (tablas, bases de datos) y no estructurada (imágenes, texto).
            - Limpiar y preparar los datos eliminando errores, valores atípicos o datos incompletos.
            - Integrar diferentes fuentes de datos para obtener una visión completa.
         - Ejemplo: analizar ventas, datos de marketing y reseñas de clientes juntos para detectar patrones.
           
      - **Construir modelos predictivos:**
         - Utilizando Machine Learning y estadística avanzada, un científico de datos desarrolla modelos que pueden:
            - Predecir comportamientos (como compras futuras o abandono de clientes).
            - Detectar anomalías, como posibles fraudes o errores.
            - Optimizar procesos, como la asignación de recursos o la gestión de inventarios.
         - Estos modelos permiten a las empresas tomar decisiones basadas en evidencia.
           
      - **Comunicar resultados de manera efectiva:**
         - La habilidad de presentar información técnica a un público no técnico es crucial. Esto se logra mediante:
            - Dashboards interactivos que muestran métricas clave en tiempo real.
            - Visualizaciones claras como gráficos y diagramas.
            - Reportes ejecutivos con conclusiones prácticas y recomendaciones.
         - Ejemplo: un dashboard para mostrar las métricas de rendimiento de una campaña de marketing en tiempo real.
           
      - **Otros roles clave:**
         - Experimentar con nuevas tecnologías: Adoptar herramientas como Python, R, TensorFlow o AWS.
         - Colaborar con diferentes áreas: Trabajar con marketing, finanzas, operaciones y tecnología para resolver problemas desde diferentes perspectivas.
         - Mantener la curiosidad: Buscar oportunidades en los datos y estar dispuesto a aprender constantemente sobre nuevos enfoques y tendencias.


[**`Ejemplo práctico 1`**](ejemplo_practico1/Ejemplo_practico1.ipynb)
- Analizar datos de ventas mensuales para identificar tendencias y realizar predicciones.

**Ejercicio para los alumnos:**
- Identificar tres problemas en su área laboral o personal donde podría aplicarse Ciencia de Datos.

---

#### **Tema 1.2: Flujo de Trabajo en Ciencia de Datos**

1. **Etapas principales:**
   - **Obtención de datos**
      - Este es el primer paso en el flujo de trabajo, donde se recolectan los datos necesarios para resolver un
        problema o responder una pregunta. Los datos pueden provenir de diversas fuentes:
         - **APIs:** Interfaces que permiten extraer datos en tiempo real desde aplicaciones o servicios (por ejemplo,
           datos meteorológicos o redes sociales).
         - **Bases de datos:** Sistemas organizados para almacenar y gestionar grandes cantidades de datos
           estructurados.
         - **Archivos locales:** Archivos como CSV, Excel o JSON que contienen datos sin conexión.
         - **Web scraping:** Extracción automatizada de datos de sitios web.
      ```python
      import pandas as pd
      df = pd.read_csv('ventas_mensuales.csv')
      print(df.head())

   - **Limpieza de datos:**
      - Este paso es crucial para asegurar que los datos sean utilizables. Se eliminan errores, valores
        inconsistentes, duplicados y datos faltantes.
         - **Inconsistencias:** Resolver problemas como diferentes formatos de fecha o errores tipográficos.
         - **Valores nulos:** Manejar celdas vacías mediante imputación (reemplazar valores nulos con la media,
           mediana, etc.) o eliminación.
         - **Outliers (valores atípicos):** Identificar y decidir si estos valores extremos deben eliminarse o
           tratarse.
     ```python
     #Usar Pandas para eliminar valores nulos: 
      df = df.dropna()
     
     #Detectar y tratar valores atípicos: 
      Q1 = df['Ventas'].quantile(0.25)
      Q3 = df['Ventas'].quantile(0.75)
      IQR = Q3 - Q1
      outliers = df[(df['Ventas'] < (Q1 - 1.5 * IQR)) | (df['Ventas'] > (Q3 + 1.5 * IQR))]
   - **Análisis exploratorio**
      - En esta etapa, los datos se analizan para identificar patrones, tendencias y relaciones clave. Este
        proceso ayuda a formular hipótesis y guiar la construcción de modelos.
         - Estadísticas descriptivas: Resúmenes de los datos, como la media, mediana, moda, rango y desviación
           estándar.
         - Visualizaciones: Usar gráficos como histogramas, diagramas de dispersión y boxplots para entender
           los datos.
     ```python
     #Generar gráficos para entender patrones: 
      df['Ventas'].hist()
      plt.title('Distribución de Ventas')
      plt.show()
   - **Modelado**
      - En esta etapa se desarrollan modelos predictivos o descriptivos utilizando algoritmos de Machine
        Learning. Esto incluye:
         - Dividir los datos en conjuntos de entrenamiento y prueba.
         - Seleccionar y ajustar un modelo (por ejemplo, regresión lineal, árboles de decisión o redes
           neuronales).
         - Evaluar el modelo utilizando métricas como precisión, recall o F1-score.
     ```python
     #Introducción a un modelo simple de regresión lineal: 
      from sklearn.linear_model import LinearRegression
      model = LinearRegression()
      X = df[['Mes']]
      y = df['Ventas']
      model.fit(X, y)
      print(f"Coeficiente: {model.coef_}, Intercepto: {model.intercept_}")
   - **Visualización y comunicación**
      - Una vez que se obtienen los resultados, es fundamental presentarlos de manera clara y efectiva a las
        partes interesadas. Esto se logra mediante:
         - **Dashboards interactivos:** Usando herramientas como Tableau o Power BI.
         - **Gráficos claros:** Creación de gráficos de barras, líneas o mapas de calor.
         - **Reportes:** Documentos estructurados con conclusiones y recomendaciones basadas en el análisis.
      ```python
      #Crear un gráfico de barras que muestre las ventas por región.
      import seaborn as sns
      sns.barplot(x="Región", y="Ventas", data=df)
      plt.title("Ventas por Región")
      plt.show()

   
2. **Diagrama del flujo de trabajo:**
   - <img src="https://scontent.fvsa1-2.fna.fbcdn.net/v/t1.6435-9/120273370_652092392159862_5896927628537707756_n.jpg?_nc_cat=100&ccb=1-7&_nc_sid=833d8c&_nc_ohc=UXMv4tXQKdMQ7kNvgHCXBgR&_nc_zt=23&_nc_ht=scontent.fvsa1-2.fna&_nc_gid=ATQVYFnQ6UpatqyFxN7DUuY&oh=00_AYCeJmPjHW4-Q2DmWIZcO5srSn3oXlCKaOhtpjl9kiWmfQ&oe=67B22EF1" alt="Descripción de la imagen" width="700">
  
4. **Herramientas utilizadas en cada etapa:**
   - **Pandas para manipulación de datos**
      - Pandas es una de las librerías más utilizadas en Python para manipular y analizar datos. Permite
        realizar operaciones sobre estructuras como Series y DataFrames, ideales para la limpieza,
        transformación y exploración de datos.
      - Funciones clave:
         - read_csv(): Leer archivos CSV.
         - dropna(): Eliminar valores nulos.
         - groupby(): Agrupar datos según una o más columnas.
         - describe(): Obtener estadísticas descriptivas del conjunto de datos.
           ```python
           import pandas as pd

            # Cargar datos desde un archivo CSV
            df = pd.read_csv("ventas_mensuales.csv")
            
            # Ver las primeras filas del DataFrame
            print(df.head())
            
            # Eliminar registros con valores nulos
            df_clean = df.dropna()
            
            # Agrupar por región y calcular la suma de ventas
            ventas_por_region = df_clean.groupby('Región')['Ventas'].sum()
            print(ventas_por_region)

   - **Matplotlib para visualización**
      - Matplotlib es una librería de visualización en Python utilizada para crear gráficos estáticos,
        dinámicos e interactivos. Es ampliamente empleada para generar visualizaciones como gráficos de
        barras, líneas, dispersión, histogramas y más.
      - Funciones clave:
         - plt.plot(): Crear gráficos de líneas.
         - plt.bar(): Crear gráficos de barras.
         - plt.scatter(): Crear gráficos de dispersión.
         - plt.title(), plt.xlabel(), plt.ylabel(): Etiquetar los gráficos.
           ```python
           import matplotlib.pyplot as plt

            # Crear un gráfico de barras para las ventas por región
            plt.bar(['Norte', 'Sur', 'Este', 'Oeste'], [1000, 1200, 1300, 1100])
            plt.title("Ventas por Región")
            plt.xlabel("Región")
            plt.ylabel("Ventas")
            plt.show()

   - **Scikit-learn para modelado**
      - Scikit-learn es una de las librerías más conocidas en Python para aprender modelos de Machine
        Learning. Proporciona herramientas para realizar tareas como regresión, clasificación, agrupamiento,
        reducción de dimensionalidad y selección de modelos.
      - Funciones clave:
         - train_test_split(): Dividir los datos en conjuntos de entrenamiento y prueba.
         - LinearRegression(): Crear modelos de regresión lineal.
         - fit(): Ajustar el modelo a los datos de entrenamiento.
         - score(): Evaluar el desempeño del modelo en los datos de prueba.
           ```python
            from sklearn.model_selection import train_test_split
            from sklearn.linear_model import LinearRegression
            
            # Crear un conjunto de datos para modelado
            X = [[100], [150], [200], [250], [300]]
            y = [150, 200, 250, 300, 350]
            
            # Dividir los datos en entrenamiento y prueba
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Crear y ajustar el modelo
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Evaluar el modelo
            print("Coeficiente:", model.coef_)
            print("Intercepto:", model.intercept_)
            print("Precisión del modelo:", model.score(X_test, y_test))


**Ejercicio práctico:**
- Crear un esquema en Google Colab con encabezados para cada etapa del flujo de trabajo.

---

#### **Tema 1.3: Herramientas Principales de Ciencia de Datos**

1. **Google Colab:**
   - Google Colab es una plataforma basada en la nube que permite ejecutar notebooks de Python directamente
     desde el navegador, eliminando la necesidad de configuraciones locales.
   - Características principales:
      - Ejecución en la nube: Realiza cálculos complejos utilizando recursos en línea, sin afectar el
      - rendimiento local.
      - Acceso a hardware avanzado: GPUs y TPUs disponibles gratuitamente para tareas como Machine Learning y
      - Deep Learning.
      - Colaboración en tiempo real: Similar a Google Docs, permite que varios usuarios editen y ejecuten
        código en simultáneo.
      - Almacenamiento en la nube: Compatible con Google Drive para guardar, acceder y compartir notebooks.
      - Entorno preconfigurado: Incluye bibliotecas populares como NumPy, Pandas, Matplotlib, TensorFlow, y
      - Scikit-learn.
   - Ventajas:
      - Gratuito y accesible desde cualquier lugar con conexión a internet.
      - Configuración cero: ideal para estudiantes y principiantes en ciencia de datos y Machine Learning.
      - Integración con herramientas externas: admite datos de APIs, almacenamiento en la nube, y bases de
        datos.
   - Limitaciones:
      - Requiere conexión a internet para funcionar.
      - Sesiones limitadas en la versión gratuita (hasta 12 horas).
      - Recursos computacionales compartidos, lo que puede reducir el rendimiento en tareas muy intensivas.
     <img src="https://github.com/user-attachments/assets/53f5ee0e-c69e-4c78-9bab-a0c48089eeb7" alt="Descripción de la imagen" width="700">
     


     -**Paso a paso**: 
      1.	Abre Google Colab en https://colab.research.google.com.
      2.	Crea un nuevo notebook y dale un título.
      3.	Escribe y ejecuta el siguiente código:
         
   ```python
        	print("Hola, Ciencia de Datos!")


2. **Python y sus librerías:**
   - **Pandas**
      - Diseñada para trabajar con datos tabulares, como hojas de cálculo o bases de datos.
      - Ofrece estructuras como DataFrames y Series, que permiten manipular filas y columnas fácilmente.
      - **Funcionalidades clave:**
         - Limpieza de datos: Manejo de valores nulos y duplicados.
         - Filtrado y selección: Extracción de datos específicos con condiciones.
         - Transformación: Realización de operaciones matemáticas o de texto en columnas.
         - Agrupación: Resumen de datos por categorías con métodos como groupby.
            ```python
            import pandas as pd

            data = {'Mes': [1, 2, 3], 'Ventas': [1000, 1200, 1300]}
            df = pd.DataFrame(data)
            print(df.describe())  # Estadísticas básicas

   - NumPy
      - Proporciona soporte para arreglos multidimensionales y operaciones matemáticas rápidas.
      - Ideal para cálculos numéricos de gran escala y operaciones vectorizadas.
      - **Funcionalidades clave:**
         - Creación de arrays y matrices.
         - Operaciones matemáticas como sumas, productos, y estadísticas.
         - Integración con otras bibliotecas como Pandas y Scikit-learn.
           ```python
           import numpy as np

            array = np.array([1, 2, 3, 4])
            print("Media:", np.mean(array))  # Media del array

   - Matplotlib
      - Herramienta para crear gráficos estáticos, animados e interactivos.
      - Compatible con otros frameworks como Pandas para graficar datos fácilmente.
      - **Tipos de gráficos comunes:**
         - Líneas: Ideal para observar tendencias.
         - Barras: Comparación entre categorías.
         - Histogramas: Distribución de datos.
         - Dispersión: Relaciones entre dos variables.
            ```python
           import matplotlib.pyplot as plt

            x = [1, 2, 3, 4]
            y = [10, 20, 30, 40]
            
            plt.plot(x, y, marker='o')
            plt.title("Gráfico de ejemplo")
            plt.xlabel("Mes")
            plt.ylabel("Ventas")
            plt.show()

   
3. **Dataset de ejemplo:**
   - Usa un dataset de ventas en formato CSV:
   - Dataset: Ventas por Mes y Región

      | Mes | Ventas | Región |
      |-----|--------|--------|
      | 1   | 1000   | Norte  |
      | 2   | 1200   | Sur    |
      | 3   | 1300   | Este   |

4. [**`Ejercicio práctico 1`**](ejercicio_practico1/Ejercicio_practico1.ipynb)

## **Tema 2: Fundamentos de Estadística y Probabilidad Aplicada**

---
#### **Tema 2.1: Conceptos Básicos de Estadística**

1. **Medidas de tendencia central:**
   - Media
      - La media es el valor promedio de un conjunto de datos. Se calcula sumando todos los valores
        dividiendo entre el número total de observaciones.
      - Uso: Es ideal para datos distribuidos de manera uniforme.
      - <img src="https://yosoytuprofe.20minutos.es/wp-content/uploads/2019/04/media-aritm%C3%A9tica-1.jpg" width="700">
     
     ```python
       print("Media:", df['Ventas'].mean())
     
   - Mediana:
      - La mediana es el valor central de un conjunto de datos ordenado.
      - Uso: Es más robusta frente a valores atípicos o extremos.
      - Cómo calcularla:
         - Si el número de observaciones es impar, la mediana es el valor central.
         - Si es par, es el promedio de los dos valores centrales.
      - <img src=" https://www.ionos.mx/digitalguide/fileadmin/DigitalGuide/Screenshots_2021/median-berechnen-1.png" width="700">
     
     ```python
       print("Mediana:", df['Ventas'].median())
     
   - Moda:
      - La moda es el valor que más se repite en un conjunto de datos.
      - Uso: Útil para datos categóricos o cuando se desea identificar la frecuencia más alta.
      - Ejemplo:
         - Si las ventas diarias son: [1200, 1500, 1200, 1700], la moda es 1200 porque aparece con mayor
           frecuencia.
      - <img src="https://mundoestudiante.com/wp-content/uploads/2021/12/la-moda-en-matematicas.png" width="700">
        
     ```python
       print("Moda:", df['Ventas'].mode()[0])
     
   - <img src="https://github.com/user-attachments/assets/57efa55b-ab80-40a3-830d-b97a9e78f17d" width="700">
   
2. **Medidas de dispersión:**
   - Las medidas de dispersión permiten entender qué tan dispersos o concentrados están los datos respecto a su tendencia central. Son fundamentales para
     evaluar la variabilidad en un conjunto de datos.
   
      - **Rango:**
         - Es la diferencia entre el valor máximo y el valor mínimo en un conjunto de datos.
         - Uso: Da una idea rápida de la extensión de los datos.
         - Ejemplo:
            - Si las ventas mensuales de un negocio son [200, 300, 450, 500], el rango es:
            - 500 − 200 = 300
         - Ventajas: Fácil de calcular y comprender.
         - Limitaciones: No muestra información sobre cómo están distribuidos los valores entre los extremos.
   
        ```python
        #Calcular rango usando pandas 
        import pandas as pd
        print("Rango:", df['Ventas'].max() - df['Ventas'].min())

        # Calcular rango usando NumPy
         import numpy as np
         print("Rango:", np.max(df['Ventas']) - np.min(df['Ventas']))

        
      - **Varianza:**
         - Definición: Mide cómo se dispersan los datos en relación a la media, calculando el promedio de las diferencias al cuadrado entre cada dato y la
           media.
         - Interpretación: Una varianza alta indica datos muy dispersos, mientras que una baja sugiere datos más cercanos a la media.
   
        ```python
        #Calcular varianza usando pandas 
        import pandas as pd
        print("Varianza:", df['Ventas'].var())

        # Calcular varianza usando NumPy
         import numpy as np
         print("Varianza:", np.var(df['Ventas'], ddof=0))

        
      - **Desviación estándar:**
         - Definición: Es la raíz cuadrada de la varianza, lo que facilita su interpretación al estar en las mismas unidades que los datos originales.
         - Uso: Permite medir cuánto se desvían los datos en promedio respecto a la media.
   
        ```python
         #Calcular desviación estándar usando pandas 
         import pandas as pd
         print("Desviación estándar:", df['Ventas'].std())

        # Calcular desviación estándar usando NumPy
         import numpy as np
         print("Desviación estándar:", np.std(df['Ventas'], ddof=0)) 
        
      - <img src="https://bookdown.org/dietrichson/metodos-cuantitativos/metodos-cuantitativos_files/figure-html/box-plot-with-explanation-1.png" width="700">

[**`Ejercicio práctico 2`**](ejercicio_practico2/Ejercicio_practico2.ipynb)
- Crear una lista de números y calcular la media, mediana y desviación estándar usando Numpy.

---

#### **Tema 2.2: Introducción a Probabilidad**

1. **Definición de Probabilidad**
   La probabilidad es una medida que cuantifica la frecuencia con la que se espera que ocurra un evento en un número infinito de ensayos.  
   - **Fórmula general:**  
     Probabilidad = (Casos Favorables) / (Total de Casos Posibles)
   
   - **Ejemplo práctico:**  
     **Lanzamiento de un dado:**  
     - Un dado tiene **6 caras numeradas** del 1 al 6 (total de casos posibles).  
     - La probabilidad de que salga un número **par** en un solo lanzamiento:  
       - **Casos favorables:** 3 (números pares: 2, 4, 6).  
       - **Total de casos posibles:** 6.  
       - **Resultado:**  
         Probabilidad = 3 / 6 = 0.5 (o 50%)

2. **Distribuciones Comunes**

   - **Distribución Uniforme**
      - En esta distribución, **cada resultado tiene la misma probabilidad** de ocurrir.  
      - **Ejemplo:** Al lanzar un dado equilibrado, la probabilidad de que salga cualquier número del 1 al 6 es igual: 1/6.  
      - **Representación gráfica:**  
        <img src="https://www.lifeder.com/wp-content/uploads/2020/11/distribucion-uniforme-continua.jpg" width="700">  

   -**Distribución Normal**
      - También conocida como la **curva de campana**.  
      - Es simétrica y se centra en la media del conjunto de datos.  
      - Es utilizada para modelar fenómenos como alturas, pesos, y puntuaciones en exámenes estandarizados.  
      - **Características principales:**  
        - **Media:** Punto central de la distribución.  
        - **Desviación estándar:** Representa la dispersión de los datos.  
        - El **68%** de los datos están dentro de una desviación estándar de la media, el **95%** dentro de dos, y el **99.7%** dentro de tres.  
      - **Representación gráfica:**  
        <img src="https://alianza.bunam.unam.mx/wp-content/uploads/2024/02/Figura-2.-Tipificacion-de-una-curva-normal-por-su-media-y-desviacion-estandar.png" width="700">  


[**`Ejercicio práctico 3`**](ejercicio_practico3/Ejercicio_practico3.ipynb)
- Graficar un histograma.

---

#### **Tema 2.3: Representación Gráfica de Datos**

**1. Tipos de Gráficos**

   - **Histogramas:**  
     Los histogramas son gráficos utilizados para mostrar la distribución de un conjunto de datos continuos.  
     - Se agrupan los datos en intervalos o "bins", y la altura de cada barra representa la frecuencia de datos dentro de ese intervalo.  
     - Ideal para identificar patrones como la simetría, sesgo, o la presencia de valores atípicos en los datos.  
     - **Ejemplo de uso:** Analizar la distribución de edades en una población.  
     - <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/Histogram_example.svg/1200px-Histogram_example.svg.png" width="700">
   
   - **Diagramas de barras:**  
     Estos gráficos se utilizan para comparar diferentes categorías de datos.  
     - Cada barra representa una categoría y su altura refleja el valor asociado a ella.  
     - Es útil para datos categóricos o discretos como encuestas, resultados de elecciones o ventas por región.  
     - **Diferencia clave con histogramas:** En los diagramas de barras, las barras suelen estar separadas porque representan categorías distintas y no intervalos continuos.  
     - **Ejemplo de uso:** Mostrar ventas por producto en un mes.  
     - <img src="https://www.jmp.com/es_mx/statistics-knowledge-portal/exploratory-data-analysis/bar-chart/_jcr_content/par/styledcontainer_2069/par/image_1203777138.img.png/1594745267192.png" width="700">
   
   - **Gráficos de dispersión:**  
     Los gráficos de dispersión muestran la relación entre dos variables numéricas.  
     - Cada punto en el gráfico representa un par de valores (x, y).  
     - Son útiles para identificar correlaciones, patrones y tendencias en los datos.  
     - **Ejemplo de uso:** Evaluar la relación entre las horas de estudio y las calificaciones obtenidas por estudiantes.  
     - **Adicional:** Se pueden añadir líneas de tendencia para modelar relaciones lineales o no lineales entre las variables.  
     - <img src="https://static.wixstatic.com/media/d7b433_8b364cba373247b78b4ea3579026a60e~mv2.png/v1/fill/w_528,h_352,al_c,q_85,usm_0.66_1.00_0.01,enc_auto/d7b433_8b364cba373247b78b4ea3579026a60e~mv2.png" width="700">

   
2. [**`Ejemplo práctico 2`**](ejemplo_practico2/Ejemplo_practico2.ipynb)
     - Gráfico de barras con ventas por región.
3. [**`Ejemplo práctico 3`**](ejemplo_practico3/Ejemplo_practico3.ipynb)
     - Gráfico de dispersión con horas de estudio vs calificaciones.

[**`Ejercicio Unificado`**](ejercicio_unificado/Ejercicio_Unificado_Modulo_1_Alumno.ipynb)


