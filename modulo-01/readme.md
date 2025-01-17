# Módulo 1: Introducción y Fundamentos de Ciencia de Datos

## :dart: **Objetivo del Módulo**
Al finalizar este módulo, los participantes comprenderán qué es la Ciencia de Datos, sus aplicaciones, el flujo de trabajo típico, las herramientas principales y los conceptos básicos de estadística y probabilidad necesarios para analizar datos. Además, habrán realizado ejercicios prácticos que les permitirán afianzar los conocimientos.

---

#### **Tema 1.1: ¿Qué es la Ciencia de Datos?**

1. **Definición:**
   - La Ciencia de Datos es un campo interdisciplinario que combina programación, estadística, matemáticas y conocimiento del dominio para extraer información valiosa de los datos.
   -  <img src="https://sp-ao.shortpixel.ai/client/to_auto,q_glossy,ret_img,w_512,h_288/https://datademia.es/wp-content/uploads/2022/09/Captura-de-Pantalla-2022-09-15-a-las-7.51.49-1024x575.png" alt="Descripción de la imagen" width="700">
      
   
2. **Aplicaciones prácticas:**
   - Predicción de ventas.
   - Análisis de comportamiento del cliente.
   - Modelado de riesgo en finanzas.
   - Detección de fraudes.
   - <img src="https://www.caosyciencia.com/wp-content/uploads/2024/10/ejemplos-de-dashboard-excel-para-mejorar-tu-toma-de-decisiones.jpg" alt="Descripción de la imagen" width="700">
   
4. **Rol de un Científico de Datos:**
   - Formular preguntas clave.
   - Manejar grandes volúmenes de datos.
   - Construir modelos predictivos.
   - Comunicar resultados a las partes interesadas.

**Ejemplo práctico:**
- Caso simplificado: Analizar datos de ventas mensuales para identificar tendencias y realizar predicciones.

**Ejercicio para los alumnos:**
- Identificar tres problemas en su área laboral o personal donde podría aplicarse Ciencia de Datos.

---

#### **Tema 1.2: Flujo de Trabajo en Ciencia de Datos**

1. **Etapas principales:**
   - Obtención de datos: Recolectar datos de diversas fuentes (APIs, bases de datos, archivos CSV, etc.).
   - Limpieza de datos: Eliminar inconsistencias, valores nulos y atípicos.
     ```python
     #Usar Pandas para eliminar valores nulos: 
      df = df.dropna()
     
     #Detectar y tratar valores atípicos: 
      Q1 = df['Ventas'].quantile(0.25)
      Q3 = df['Ventas'].quantile(0.75)
      IQR = Q3 - Q1
      outliers = df[(df['Ventas'] < (Q1 - 1.5 * IQR)) | (df['Ventas'] > (Q3 + 1.5 * IQR))]
   - Análisis exploratorio: Comprender patrones y relaciones en los datos mediante estadísticas y visualizaciones.
     ```python
     #Generar gráficos para entender patrones: 
      df['Ventas'].hist()
      plt.title('Distribución de Ventas')
      plt.show()
   - Modelado: Crear modelos predictivos usando Machine Learning.
     ```python
     #Introducción a un modelo simple de regresión lineal: 
      from sklearn.linear_model import LinearRegression
      model = LinearRegression()
      X = df[['Mes']]
      y = df['Ventas']
      model.fit(X, y)
      print(f"Coeficiente: {model.coef_}, Intercepto: {model.intercept_}")
   - Visualización y comunicación: Presentar los hallazgos en forma de dashboards o reportes claros.
   
2. **Diagrama del flujo de trabajo:**
   - <img src="https://scontent.fvsa1-2.fna.fbcdn.net/v/t1.6435-9/120273370_652092392159862_5896927628537707756_n.jpg?_nc_cat=100&ccb=1-7&_nc_sid=833d8c&_nc_ohc=UXMv4tXQKdMQ7kNvgHCXBgR&_nc_zt=23&_nc_ht=scontent.fvsa1-2.fna&_nc_gid=ATQVYFnQ6UpatqyFxN7DUuY&oh=00_AYCeJmPjHW4-Q2DmWIZcO5srSn3oXlCKaOhtpjl9kiWmfQ&oe=67B22EF1" alt="Descripción de la imagen" width="700">
  
4. **Herramientas utilizadas en cada etapa:**
   - Pandas para manipulación de datos
   - Matplotlib para visualización
   - Scikit-learn para modelado.

**Ejercicio práctico:**
- Crear un esquema en Google Colab con encabezados para cada etapa del flujo de trabajo.

---

#### **Tema 1.3: Herramientas Principales de Ciencia de Datos**

1. **Google Colab:**
   - Plataforma en la nube que permite ejecutar notebooks de Python sin configuración local.
   - Ventajas: Colaboración en tiempo real, acceso a GPUs, almacenamiento en la nube.
     <img src="https://github.com/user-attachments/assets/53f5ee0e-c69e-4c78-9bab-a0c48089eeb7" alt="Descripción de la imagen" width="700">

2. **Python y sus librerías:**
   - Pandas: Manipulación de datos estructurados.
   - NumPy: Operaciones matemáticas y matrices.
   - Matplotlib: Visualización gráfica.
   
3. **Dataset de ejemplo:**
   - Usa un dataset de ventas en formato CSV:
   - Dataset: Ventas por Mes y Región

      | Mes | Ventas | Región |
      |-----|--------|--------|
      | 1   | 1000   | Norte  |
      | 2   | 1200   | Sur    |
      | 3   | 1300   | Este   |

**Ejercicio práctico:**
- Abrir Google Colab y cargar un dataset en formato CSV. Explorar el dataset con las funciones `head()` y `info()`.
