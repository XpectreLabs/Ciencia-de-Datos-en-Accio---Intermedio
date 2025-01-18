## **Módulo 2: Fundamentos de Estadística y Probabilidad Aplicada**

### 🎯**Objetivo del Módulo**
Este módulo tiene como objetivo brindar a los participantes una comprensión sólida de los conceptos básicos de estadística y probabilidad, así como sus aplicaciones prácticas. Los estudiantes aprenderán a calcular y analizar medidas de tendencia central y dispersión en conjuntos de datos, comprenderán los fundamentos de la probabilidad, las distribuciones comunes, y cómo representarlas gráficamente. Al finalizar, los participantes estarán capacitados para aplicar estos conceptos en la interpretación de datos, la toma de decisiones basadas en la probabilidad, y la visualización efectiva de información.

---
#### **Tema 2.1: Conceptos Básicos de Estadística**

1. **Medidas de tendencia central:**
   - Media: Promedio de los valores.
     
     ```python
       print("Media:", df['Ventas'].mean())
   - Mediana: Valor central de un conjunto ordenado.
     
     ```python
       print("Mediana:", df['Ventas'].median())
   - Moda: Valor que más se repite.
     
     ```python
       print("Moda:", df['Ventas'].mode()[0])
   - <img src="https://github.com/user-attachments/assets/57efa55b-ab80-40a3-830d-b97a9e78f17d" width="700">
   
2. **Medidas de dispersión:**
   - Rango: La diferencia entre el valor máximo y el valor mínimo en un conjunto de datos.

     ```python
     print("Rango:", df['Ventas'].max() - df['Ventas'].min())
   - Varianza: Una medida de la dispersión que calcula la media de los cuadrados de las diferencias entre los valores y la media del conjunto.

     ```python
       print("Varianza:", df['Ventas'].var())
   - Desviación estándar: La raíz cuadrada de la varianza, que representa la dispersión promedio de los datos en relación a su media.

     ```python
       print("Desviación estándar:", df['Ventas'].std())
   - <img src="https://bookdown.org/dietrichson/metodos-cuantitativos/metodos-cuantitativos_files/figure-html/box-plot-with-explanation-1.png" width="700">

📚**Ejercicio práctico:**
- Crear una lista de números y calcular la media, mediana y desviación estándar usando Numpy.

---

#### **Tema 2.2: Introducción a Probabilidad**

1. **Definición de probabilidad:**
   - La probabilidad es una medida que indica la frecuencia con la que se espera que ocurra un evento en un número infinito de ensayos.
   - Se define como el cociente entre el número de casos favorables y el número total de casos posibles.
   - Ejemplo: Lanzamiento de un dado:
   - El dado tiene 6 caras numeradas (casos posibles).
   - La probabilidad de que salga un número par en un solo lanzamiento es el número de casos favorables (3 caras: 2, 4, 6) dividido entre el total de casos posibles (6 caras).
   
2. **Distribuciones comunes:**
   - Uniforme: Cada resultado tiene la misma probabilidad.
   - <img src="https://www.lifeder.com/wp-content/uploads/2020/11/distribucion-uniforme-continua.jpg" width="700">
   - Normal: Distribución en forma de campana.
   - <img src="https://alianza.bunam.unam.mx/wp-content/uploads/2024/02/Figura-2.-Tipificacion-de-una-curva-normal-por-su-media-y-desviacion-estandar.png" width="700">
   
📚**Ejercicio práctico:**
- Generar datos con una distribución uniforme y graficar un histograma.

---

#### **Tema 2.3: Representación Gráfica de Datos**

1. **Tipos de gráficos:**
   - Histogramas: Para distribuciones de datos.
   - <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/Histogram_example.svg/1200px-Histogram_example.svg.png" width="700">
   - Diagramas de barras: Para datos categóricos.
   - <img src="https://www.jmp.com/es_mx/statistics-knowledge-portal/exploratory-data-analysis/bar-chart/_jcr_content/par/styledcontainer_2069/par/image_1203777138.img.png/1594745267192.png" width="700">
   - Gráficos de dispersión: Para relaciones entre dos variables.
   - <img src="https://static.wixstatic.com/media/d7b433_8b364cba373247b78b4ea3579026a60e~mv2.png/v1/fill/w_528,h_352,al_c,q_85,usm_0.66_1.00_0.01,enc_auto/d7b433_8b364cba373247b78b4ea3579026a60e~mv2.png" width="700">
   
2. **Ejemplo práctico:**
   - Gráfico de barras con ventas por región.
   - Gráfico de dispersión con horas de estudio vs calificaciones.

**Ejercicio para los alumnos:**
- Crear un gráfico de barras con sus propios datos usando Matplotlib.
- Crear un gráfico de dispersión que relacione dos variables numéricas.
