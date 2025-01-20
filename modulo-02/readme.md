# Módulo 2: Manipulación y Limpieza de Datos

## Tema 2.1: Conceptos Básicos de Pandas

### 1. Estructuras de datos principales:
- **Series**:
  - Una estructura unidimensional similar a un array, lista o columna de una tabla. Cada elemento está etiquetado por un índice, que puede ser un número entero, una cadena de texto o cualquier otro tipo de dato. Es ideal para manejar datos homogéneos, es decir, todos los elementos deben ser del mismo tipo (por ejemplo, todos numéricos o todos cadenas de texto).
  - **Características principales**:
    - Soporta operaciones elementales como suma, multiplicación, etc.
    - Compatible con funciones de NumPy.
    - Permite el acceso a elementos individuales utilizando el índice.
  - **Creación de una Serie**:
    ```python
    import pandas as pd
    # Desde una lista
    serie = pd.Series([10, 20, 30], index=['a', 'b', 'c'])
    
    # Desde un diccionario
    serie_dict = pd.Series({'a': 10, 'b': 20, 'c': 30})
    ```
- <img src="https://www.scaler.com/topics/images/pandas-series-thumbnail.webp" width="500">

---

- **DataFrames**:
  - Una estructura bidimensional que representa una tabla etiquetada con filas y columnas. Cada columna puede contener datos de diferentes tipos (números, cadenas, fechas, etc.). Es el formato más común para trabajar en Pandas, ya que ofrece flexibilidad y herramientas poderosas para manipulación de datos.
  - **Características principales**:
    - Etiquetas tanto para filas (índice) como para columnas.
    - Admite operaciones entre columnas o filas.
    - Fácil integración con fuentes de datos como archivos CSV, JSON, bases de datos, y más.
    - Permite realizar análisis y transformaciones de datos complejas con facilidad.
  - **Creación de un DataFrame**:
    ```python
    # Desde un diccionario
    data = {'Columna1': [1, 2, 3], 'Columna2': [4, 5, 6]}
    df = pd.DataFrame(data)

    # Desde listas de listas
    df_list = pd.DataFrame([[1, 4], [2, 5], [3, 6]], columns=['Columna1', 'Columna2'])
    ```

  - <img src="https://pynative.com/wp-content/uploads/2021/02/dataframe.png" width="600">
   
---

### 2. Creación de un DataFrame:
Un **DataFrame** se puede crear a partir de diversas fuentes de datos. La flexibilidad de Pandas permite importar, combinar y manipular datos fácilmente desde múltiples formatos.

---

#### **Desde listas**:
Un DataFrame puede crearse a partir de una lista simple o una lista de listas (anidada). En el caso de las listas anidadas, cada sublista representará una fila del DataFrame.

- **Ejemplo con una lista simple**:
    ```python
    import pandas as pd
    data = [10, 20, 30]
    df = pd.DataFrame(data, columns=['Valores'])
    print(df)
    ```
    **Salida**:
    | Índice | Valores |
    |--------|---------|
    | 0      | 10      |
    | 1      | 20      |
    | 2      | 30      |

- **Ejemplo con listas anidadas**:
    ```python
    data = [[1, 'A'], [2, 'B'], [3, 'C']]
    df = pd.DataFrame(data, columns=['Número', 'Letra'])
    print(df)
    ```
    **Salida**:
    | Índice | Número | Letra |
    |--------|--------|-------|
    | 0      | 1      | A     |
    | 1      | 2      | B     |
    | 2      | 3      | C     |

---

#### **Desde diccionarios**:
Los diccionarios son una fuente común para crear DataFrames. Las claves representan los nombres de las columnas y los valores, listas de datos.

- **Ejemplo**:
    ```python
    data = {'Columna1': [1, 2, 3], 'Columna2': [4, 5, 6]}
    df = pd.DataFrame(data)
    print(df)
    ```
    **Salida**:
    | Índice | Columna1 | Columna2 |
    |--------|----------|----------|
    | 0      | 1        | 4        |
    | 1      | 2        | 5        |
    | 2      | 3        | 6        |

- **Con diferentes longitudes de listas**:
    Si las listas en el diccionario tienen longitudes desiguales, Pandas rellena las celdas faltantes con valores `NaN`:
    ```python
    data = {'A': [1, 2], 'B': [3]}
    df = pd.DataFrame(data)
    print(df)
    ```
    **Salida**:
    | Índice | A   | B   |
    |--------|-----|-----|
    | 0      | 1.0 | 3.0 |
    | 1      | 2.0 | NaN |

---

#### **Desde archivos CSV**:
Se pueden importar datos desde archivos CSV (Comma-Separated Values), que es un formato común para datos tabulares.

- **Ejemplo básico**:
    ```python
    df = pd.read_csv('archivo.csv')
    print(df.head())  # Muestra las primeras 5 filas
    ```

- **Parámetros útiles**:
    - `sep`: Define el separador (por defecto `,`).
    - `header`: Especifica la fila que contiene los nombres de las columnas.
    - `index_col`: Define la columna que se usará como índice.
    - `na_values`: Identifica valores que se considerarán como `NaN`.

    ```python
    df = pd.read_csv('archivo.csv', sep=';', index_col=0, na_values=['?', 'NA'])
    ```

---

#### **Desde otros formatos**:
Pandas también permite crear DataFrames desde otras fuentes de datos:

- **Desde JSON**:
    ```python
    data = '{"col1": [1, 2], "col2": [3, 4]}'
    df = pd.read_json(data)
    print(df)
    ```
    **Salida**:
    | Índice | col1 | col2 |
    |--------|------|------|
    | 0      | 1    | 3    |
    | 1      | 2    | 4    |

- **Desde una base de datos SQL**:
    Se puede utilizar la función `read_sql()` para importar datos directamente desde bases de datos SQL:
    ```python
    import sqlite3
    conn = sqlite3.connect('base_datos.db')
    query = 'SELECT * FROM tabla'
    df = pd.read_sql(query, conn)
    ```

- **Desde archivos Excel**:
    ```python
    df = pd.read_excel('archivo.xlsx', sheet_name='Hoja1')
    ```

---

#### **Desde datos vacíos o generados**:
Puedes crear DataFrames completamente vacíos o con datos generados por funciones:
- **Vacío**:
    ```python
    df = pd.DataFrame(columns=['Columna1', 'Columna2'])
    print(df)
    ```
    **Salida**:
    | Índice | Columna1 | Columna2 |
    |--------|----------|----------|

- **Con datos generados (usando NumPy)**:
    ```python
    import numpy as np
    df = pd.DataFrame(np.random.rand(3, 2), columns=['A', 'B'])
    print(df)
    ```
    **Salida**:
    | Índice | A       | B       |
    |--------|---------|---------|
    | 0      | 0.12    | 0.89    |
    | 1      | 0.45    | 0.56    |
    | 2      | 0.78    | 0.34    |


---

### 3. Exploración inicial de datos:

Explorar un DataFrame es fundamental para entender la estructura y el contenido del conjunto de datos antes de realizar análisis profundos. Pandas ofrece varias herramientas clave para realizar esta exploración.

---

#### **Métodos clave**:

1. **`head(n)`**:
   - Muestra las primeras `n` filas del DataFrame. Por defecto, `n` es 5.
   - Muy útil para revisar rápidamente las primeras entradas y conocer el formato de los datos.
   - **Ejemplo**:
     ```python
     df = pd.DataFrame({'Columna A': [1, 2, 3], 'Columna B': [100, 200, 300]})
     print(df.head(2))
     ```
   - **Salida**:
     | Columna A | Columna B |
     |-----------|-----------|
     | 1         | 100       |
     | 2         | 200       |

---

2. **`info()`**:
   - Proporciona un resumen completo del DataFrame, incluyendo el número de entradas no nulas por columna, el tipo de datos, la memoria utilizada y los índices.
   - Muy útil para identificar rápidamente datos nulos, el tipo de datos en cada columna y la cantidad de memoria que está ocupando.
   - **Ejemplo**:
     ```python
     print(df.info())
     ```
   - **Salida**:
     ```plaintext
     <class 'pandas.core.frame.DataFrame'>
     RangeIndex: 3 entries, 0 to 2
     Data columns (total 2 columns):
      #   Column  Non-Null Count  Dtype 
     ---  ------  --------------  ----- 
      0   Columna A  3 non-null      int64 
      1   Columna B  3 non-null      int64 
     dtypes: int64(2)
     memory usage: 179.0 bytes
     None
     ```

---

3. **`describe()`**:
   - Proporciona estadísticas descriptivas para columnas numéricas como el valor mínimo, máximo, media, desviación estándar, percentiles, entre otros.
   - Útil para entender la distribución de los datos y las características generales de las columnas numéricas.
   - **Ejemplo**:
     ```python
     df = pd.DataFrame({'Columna A': [1, 2, 3], 'Columna B': [100, 200, 300]})
     print(df.describe())
     ```
   - **Salida**:
     |       | Columna A | Columna B |
     |-------|-----------|-----------|
     | count | 3.0       | 3.0       |
     | mean  | 2.0       | 200.0     |
     | std   | 1.0       | 100.0     |
     | min   | 1.0       | 100.0     |
     | 25%   | 1.5       | 150.0     |
     | 50%   | 2.0       | 200.0     |
     | 75%   | 3.0       | 250.0     |
     | max   | 3.0       | 300.0     |

---

## Tema 2.2: Selección y Filtrado de Datos

### 1. Selección de columnas y filas:

En Pandas, la selección de columnas y filas es una operación básica que permite extraer y manipular los datos de manera eficiente. Existen diversas formas de realizar esta selección, tanto basada en índices como en etiquetas.

---

#### **Selección de columnas**:
- Para seleccionar una o varias columnas, se utiliza el operador `[]`. Los nombres de las columnas deben estar entre comillas o como un arreglo de nombres.

- **Ejemplo**:
    ```python
    import pandas as pd
    data = {'Columna A': [10, 20, 30], 'Columna B': [100, 200, 300]}
    df = pd.DataFrame(data)

    # Seleccionar una sola columna
    seleccion_simple = df['Columna A']
    print(seleccion_simple)
    ```
    **Salida**:
    ```
    0    10
    1    20
    2    30
    Name: Columna A, dtype: int64
    ```

- **Ejemplo para seleccionar múltiples columnas**:
    ```python
    seleccion_multiple = df[['Columna A', 'Columna B']]
    print(seleccion_multiple)
    ```
    **Salida**:
    | Índice | Columna A | Columna B |
    |--------|-----------|-----------|
    | 0      | 10        | 100       |
    | 1      | 20        | 200       |
    | 2      | 30        | 300       |

---

#### **Selección de filas por índice**:
- **`loc[]`**: Selección basada en etiquetas, donde se puede utilizar el nombre o el índice específico.
- **`iloc[]`**: Selección basada en posición numérica, donde se selecciona directamente por el número de la fila.

- **Ejemplo utilizando `loc[]`**:
    ```python
    df = pd.DataFrame({'Columna A': [10, 20, 30], 'Columna B': [100, 200, 300]})
    
    # Seleccionar la primera fila utilizando etiquetas
    seleccion_etiquetas = df.loc[0]
    print(seleccion_etiquetas)
    ```
    **Salida**:
    | Índice | Columna A | Columna B |
    |--------|-----------|-----------|
    | 0      | 10        | 100       |

- **Ejemplo utilizando `iloc[]`**:
    ```python
    # Seleccionar la primera fila utilizando posición
    seleccion_posicion = df.iloc[0]
    print(seleccion_posicion)
    ```
    **Salida**:
    | Columna A | Columna B |
    |-----------|-----------|
    | 10        | 100       |

---

#### **Selección de filas por condiciones**:
- Para seleccionar filas según condiciones, se puede aplicar directamente una expresión condicional dentro de los corchetes `[]`.

- **Ejemplo**:
    ```python
    # Seleccionar filas donde Columna A sea mayor que 15
    seleccion_condicional = df[df['Columna A'] > 15]
    print(seleccion_condicional)
    ```
    **Salida**:
    | Índice | Columna A | Columna B |
    |--------|-----------|-----------|
    | 1      | 20        | 200       |
    | 2      | 30        | 300       |

    ```

---

### 2. Filtrado de datos:

El filtrado de datos es una operación fundamental en Pandas para extraer subconjuntos de un DataFrame basados en condiciones específicas. Se puede utilizar para manejar grandes conjuntos de datos y enfocarse solo en la información relevante.

---

#### **Filtrado básico**:
- Consiste en aplicar una condición a una columna para devolver las filas que cumplen con esa condición.

- **Ejemplo**:
    ```python
    import pandas as pd
    data = {'Columna A': [5, 10, 15, 20], 'Columna B': [100, 200, 300, 400]}
    df = pd.DataFrame(data)

    # Filtrar filas donde los valores en 'Columna A' son mayores a 10
    df_filtrado = df[df['Columna A'] > 10]
    print(df_filtrado)
    ```
    **Salida**:
    | Índice | Columna A | Columna B |
    |--------|-----------|-----------|
    | 2      | 15        | 300       |
    | 3      | 20        | 400       |

---

#### **Filtrado con múltiples condiciones**:
- Puedes combinar varias condiciones utilizando operadores lógicos:
  - `&` (AND): Devuelve las filas que cumplen ambas condiciones.
  - `|` (OR): Devuelve las filas que cumplen al menos una de las condiciones.
  - `~` (NOT): Devuelve las filas que no cumplen una condición.

- **Ejemplo con AND**:
    ```python
    df_filtrado = df[(df['Columna A'] > 10) & (df['Columna B'] < 400)]
    print(df_filtrado)
    ```
    **Salida**:
    | Índice | Columna A | Columna B |
    |--------|-----------|-----------|
    | 2      | 15        | 300       |

- **Ejemplo con OR**:
    ```python
    df_filtrado = df[(df['Columna A'] > 15) | (df['Columna B'] == 200)]
    print(df_filtrado)
    ```
    **Salida**:
    | Índice | Columna A | Columna B |
    |--------|-----------|-----------|
    | 1      | 10        | 200       |
    | 3      | 20        | 400       |

- **Ejemplo con NOT**:
    ```python
    df_filtrado = df[~(df['Columna A'] > 15)]
    print(df_filtrado)
    ```
    **Salida**:
    | Índice | Columna A | Columna B |
    |--------|-----------|-----------|
    | 0      | 5         | 100       |
    | 1      | 10        | 200       |
    | 2      | 15        | 300       |

---

#### **Filtrar usando métodos**:
Además de condiciones, Pandas ofrece métodos específicos para realizar filtrados comunes:

- **Filtrar por valores específicos**:
    ```python
    df_filtrado = df[df['Columna A'].isin([10, 20])]
    print(df_filtrado)
    ```
    **Salida**:
    | Índice | Columna A | Columna B |
    |--------|-----------|-----------|
    | 1      | 10        | 200       |
    | 3      | 20        | 400       |

- **Filtrar por cadenas de texto**:
    - Utilizando el método `str.contains()` para buscar patrones o palabras específicas en columnas de texto.
    ```python
    data = {'Nombre': ['Ana', 'Luis', 'Pedro'], 'Edad': [25, 30, 22]}
    df = pd.DataFrame(data)

    # Filtrar filas donde 'Nombre' contiene la letra 'a'
    df_filtrado = df[df['Nombre'].str.contains('a')]
    print(df_filtrado)
    ```
    **Salida**:
    | Índice | Nombre | Edad |
    |--------|--------|------|
    | 0      | Ana    | 25   |
    | 2      | Pedro  | 22   |

- **Filtrar por valores nulos**:
    ```python
    data = {'A': [1, 2, None], 'B': [4, None, 6]}
    df = pd.DataFrame(data)

    # Filtrar filas donde 'A' tiene valores no nulos
    df_filtrado = df[df['A'].notnull()]
    print(df_filtrado)
    ```
    **Salida**:
    | Índice | A    | B    |
    |--------|------|------|
    | 0      | 1.0  | 4.0  |
    | 1      | 2.0  | NaN  |

---

#### **Filtrar por índices**:
- Puedes seleccionar filas basadas en los índices utilizando `loc[]` o `iloc[]` junto con condiciones.

- **Ejemplo con índices específicos**:
    ```python
    df_filtrado = df.loc[[0, 2]]
    print(df_filtrado)
    ```
    **Salida**:
    | Índice | Columna A | Columna B |
    |--------|-----------|-----------|
    | 0      | 5         | 100       |
    | 2      | 15        | 300       |

- **Ejemplo filtrando por un rango de índices**:
    ```python
    df_filtrado = df.iloc[1:3]
    print(df_filtrado)
    ```
    **Salida**:
    | Índice | Columna A | Columna B |
    |--------|-----------|-----------|
    | 1      | 10        | 200       |
    | 2      | 15        | 300       |

---

#### **Visualización del proceso de filtrado**:
Un diagrama de flujo básico del filtrado sería:

1. Determinar las condiciones o criterios de filtrado.
2. Aplicar las condiciones al DataFrame utilizando las técnicas anteriores.
3. Visualizar o almacenar el resultado filtrado.


---

### 3. Modificación de datos:

La modificación de datos en un DataFrame es esencial para preparar los datos para análisis o modelado. Puedes agregar nuevas columnas, modificar las existentes o actualizar valores basados en condiciones.

---

#### **Crear nuevas columnas**:
- Puedes crear nuevas columnas directamente asignándoles valores calculados a partir de columnas existentes o utilizando valores constantes.

- **Ejemplo básico**:
    ```python
    import pandas as pd
    data = {'Columna A': [10, 20, 30]}
    df = pd.DataFrame(data)

    # Crear una nueva columna multiplicando los valores de 'Columna A' por 2
    df['Nueva Columna'] = df['Columna A'] * 2
    print(df)
    ```
    **Salida**:
    | Índice | Columna A | Nueva Columna |
    |--------|-----------|---------------|
    | 0      | 10        | 20            |
    | 1      | 20        | 40            |
    | 2      | 30        | 60            |

- **Crear columnas basadas en condiciones**:
    ```python
    # Crear una columna basada en una condición
    df['Condicional'] = df['Columna A'].apply(lambda x: 'Mayor a 15' if x > 15 else 'Menor o igual a 15')
    print(df)
    ```
    **Salida**:
    | Índice | Columna A | Nueva Columna | Condicional          |
    |--------|-----------|---------------|----------------------|
    | 0      | 10        | 20            | Menor o igual a 15   |
    | 1      | 20        | 40            | Mayor a 15           |
    | 2      | 30        | 60            | Mayor a 15           |

---

#### **Actualizar valores existentes**:
- Puedes actualizar valores específicos utilizando métodos como `loc[]` o aplicando funciones sobre columnas.

- **Ejemplo básico**:
    ```python
    # Actualizar valores donde 'Columna A' sea mayor a 15
    df.loc[df['Columna A'] > 15, 'Nueva Columna'] = 0
    print(df)
    ```
    **Salida**:
    | Índice | Columna A | Nueva Columna |
    |--------|-----------|---------------|
    | 0      | 10        | 20            |
    | 1      | 20        | 0             |
    | 2      | 30        | 0             |

- **Actualizar con valores constantes**:
    ```python
    # Establecer un valor constante en una columna
    df['Nueva Columna'] = 100
    print(df)
    ```
    **Salida**:
    | Índice | Columna A | Nueva Columna |
    |--------|-----------|---------------|
    | 0      | 10        | 100           |
    | 1      | 20        | 100           |
    | 2      | 30        | 100           |

- **Actualizar usando una función personalizada**:
    ```python
    # Aplicar una función para modificar los valores de 'Columna A'
    df['Columna A'] = df['Columna A'].apply(lambda x: x + 5)
    print(df)
    ```
    **Salida**:
    | Índice | Columna A | Nueva Columna |
    |--------|-----------|---------------|
    | 0      | 15        | 100           |
    | 1      | 25        | 100           |
    | 2      | 35        | 100           |

---

#### **Eliminar columnas o filas**:
- En ocasiones, es necesario eliminar columnas o filas que ya no son relevantes.

- **Eliminar una columna**:
    ```python
    # Eliminar la columna 'Nueva Columna'
    df = df.drop(columns=['Nueva Columna'])
    print(df)
    ```
    **Salida**:
    | Índice | Columna A |
    |--------|-----------|
    | 0      | 15        |
    | 1      | 25        |
    | 2      | 35        |

- **Eliminar filas basadas en una condición**:
    ```python
    # Eliminar filas donde 'Columna A' sea mayor a 30
    df = df[df['Columna A'] <= 30]
    print(df)
    ```
    **Salida**:
    | Índice | Columna A |
    |--------|-----------|
    | 0      | 15        |
    | 1      | 25        |

---

#### **Renombrar columnas**:
- Cambiar el nombre de columnas es útil para mejorar la legibilidad o adaptar nombres a un formato estándar.

- **Ejemplo de renombrar columnas**:
    ```python
    df = df.rename(columns={'Columna A': 'Nueva Columna A'})
    print(df)
    ```
    **Salida**:
    | Índice | Nueva Columna A |
    |--------|-----------------|
    | 0      | 15              |
    | 1      | 25              |

---

#### **Agregar nuevas filas**:
- Se pueden agregar nuevas filas utilizando el método `append()` o concatenando DataFrames.

- **Ejemplo de agregar una fila**:
    ```python
    nueva_fila = {'Columna A': 40, 'Nueva Columna': 200}
    df = df.append(nueva_fila, ignore_index=True)
    print(df)
    ```
    **Salida**:
    | Índice | Columna A | Nueva Columna |
    |--------|-----------|---------------|
    | 0      | 15        | 100           |
    | 1      | 25        | 100           |
    | 2      | 40        | 200           |

---

## Tema 2.3: Identificación y Manejo de Valores Nulos

### 1. Detección de valores nulos:

La detección de valores nulos es una tarea clave para identificar datos faltantes que podrían afectar el análisis o los modelos predictivos.

---

#### **Identificar valores faltantes**:
- El método `isnull()` devuelve un DataFrame booleano donde cada celda es `True` si el valor es nulo y `False` en caso contrario.
- **Ejemplo básico**:
    ```python
    import pandas as pd
    data = {'Columna A': [1, None, 3], 'Columna B': [4, 5, None]}
    df = pd.DataFrame(data)

    # Identificar valores nulos por columna
    print(df.isnull().sum())
    ```
    **Salida**:
    ```
    Columna A    1
    Columna B    1
    dtype: int64
    ```

---

#### **Visualizar valores nulos**:
- Para una vista rápida del total de valores faltantes en el DataFrame:
    ```python
    print(df.isnull().sum().sum())  # Total de valores nulos en todo el DataFrame
    ```
    **Salida**:
    ```
    2
    ```

---

#### **Detección avanzada con métodos adicionales**:
- **Verificar si existen valores nulos**:
    ```python
    print(df.isnull().any())  # True si al menos una celda es nula por columna
    ```
    **Salida**:
    ```
    Columna A    True
    Columna B    True
    dtype: bool
    ```

- **Verificar filas con valores nulos**:
    ```python
    print(df[df.isnull().any(axis=1)])  # Filas que contienen al menos un valor nulo
    ```
    **Salida**:
    | Índice | Columna A | Columna B |
    |--------|-----------|-----------|
    | 1      | NaN       | 5.0       |
    | 2      | 3.0       | NaN       |

---

#### **Visualización gráfica de valores nulos**:
- Las bibliotecas como `matplotlib` o `seaborn` permiten visualizar los valores faltantes.
- Esta visualización ayuda a identificar patrones de valores faltantes.
- **Ejemplo con heatmap de `seaborn`**:
    ```python
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.heatmap(df.isnull(), cmap='viridis', cbar=False)
    plt.title('Mapa de valores nulos')
    plt.show()
    ```



---

### 2. Estrategias para manejar valores nulos:

El manejo de valores nulos es esencial para preparar los datos antes del análisis. Existen diversas estrategias dependiendo del impacto de los datos faltantes en el análisis.

---

#### **Eliminación de datos**:
- Se puede optar por eliminar filas o columnas con valores nulos si estos representan una pequeña proporción del conjunto de datos.
- **Ejemplos básicos**:
    ```python
    # Eliminar filas con al menos un valor nulo
    df_sin_nulos_filas = df.dropna()

    # Eliminar columnas con al menos un valor nulo
    df_sin_nulos_columnas = df.dropna(axis=1)

    # Eliminar filas solo si todas las celdas están nulas
    df_sin_nulos_todas = df.dropna(how='all')

    # Eliminar filas con menos de un número mínimo de valores no nulos
    df_sin_nulos_thresh = df.dropna(thresh=2)  # Al menos 2 valores no nulos por fila
    ```

---

#### **Imputación de valores**:
- La imputación consiste en reemplazar los valores nulos con datos razonables (promedio, mediana, constantes, etc.).
- **Ejemplo básico con promedio**:
    ```python
    # Reemplazar valores nulos con el promedio de la columna
    df['Columna A'].fillna(df['Columna A'].mean(), inplace=True)
    ```

- **Otras técnicas de imputación**:
    ```python
    # Rellenar con un valor constante
    df['Columna A'].fillna(0, inplace=True)

    # Rellenar con la mediana
    df['Columna A'].fillna(df['Columna A'].median(), inplace=True)

    # Rellenar con el valor más frecuente (moda)
    df['Columna A'].fillna(df['Columna A'].mode()[0], inplace=True)

    # Rellenar usando interpolación (valores intermedios calculados)
    df['Columna A'].interpolate(method='linear', inplace=True)
    ```

---

#### **Imputación avanzada**:
- Puedes utilizar modelos predictivos o algoritmos para estimar los valores faltantes.
    ```python
    from sklearn.impute import SimpleImputer

    # Reemplazo de valores nulos con la mediana utilizando Scikit-learn
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='median')
    df['Columna A'] = imputer.fit_transform(df[['Columna A']])
    ```

---

#### **Seleccionar estrategia adecuada**:
- **Eliminar datos**:
  - Útil cuando los valores nulos son pocos y no tienen un impacto significativo en el análisis.
- **Imputación simple**:
  - Ideal para conjuntos de datos pequeños o cuando se asume que los valores faltantes son aleatorios.
- **Imputación avanzada**:
  - Útil para grandes conjuntos de datos donde los valores nulos no son aleatorios y se pueden estimar con precisión.

---

## Tema 2.4: Detección y Manejo de Datos Atípicos

### 1. Qué son los datos atípicos:

Los **datos atípicos**, también conocidos como **outliers**, son valores que se encuentran significativamente alejados del rango esperado en un conjunto de datos. Estos valores se desvían de la distribución central y pueden tener un impacto importante en los resultados de los análisis estadísticos, como los modelos predictivos. Los datos atípicos pueden surgir por varias razones, incluyendo:

- **Errores de medición**: Pueden ser causados por fallos en los instrumentos o malas prácticas de recolección de datos.
- **Datos extremados**: Algunos registros pueden representar eventos raros, pero no necesariamente errores.
- **Variabilidad natural**: En ciertos campos, como las ciencias sociales o económicas, los datos atípicos pueden ser representativos de eventos o fenómenos específicos.

Imagina que en una clase de 10 alumnos, 9 miden entre 1.40 m y 1.50 m, pero hay uno que mide 1.90 m. Ese sería un outlier.

Los **boxplots** son uno de los métodos más comunes para visualizar los datos atípicos, ya que muestran la distribución de los datos a través de los cuartiles, lo que permite identificar valores que se encuentran fuera del rango intercuartílico (IQR). El **IQR** es el rango intercuartílico que mide la dispersión de los datos entre el 25% y el 75%.

Además de los boxplots, otros gráficos útiles para identificar datos atípicos incluyen **scatter plots** y diagramas de dispersión, que muestran las relaciones entre las variables y pueden ayudar a detectar puntos dispersos.

---

**Ejemplo visual de un boxplot mostrando outliers**:
- <img src="https://miro.medium.com/v2/resize:fit:1200/1*0MPDTLn8KoLApoFvI0P2vQ.png" width="600">
   

---

### Características de los datos atípicos:
- **Desvío significativo**: Los datos atípicos se desvían considerablemente de la media y del rango intercuartílico, lo que los hace destacar.
- **Influencia**: Los datos atípicos pueden influir en la media, la varianza y otros estadísticos, lo que puede afectar los modelos predictivos.
- **Visibilidad**: Se pueden identificar visualmente utilizando herramientas como los boxplots, scatter plots o histogramas.

Los datos atípicos pueden ser importantes para entender ciertos fenómenos en los datos, como situaciones extremas o eventos específicos, pero también deben ser manejados adecuadamente para evitar que distorsionen los análisis estadísticos.

---

### 2. Métodos para identificar atípicos:

#### **Cuartiles y rango intercuartílico (IQR)**:
Uno de los métodos más comunes para identificar datos atípicos es el uso de **cuartiles** y el **rango intercuartílico (IQR)**. Este método se basa en la medición de dispersión de los datos divididos en cuatro partes iguales. El **cuartil 1 (Q1)** representa el 25% de los datos, mientras que el **cuartil 3 (Q3)** representa el 75% de los datos. 

- **Cuartil 1 (Q1)**: Es el valor que se encuentra en el 25% del conjunto de datos ordenado.
- **Cuartil 3 (Q3)**: Es el valor que se encuentra en el 75% del conjunto de datos ordenado.
- **Rango intercuartílico (IQR)**: Se calcula como \( IQR = Q3 - Q1 \), proporcionando una medida del rango central de los datos entre el 25% y el 75%.

Los **valores atípicos** se identifican como aquellos que están más allá de `Q1 - 1.5 * IQR` o `Q3 + 1.5 * IQR`. Cualquier dato que caiga fuera de este rango es considerado un outlier.

- **Ejemplo práctico**:
    ```python
    import pandas as pd
    
    # Crear un DataFrame de ejemplo
    data = {'Columna A': [1, 3, 5, 7, 9, 100, 200, 300]}
    df = pd.DataFrame(data)
    
    Q1 = df['Columna A'].quantile(0.25)
    Q3 = df['Columna A'].quantile(0.75)
    IQR = Q3 - Q1
    
    outliers = df[(df['Columna A'] < Q1 - 1.5 * IQR) | (df['Columna A'] > Q3 + 1.5 * IQR)]
    print(outliers)
    ```
    **Salida**:
    ```
    Columna A
    6    100
    7    200
    8    300
    ```

#### **Visualización**:
Además de los cuartiles y el rango intercuartílico, los gráficos también son herramientas valiosas para identificar datos atípicos:

- **Boxplots**: Muestran visualmente los cuartiles, el rango intercuartílico, y los valores atípicos. Los **outliers** se representan con puntos fuera del recuadro del boxplot.
- **Scatter plots**: Ayudan a observar la dispersión de los datos y los puntos dispersos que podrían representar datos atípicos.

---

**Ejemplo visual de un scatter plot que muestra datos atípicos**:
- <img src="https://fhernanb.github.io/libro_regresion/images/atipico_influyente.png" width="600">

Ambos gráficos permiten detectar rápidamente dónde están los valores atípicos, permitiendo una mejor toma de decisiones para su manejo.

---

### 3. Estrategias para manejar datos atípicos:

Cuando se identifican datos atípicos en un conjunto de datos, es importante considerar distintas estrategias para manejarlos, ya que estos pueden afectar significativamente la calidad y precisión de los análisis estadísticos. Las **estrategias** principales incluyen **eliminación** y **transformación**. Cada una de estas opciones tiene su enfoque y uso en función del objetivo del análisis, la naturaleza de los datos, y el propósito del estudio.

---

#### **Conceptos clave sobre datos atípicos**:

- **Datos atípicos**: Son aquellos valores que se desvían considerablemente del patrón central en un conjunto de datos. Pueden surgir por diversos motivos, como errores de medición, registros raros, o representaciones extremas de fenómenos naturales.
  
- **Cuartiles y rango intercuartílico (IQR)**: El **cuartil 1 (Q1)** divide los datos en dos partes iguales, donde el 25% de los datos se encuentran por debajo de Q1. El **cuartil 3 (Q3)** divide los datos de la misma manera, pero para el 75%. El **rango intercuartílico (IQR)** es la diferencia entre Q3 y Q1, y se utiliza para calcular el límite superior e inferior para identificar valores atípicos, que son aquellos que se encuentran fuera de \( Q1 - 1.5 * IQR \) o \( Q3 + 1.5 * IQR \).

- **Visualización**: Los gráficos como **boxplots** y **scatter plots** son herramientas útiles para identificar datos atípicos, ya que muestran la dispersión de los datos y resaltan los puntos que están lejos de la distribución central.

---

#### **Eliminación de valores extremos**:
- Esta estrategia implica la **eliminación** de aquellos valores que se encuentran fuera del rango intercuartílico, definido como \( Q1 - 1.5 * IQR \) o \( Q3 + 1.5 * IQR \). Los datos que caen fuera de estos límites se consideran atípicos y pueden ser eliminados si representan errores o puntos irrelevantes para el análisis.

- **Ventajas**:
  - Ayuda a eliminar posibles errores de medición o registros incorrectos.
  - Protege la precisión de los modelos estadísticos al reducir la influencia de los valores extremos.
- **Desventajas**:
  - Pérdida de información, ya que se eliminan algunos puntos del conjunto de datos.
  - Puede ser riesgoso si los datos atípicos son representativos de fenómenos naturales o eventos importantes.

- **Ejemplo práctico**:
    ```python
    import pandas as pd
    
    # Crear un DataFrame de ejemplo
    data = {'Columna A': [1, 3, 5, 7, 9, 100, 200, 300]}
    df = pd.DataFrame(data)
    
    Q1 = df['Columna A'].quantile(0.25)
    Q3 = df['Columna A'].quantile(0.75)
    IQR = Q3 - Q1
    
    # Eliminación de valores atípicos
    df_limpio = df[(df['Columna A'] >= Q1 - 1.5 * IQR) & (df['Columna A'] <= Q3 + 1.5 * IQR)]
    print(df_limpio)
    ```
    **Salida**:
    | Columna A |
    |-----------|
    | 1         |
    | 3         |
    | 5         |
    | 7         |
    | 9         |

---

#### **Transformación**:
- Cuando los datos atípicos no son errores, sino que representan eventos naturales o fenómenos específicos, una **transformación** puede ser preferible. Esto implica ajustar los datos atípicos mediante técnicas como **logaritmos** o **normalización** para reducir su influencia.

- **Ventajas**:
  - Ayuda a conservar la información valiosa en los datos atípicos.
  - Reduce la influencia de los valores extremos en los análisis estadísticos.
- **Desventajas**:
  - Puede ser difícil decidir cuál transformación es la más adecuada.
  - No todos los datos atípicos son susceptibles de ser transformados sin perder información.

- **Ejemplo práctico**:
    ```python
    # Transformar los valores atípicos utilizando logarítmica
    df['Columna A'] = df['Columna A'].apply(lambda x: x if (x >= Q1 - 1.5 * IQR) and (x <= Q3 + 1.5 * IQR) else x**0.5)
    print(df)
    ```
    **Salida**:
    | Columna A |
    |-----------|
    | 1         |
    | 3         |
    | 5         |
    | 7         |
    | 9         |
    | 10.0      |
    | 14.0      |
    | 17.32     |

---

Ambas estrategias, **eliminación** y **transformación**, tienen sus aplicaciones dependiendo del contexto de los datos y los objetivos del análisis. Es importante seleccionar la opción que mejor se ajuste al objetivo del estudio y al tipo de datos con los que se trabaja.
