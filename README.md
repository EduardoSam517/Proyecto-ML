# <h1 align=center> **PROYECTO INDIVIDUAL Nº1 (Machine Learning)**
# <h1 align=center> **Eduardo Samuel Garduño Gonzalez**
### <h1 align=center> `Machine Learning Operations` (MLOps)


En este proyecto se ha creado un modelo de aprendizaje automático (ML) con el propósito de abordar un problema empresarial: Steam requiere la creación de un sistema de recomendación de videojuegos para sus usuarios. Durante la ejecución del proyecto, se llevaron a cabo labores de ingeniería de datos para desarrollar un Producto Mínimo Viable (PMV) centrado en la extracción de algunos datos específicos y en la recomendación de juegos similares a los proporcionados por el usuario de Steam.

El objetivo primordial del proyecto consistió en desarrollar una API para poner a disposición los datos de la empresa utilizando el framework FastAPI. A través de esta API, se pueden realizar consultas específicas a una base de datos filtrada, garantizando la integridad y calidad de los datos mediante un exhaustivo trabajo de Extracción, Transformación y Carga (ETL).

## <h1 align=center> **`Extracción, Transformación y Carga de Datos (Descripción General)`**

Carpeta: ETL

El objetivo principal de la carga de datos consistió en procesar 3 archivos JSON, empleando una función que iteraba sobre cada línea y la transformaba en un diccionario de Python. Estos diccionarios se almacenaron en una lista. Posteriormente, la lista de diccionarios se convirtió en un DataFrame de pandas. A continuación, se detalla de forma general el proceso de limpieza y transformación de los DataFrames de pandas generados a partir de los archivos JSON. A continuación se describe paso a paso el funcionamiento de cada sección del código:

1. La función `explode()` se utiliza para transformar cada elemento de una lista en una fila, replicando los valores del índice. En este caso, se utiliza para expandir la columna con datos anidados en el marco de datos.
2. la función `json_normalize()` se utiliza para aplanar datos JSON semiestructurados en una tabla plana. Aquí, se utiliza para convertir la columna especifica con datos anidados, que es una lista de diccionarios, en un DataFrame. Luego, la función `set_index()` se usa para establecer el índice del nuevo DataFrame para que coincida con el índice de la columna 'elementos' en el DataFrame original.
3. La función `concat()` se usa para concatenar dos o más DataFrames a lo largo de un eje particular (columnas en este caso). Aquí, se utiliza para unir el DataFrame original con el nuevo DataFrame creado a partir de la columna con los datos anidados.
4. la función `drop()` se utiliza para eliminar etiquetas específicas de filas o columnas, en este caso se eliminaron columnas especificas con poca o nula relevancia para el análisis.
5. La función `dropna()` se utiliza para eliminar valores faltantes. Aquí, se utiliza para eliminar filas donde faltanen las columnas seleccionadas.
8. selección de columnas del DataFrame definitivas.
9. La función `drop_duplicates()` se utiliza para eliminar filas duplicadas del DataFrame.
10. dropna()  elimina las filas restantes con valores faltantes del DataFrame.
11. exportar archivos: La función `to_csv()` se utiliza para escribir el DataFrame en un archivo CSV. Aquí, se utiliza para guardar el DataFrame en 3 archivos llamados 'user_items.csv', ‘user_reviews.csv’ y ‘games.csv’. El parámetro `index=False` se utiliza para evitar que los pandas escriban índices de filas en el archivo CSV.


**Condiciones específicas para el tratamiento de datos:**

a.	Para el caso del dataset ‘australian_user_items’:

Carpeta: ETL

Documento: ETL_items.ipynb

1.	 `data_it3 = data_it3[data_it3['playtime_forever'] != 0]`: esta línea filtra el DataFrame para incluir solo filas donde 'playtime_forever' no es igual a 0. Es decir, solo se consideraron todos aquellos ítems donde el ti9empode juego fue mayor a 0, en otras palabras, que fueron jugados como mínimo una hora.

b.	Para el caso del dataset ‘australian_user_reviews’:

Carpeta: ETL

Documento: ETL_reviews.ipynb


1.	`data_re2 = data_re1['reviews'].apply(pd.Series)`: La función `apply()`: se utiliza para aplicar una función a lo largo de un eje de un DataFrame. Aquí, se utiliza para convertir la columna 'reviews', que es una lista de diccionarios, en un DataFrame, para luego ser concatenado con el DataFrame original.

2.	`data_re3['year_posted'] = data_re3['posted'].str.extract('(\d{4})')`: Esta línea crea una nueva columna 'year_posted' extrayendo el año del 'publicado' ' columna usando una expresión regular.

3.	`data_re3['recommend'] = data_re3['recommend'].replace({'False': 0, 'True': 1}).astype(int)`: esta línea reemplaza los valores booleanos 'True' y ' Falso' en la columna ' recommend ' con 1 y 0, respectivamente, y cambia el tipo de datos de la columna a entero.

4.	La función `get_sentiment(text)` está definida para analizar el sentimiento del texto utilizando la biblioteca TextBlob. Devuelve 0 si la polaridad del sentimiento es menor que -0.1 (negativo), 2 si es mayor que 0,1 (positivo) y 1 en caso contrario (neutral).

5.	`data_re3['sentiment_analysis'] = data_re3['review'].apply(get_sentiment)`: esta línea aplica la función `get_sentiment()` a la columna 'review' y almacena los resultados en una nueva columna 'sentiment_analysis' .

Nota: a considerar que los registros válidos fueron aquellos que fueron comentados, valorados y recomendados y por lo tanto se pudo realizar el análisis pertinente.

c.	Para el caso del dataset ‘output_steam_games’:

Carpeta: ETL

Documento: ETL_games.ipynb


1.	`data_games['release_date'] = pd.to_datetime(data_games['release_date'], errores='coerce').dt.year`: esta línea convierte la columna 'release_date' a fecha y hora y extrae el año.

2.	Lo siguiente es reemplazar varios valores de cadena (‘Free to Play’, ‘’Free To Play, ’Play For Free’) en la columna 'price' con 0.

3.	`data_games = data_games[pd.to_numeric(data_games['price'], errores='coerce').notnull()]`: esta línea convierte la columna 'price' a numérica y elimina cualquier fila con valores no numéricos en el columna 'price'.

4.	`data_games1 = data_games['genres'].apply(pd.Series)`: La función `apply()` se utiliza para aplicar una función a lo largo de un eje del DataFrame Aquí, se utiliza para dividir la columna 'genres' en varias columnas, luego se concatena el DataFrame generado al DataFrame original, se toma como valido el primer valor de la lista en la columna ‘genres’, y se eliminan los demás “géneros” y también se elimina la columna ‘genres’.

**Relación y unión de tablas:**

Carpeta: ETL - MERGE

Documento: MERGE_PI.ipynb

Para esta parte se realiza la limpieza, transformación y fusión de datos en múltiples DataFrames de pandas. Aquí hay una explicación paso a paso:

1.	Se crea un identificador único ‘id’ en los DataFrame 'items' y ‘reviews’ al concatenar las columnas 'user_id' y 'item_id'.

2.	`merged_df = reviews.merge(games, on='item_id', how='left')`: La función `merge()` se utiliza para fusionar dos DataFrames basados en una columna común. En este caso, se fusiona el DataFrame de 'reviews' con el DataFrame de 'games' según la columna 'item_id'.

3.	`steam = items.merge(merged_df, on='id')`: esta línea fusiona el DataFrame 'items' con el DataFrame 'merged_df' según el identificador único 'id'.

4.	`steam = steam ['id','user_id','item_id','title','genre','developer','release_date', 'price','recommend','year_posted','sentiment_analysis ','playtime_forever']]`: esta línea selecciona solo las columnas especificadas del DataFrame 'steam'.

5.	`steam.to_csv('data_steam.csv', index=False)`: la función `to_csv()` se usa para escribir el DataFrame en un archivo CSV. Aquí, se utiliza para guardar el DataFrame 'steam' en un archivo llamado 'data_steam.csv'.

## <h1 align=center> **`Análisis de Datos Exploratorio`** 

Documento: EDA_PI.ipynb

Tras el análisis de las variables numéricas mediante la descripción estadística y la visualización gráfica, se detectaron valores atípicos (outliers) en las variables "fecha de lanzamiento", "precio" y "tiempo de juego total". Estos valores fueron procesados y eliminados, ya que podían afectar negativamente al análisis posterior.

Se calcula el rango intercuartil (IQR) para tres variables diferentes en tres marcos de datos de pandas diferentes: data1 = data['release_date'], data2 = data['price], data3 = data['playtime_forever']. 
`Q1 = data.quantile(0.25)`: La función `quantile()` se utiliza para calcular el cuartil de un DataFrame. Aquí, se utiliza para calcular el primer cuartil (Q1) del marco de datos 'data'.
`Q3 = data.quantile(0.75)`: esta línea calcula el tercer cuartil (Q3) del DataFrame 'data'.
`IQR = Q3 - Q1`: Esta línea calcula el Rango Intercuartil (IQR) restando Q1 de Q3. El IQR es una medida de dispersión estadística y es el rango dentro del cual se encuentra el 50% central de los datos. Se repite el mismo proceso para los caculos respectivos para las tres variables. Es importante mencionar que, El IQR es una medida sólida de diferencial que se ve menos afectada por valores atípicos que el rango.

Posteriormente se realiza el cálculo del rango y se definen los límites para luego eliminar valores fuera de rango.
Luego de eliminar los outliers de las variables mencionadas, se nota una distribución típica de los datos, pero con diferente comportamiento.
El siguiente paso es el tratamiento de las variables categóricas, las cuales fueron transformadas a través de una instancia de LabelEncoder y se usa para transformar las columnas ‘user_id’, ‘item_id’, ‘genre’, ‘developer’ y ‘title’ específicamente del DataFrame data en valores numéricos. Los valores transformados se almacenan en nuevas columnas llamadas user_id1, item_id1, genre1, developer1 y title1, respectivamente.

Luego se analiza la correlación de las variables con la variable objetivo (‘item_id’), para esto se utilizaron dos métodos, por un lado, se calculó una matriz de correlación y posteriormente se graficó para observar de manera más visual la correlación del DataFrame, por otro lado, se usó el método de SelectKBest(mutual_info_classif, k), que se utiliza para seleccionar las mejores k características en función de sus puntuaciones. La función de puntuación utilizada para evaluar las características se puede especificar utilizando el parámetro score_func. En este caso, mutual_info_classif es la función de puntuación que se está utilizando. Estima la información mutua entre cada característica y la variable objetivo para variables objetivo discretas. En ambos casos se determinó que las variables, características o etiquetas más relacionadas son: título, desarrollador, año de lanzamiento, precio, tiempo de juego y género.

## <h1 align=center> **`Machine Learning`**

Documento: ML_PI_SC.ipynb

**Preparación de datos:**
	Para este apartado se dispuso a preparar los datos para aplicar el método de Similitud de Coseno con las variables más relacionadas con el target, en ese sentido, se generan dos subdatasets para obtener los resultados deseados, por un lado ‘juegos_id’ que contiene ‘item_id’ y ‘título del juego’, y el otro ‘juegos_steam’, dataframe contiene ‘item_id’ y ‘features’, siendo ‘features’ la combinacion las columnas 'title', 'developer' y 'realase_date' en una sola columna 'features', con valores separados por una coma y un espacio, siendo estas últimas variables mencionadas las seleccionadas para aplicar el modelo de similitud de coseno para la recomendación.  

**Modelo de recomendación:**
explicación paso a paso de lo que hace cada parte del código de la función definida `recomendacion_juego(item_id)`:

`data = pd.read_csv('juegos_steam.csv')` y 
`data_juegos_steam = pd.read_csv('juegos_id.csv')`
Estas líneas leen datos de dos archivos CSV en dos DataFrames pandas.

`tfidv = TfidfVectorizer(min_df=2, max_df=0.7, token_pattern=r'\b[a-zA-Z0-9]\w+\b')` 
esta línea inicializa un TfidfVectorizer, que transforma el texto en vectores de características. que se puede utilizar como entrada para el estimador. Establece la frecuencia mínima de documentos para filtrar términos que ocurren en menos de 2 documentos, y la frecuencia máxima de documentos para filtrar términos que ocurren en más del 70% de los documentos.

`data_vector = tfidv.fit_transform(data['features'])`:
esta línea ajusta el TfidfVectorizer a la columna 'features' del DataFrame `data` y luego transforma las 'features' en una matriz de TF-IDF.

`data_vector_df = pd.DataFrame(data_vector.toarray(), index=data['item_id'], columns = tfidv.get_feature_names_out())` 
esta línea convierte la matriz de características TF-IDF en un DataFrame.

`vector_similitud_coseno = cosine_similarity(data_vector_df.values)`
Esta línea calcula la similitud del coseno entre todos los pares de vectores TF-IDF.

`cos_sim_df = pd.DataFrame(vector_similitud_coseno, index=data_vector_df.index, columns=data_vector_df.index)`
 Esta línea convierte la matriz de similitud de coseno en un DataFrame.

`juego_simil = cos_sim_df.loc[item_id]`
Esta línea selecciona la fila correspondiente al ‘item_id’ del juego de entrada del DataFrame de similitud de coseno.

`simil_ordenada = juego_simil.sort_values(ascending=False)` 
Esta línea ordena la fila seleccionada en orden descendente de similitud de coseno.

`resultado = simil_ordenada.head(6).reset_index()`
Esta línea selecciona los 6 juegos más similares al juego de entrada y restablece el índice del DataFrame resultante.

`result_df = resultado.merge(data_juegos_steam, on='item_id',how='left')`
Esta línea fusiona el DataFrame resultante con el DataFrame 'data_juegos_steam' basado en la columna 'item_id', para obtener los títulos de Los juegos recomendados.
Las siguientes tres líneas crean un mensaje recomendando los 5 mejores juegos más similares al juego de entrada y almacenan el mensaje y los juegos recomendados en un diccionario.

`return result_dict`
 esta línea devuelve el diccionario que contiene el mensaje y los juegos recomendados.

## <h1 align=center> **`Funciones y API`**

Documento: main.py

Creación de una app FastAPI que brinda múltiples puntos de acceso para explorar y examinar un dataset de juegos de la plataforma Steam. Los datos se importan desde un archivo CSV y se manejan mediante pandas. La app FastAPI atiende peticiones GET en diversos puntos de acceso, cada uno ejecutando una variedad de análisis sobre los datos.

Los endpoints incluyen:

- `/developer_free/{desarrollador}`: Este endpoint devuelve el número de elementos y el porcentaje de contenido gratuito por año para un desarrollador determinado.

- `/user_data/{user_id}`: este punto final devuelve la cantidad total gastada por un usuario, el porcentaje de recomendación basado en reseñas y la cantidad de artículos.

- `/user_for_genre/{genero}`: Este endpoint devuelve el usuario que ha pasado más horas jugando para un género determinado y una lista de la acumulación de horas jugadas por año.

- `/best_developer_year/{anio}`: este punto final devuelve los 3 principales desarrolladores con los juegos más recomendados por los usuarios para un año determinado.

- `/developer/{desarrolladora}`: Este endpoint devuelve un diccionario con el nombre del desarrollador como clave y una lista con el número total de registros de reseñas de usuarios categorizados como sentimiento positivo o negativo.

- `/recomendacion_juego/{item_id}`: Este endpoint devuelve una lista de 5 juegos recomendados para un id de juego determinado.

Cada función de punto final lee el conjunto de datos del archivo CSV, realiza algún procesamiento con pandas y luego devuelve una respuesta. La respuesta generalmente es un diccionario o una lista de diccionarios que contiene los resultados del análisis.

Nota: para hacer las consultas efectivas, se debe escribir en el campo respetando las mayúsculas y minúsculas.

Por ejemplo: en el caso de la primera función, se introduce el dato de esta manera, 'Valve', si se escribe 'Valve' (con minúscula), no devolverá una respuesta a la consulta.
Contacto:

linkedin - https://www.linkedin.com/in/eduardo517gardu%C3%B1o/
email - eduardo517samuel@gmail.com

El estatus correspondiente al proyecto es de: completo/publicado.
