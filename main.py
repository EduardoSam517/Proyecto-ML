"""
Este módulo sirve como punto de entrada principal para la aplicación FastAPI.
Define las rutas API y la lógica para manejar solicitudes.
"""
from fastapi import FastAPI
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

app = FastAPI()

# http://127.0.0.1:8000 # Ruta raiz del puerto
# @app.get("/") # Tenemos el objeto y la ("/") es la ruta raiz. Ejecuta la funcion

steam = pd.read_csv('data_steam.csv')
steam['release_date'] = steam['release_date'].astype(int)


@app.get("/developer_free/{desarrollador}")
def developer_free(desarrollador: str):
    """Cantidad de items y porcentaje de contenido Free por año según empresa desarrolladora"""

    # Filtrar el DataFrame para obtener solo las filas donde 'developer' es el desarrollador de entrada
    df_filt1 = steam[(steam['developer'] == desarrollador)].copy()

    # Agrupe por 'release_date', cuente los valores únicos de 'item_id' y cuente la cantidad de valores de 'price' que son 0
    df_group1 = df_filt1.groupby('release_date').agg(
        {'developer': 'count', 'item_id': pd.Series.nunique, 'price': lambda x: (x == 0).sum()}).reset_index()

    # Calcular el porcentaje de contenido gratuito.
    df_group1['% contenido free'] = round(
        (df_group1['price']/df_group1['developer'])*100, 0)

    df_group1.rename(
        columns={'item_id': 'Cantidad de items', 'release_date': 'Año'}, inplace=True)

    # Elimine las columnas 'developer' y 'price'
    df_group1.drop('price', axis=1, inplace=True)
    df_group1.drop('developer', axis=1, inplace=True)

    result_dicc1 = {
        "Año": df_group1['Año'].to_dict(),
        "Cantidad de items": df_group1['Cantidad de items'].tolist(),
        "% contenido free": df_group1['% contenido free'].tolist()
    }

    return result_dicc1


@app.get('/user_data/{user_id_}')
def user_data(user_id_: str):
    """Debe devolver cantidad de dinero gastado por el usuario, el porcentaje de recomendación en
    base a reviews.recommend y cantidad de items"""

    df_filt2 = steam[steam['user_id'] == user_id_]

    df_group2 = df_filt2.agg({
        'price': 'sum',
        'recommend': 'sum',
        'user_id': 'count'
    }).rename(
        index={'user_id': 'cantidad de items', 'price': 'gasto total'})

    df_group2['% recomendacion'] = (
        (df_group2['recommend']/df_group2['cantidad de items'])*100)

    result_dicc2 = {
        'Usuario': user_id_,
        'Dinero Gastado': str(round(df_group2['gasto total'], 2)) + ' USD',
        '% de Recomendacion': str(round(df_group2['% recomendacion'], 0)) + ' %',
        'Cantidad de Items': df_group2['cantidad de items']
    }

    return result_dicc2


@app.get('/user_for_genre/{genero}')
def user_for_genre(genero: str):
    """Debe devolver el usuario que acumula más horas jugadas para el género dado y una lista de la 
    acumulación de horas jugadas por año"""

    # Filtrar el DataFrame para el género especificado
    df_filt3 = steam[steam['genre'] == genero]

    # Agrupar por 'user_id' y sumar 'playtime_forever'
    df_group3 = df_filt3.groupby('user_id').agg(
        {'playtime_forever': 'sum'}).reset_index()

    # Ordenar por 'playtime_forever'
    df_sort3 = df_group3.sort_values('playtime_forever', ascending=False)

    # Filtrar el DataFrame original para el 'user_id' con el mayor 'playtime_forever'
    df_filt31 = steam[steam['user_id'] == df_sort3.iloc[0, 0]]

    # Agrupar por 'year_posted' y sumar 'playtime_forever'
    df_group31 = df_filt31.groupby('year_posted').agg(
        {'playtime_forever': 'sum'}).reset_index()

    df_group31 = df_group31.rename(
        columns={'year_posted': 'Año', 'playtime_forever': 'Horas'})

    # Convertir el resultado a un diccionario
    result_dicc3 = {
        'usuario con mas horas jugadas para el genero ' + genero: df_sort3.iloc[0, 0],
        'horas jugadas': df_group31.to_dict('records')
    }

    return result_dicc3


@app.get('/best_developer_year/{anio}')
def best_developer_year(anio: int):
    """Devuelve el top 3 de desarrolladores con juegos MÁS recomendados por usuarios para el año dado"""

    # Filtrar el DataFrame para obtener solo las filas donde 'year_posted' es el año de entrada
    df_filt4 = steam[steam['year_posted'] == anio].copy()

    # Crea una nueva columna 'reviews.recommend' que es la suma de 'recommend' y 'sentiment_analysis'
    df_filt4['reviews.recommend'] = df_filt4['recommend'] + \
        df_filt4['sentiment_analysis']

    # Agrupar por 'desarrollador' y sumar 'reviews.recommend'
    df_group4 = df_filt4.groupby('developer')[
        'reviews.recommend'].sum().reset_index()

    # Ordenar por 'reviews.recommend'
    df_sort4 = df_group4.sort_values('reviews.recommend', ascending=False)

    # Crea una lista de diccionarios para los 3 principales desarrolladores
    result_list4 = [{'Puesto ' + str(i+1): df_sort4.iloc[i, 0]}
                    for i in range(3)]

    return result_list4


@app.get('/developer/{desarrolladora}')
def developer(desarrolladora: str):
    """Según el desarrollador, se devuelve un diccionario con el nombre del desarrollador como llave y una lista 
    con la cantidad total de registros de reseñas de usuarios que se encuentren categorizados con un análisis de 
    sentimiento como valor positivo o negativo"""

    # Filtrar el DataFrame para obtener solo las filas donde 'developer' es el desarrollador de entrada
    # y 'sentiment_analysis' es 2 o 0
    df_filt5 = steam[(steam['developer'] == desarrolladora) &
                     steam['sentiment_analysis'].isin([0, 2])]

    # Agrupar por 'developer' y 'sentiment_analysis', y contar 'developer'
    df_group5 = df_filt5.groupby(
        ['developer', 'sentiment_analysis']).size().reset_index(name='count')

    # Crear un diccionario con los resultados
    result_dicc5 = {
        desarrolladora: [
            'Negativo = ' +
            str(df_group5[df_group5['sentiment_analysis'] == 0]
                ['count'].values[0]),
            'Positivo = ' +
            str(df_group5[df_group5['sentiment_analysis'] == 2]
                ['count'].values[0])
        ]
    }

    return result_dicc5


@app.get('/recomendacion_juego/{item_id}')
def recomendacion_juego(item_id : int):

    """Ingresando el id de un juego, deberíamos recibir una lista con 5 juegos recomendados para dicho juego"""

    data = pd.read_csv('juegos_steam.csv')
    data_juegos_steam = pd.read_csv('juegos_id.csv')

    tfidv = TfidfVectorizer(min_df=2, max_df=0.7, token_pattern=r'\b[a-zA-Z0-9]\w+\b')
    data_vector = tfidv.fit_transform(data['features'])

    data_vector_df = pd.DataFrame(data_vector.toarray(), index=data['item_id'], columns = tfidv.get_feature_names_out())

    vector_similitud_coseno = cosine_similarity(data_vector_df.values)

    cos_sim_df = pd.DataFrame(vector_similitud_coseno, index=data_vector_df.index, columns=data_vector_df.index)

    juego_simil = cos_sim_df.loc[item_id]

    simil_ordenada = juego_simil.sort_values(ascending=False)
    resultado = simil_ordenada.head(6).reset_index()

    result_df = resultado.merge(data_juegos_steam, on='item_id',how='left')

    # Obtén el título del juego de entrada.
    juego_title = data_juegos_steam[data_juegos_steam['item_id'] == item_id]['title'].values[0]

    # Crea un mensaje de salida
    message = f"Si te gustó el juego {item_id} : {juego_title}, también te pueden gustar:"

    # Crea un diccionario de retorno de la funcion
    result_dict = {
        'mensaje': message,
        'juegos_recomendados': result_df['title'][1:6].tolist()
    }

    return result_dict