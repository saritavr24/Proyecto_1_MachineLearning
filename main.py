from fastapi import FastAPI
import pandas as pd
import gc
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

app = FastAPI()

@app.get("/")
def mensaje():
    return {"¡Bienvenid@! Esta es una API para realizar consultas sobre juegos de STEAM."}

ruta_games = 'games_sample.parquet'
ruta_reviews = 'reviews_sample.parquet'
ruta_items = 'items_sample.parquet'

def load_data():
    if not hasattr(load_data, "games_sample"):
        load_data.games_sample = pd.read_parquet(ruta_games)
        load_data.reviews_sample = pd.read_parquet(ruta_reviews)
        load_data.items_sample = pd.read_parquet(ruta_items)
    return load_data.games_sample, load_data.reviews_sample, load_data.items_sample

# Función para calcular la cantidad de items y el porcentaje de contenido gratuito según desarrollador
def developer_items_free(df):

    games_agrupado = df.groupby(df['year']).agg(
    cantidad_items=('item_name', 'count'),
    items_gratuitos=('price', lambda x: (x == 0).sum())
    )

    # Calculamos el porcentaje de contenido free
    games_agrupado['contenido_free'] = (games_agrupado['items_gratuitos'] / games_agrupado['cantidad_items']) * 100
    games_agrupado['contenido_free'] = games_agrupado['contenido_free'].round(2).astype(str) + '%'

    # Reseteamos el índice para tener una estructura de DataFrame más convencional
    games_agrupado = games_agrupado.reset_index()
    return(games_agrupado)

# Función para calcular cantidad de dinero gastado por el usuario, el porcentaje de recomendación y cantidad de items.
def user_statistics(user, df, df2, df3):

    # Unir el DataFrame de games con el de items
    user_data = pd.merge(df3, df, on=['item_name'], how='right')

    # Calcular la cantidad total de dinero gastado
    total_spent = user_data['price'].sum()
    total_spent = round(total_spent, 2)

    # Calcular la cantidad total de juegos comprados
    total_games = df.shape[0]

    # Calcular el porcentaje de recomendación (True vs False en 'recommend')
    total_recommendations = df2['recommend'].sum()  # Cuenta cuántos son True
    recommendation_percentage = (total_recommendations / total_games) * 100 if total_games > 0 else 0

    # Redondear el porcentaje a 2 decimales
    recommendation_percentage = round(recommendation_percentage, 2)

    # Devolver los resultados en un diccionario
    return {
        "user_id": user,
        "total_spent": f"${total_spent}",
        "recommendation_percentage": f"{recommendation_percentage}%",
        "total_games": total_games
    }

# Función para calcular el usuario que acumula más horas jugadas para el género dado y una lista de la acumulación de horas jugadas por año de lanzamiento.
def genre(gen, df, df2):

    # Unir el DataFrame de juegos con el de items
    df_merged = pd.merge(df, df2, on=['item_name', 'item_id'], how='inner')

    # Agrupar por usuario y sumar las horas jugadas
    df_grouped = df_merged.groupby('user_id').agg(total_playtime=('playtime_forever', 'sum'))

    # Ordenar por el total de horas jugadas en orden descendente
    df_grouped = df_grouped.sort_values(by='total_playtime', ascending=False).reset_index()

    # Guardar en una variable el usuario con más horas jugadas
    most_played_user = df_grouped['user_id'][0]  # Usuario con más horas jugadas

    # Filtrar el DataFrame para obtener solo los datos del usuario específico
    df_usuario = df_merged[df_merged['user_id'] == most_played_user]

    # Agrupar por año y sumar las horas jugadas del usuario
    df_grouped_usuario = df_usuario.groupby('year').agg(total_playtime=('playtime_forever', 'sum')).reset_index()

    return {
        'Usuario con más horas jugadas': most_played_user,
        'Horas jugadas por año': df_grouped_usuario.to_dict(orient='records')
    }

# Función para obtener el top 3 de desarrolladores con juegos más recomendados en un año dado
def best_developer(df, df2):

    # Unir los DataFrames de games y reviews en base al nombre del juego (app_name)
    merged_df = pd.merge(df2, df, on='item_id', how='inner')

    # Filtrar por recomendaciones positivas y comentarios positivos
    juegos_recomendados = merged_df[(merged_df['recommend'] == True) & (merged_df['sentiment_analysis'] == 2)]

    # Agrupar por desarrollador y contar las recomendaciones positivas
    desarrollador_recomendaciones = juegos_recomendados.groupby('developer').size().reset_index(name='num_recommendations')

    # Ordenar los desarrolladores por la cantidad de recomendaciones y seleccionar el top 3
    top_3_desarrolladores = desarrollador_recomendaciones.sort_values(by='num_recommendations', ascending=False).head(3)

    return {
        'Top 3 Desarrolladores': top_3_desarrolladores.to_dict(orient='records')
    }

# Función para devolver la cantidad de reseñas positivas y negativas de una desarrolladora
def developer_reviews(desarrolladora, df, df2):

    # Unir los DataFrames de juegos y reseñas en base al id del juego (item_id)
    merged_df = pd.merge(df2, df, on='item_id', how='inner')

    # Contar la cantidad de reseñas positivas y negativas
    reseñas_positivas = merged_df[merged_df['sentiment_analysis'] == 2].shape[0]
    reseñas_negativas = merged_df[merged_df['sentiment_analysis'] == 0].shape[0]

    # Devolver el resultado en formato diccionario
    resultado = {
        desarrolladora: {
            "Positive": reseñas_positivas,
            "Negative": reseñas_negativas
        }
    }

    return resultado

# MODELO DE APRENDIZAJE AUTOMÁTICO:

# Función para ajustar similitud basada en la diferencia de años
def year_penalty(year1, year2):
    return 1 - (abs(year1 - year2) / 10)  # Penalizamos si los años son lejanos

# Función para obtener recomendaciones
def get_recommendations(item_id, games_sample, num_recommendations=5):

    # Procesar las etiquetas de los juegos
    games_sample['tags_combined'] = games_sample['tags'].fillna('').apply(lambda x: ' '.join(x))

    # Crear un vectorizador TF-IDF basado en las etiquetas
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(games_sample['tags_combined'])

    # Similitud del coseno basada en etiquetas
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    try:
        # Obtener el índice del juego
        idx = games_sample[games_sample['item_id'] == item_id].index[0]
        
        # Obtener el año del juego
        target_year = games_sample.loc[idx, 'year']

        # Obtener las similitudes basadas en géneros
        sim_scores = list(enumerate(cosine_sim[idx]))

        # Ajustar las similitudes en base a la cercanía de los años
        for i, (index, score) in enumerate(sim_scores):
            game_year = games_sample.loc[index, 'year']
            sim_scores[i] = (index, score * year_penalty(target_year, game_year))

        # Ordenar por similitud y obtener los 5 juegos más similares
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations+1]

        # Obtener los índices de los juegos recomendados
        game_indices = [i[0] for i in sim_scores]
        
        # Devolver los juegos recomendados (IDs y nombres)
        return games_sample[['item_id', 'item_name']].iloc[game_indices]

    except IndexError:
        return "Juego no encontrado"


# Ruta para obtener la cantidad de items y el porcentaje de contenido gratuito según desarrollador.
@app.get("/desarrollador/{desarrollador}")
async def developer( desarrollador : str ):

    games_sample, _, _ = load_data()

    # Filtrar por el desarrollador proporcionado
    games_filtrado = games_sample[games_sample['developer'] == desarrollador]
    
    result = developer_items_free(games_filtrado)
    
    del games_filtrado
    gc.collect()

    return {"result": result.to_dict(orient='records')}

# Ruta para obtener la cantidad de dinero gastado por el usuario, el porcentaje de recomendación y cantidad de items.
@app.get("/user_id/{user_id}")
async def userdata( user_id : str ):

    games_sample, reviews_sample, items_sample = load_data()

    # Filtrar los juegos comprados por el usuario
    user_items_data = items_sample[items_sample['user_id'] == user_id]
    # Filtrar las recomendaciones hechas por el usuario
    user_reviews_data = reviews_sample[reviews_sample['user_id'] == user_id]

    result = user_statistics(user_id, user_items_data, user_reviews_data, games_sample)

    del user_items_data, user_reviews_data
    gc.collect()

    return result

# Ruta para calcular el usuario que acumula más horas jugadas para el género dado y una lista de la acumulación de horas jugadas por año de lanzamiento.
@app.get("/genero/{genero}")
async def userforgenre(genero: str):

    games_sample, items_sample, _ = load_data()

    # Filtrar los juegos que contengan el genero dado
    df_juegos = games_sample[games_sample['genres'].apply(lambda x: isinstance(x, list) and pd.notnull(x) and any(genero == g for g in x))]
    
    result = genre(genero, items_sample, df_juegos)

    del df_juegos
    gc.collect()

    return result

# Ruta para obtener el top 3 de desarrolladores con juegos más recomendados en un año dado.
@app.get("/anio/{anio}")
async def best_developer_year( anio : int ):

    games_sample, _, reviews_sample = load_data()

    # Filtrar los juegos lanzados en el año dado
    juegos_del_año = games_sample[games_sample['year'] == anio]

    result = best_developer(reviews_sample, juegos_del_año)

    del juegos_del_año
    gc.collect()

    return result

# Ruta para devolver la cantidad de reseñas positivas y negativas de una desarrolladora.
@app.get("/desarrolladora/{desarrolladora}")
async def developer_reviews_analysis( desarrolladora : str ):

    games_sample, _, reviews_sample = load_data()

    # Filtrar los juegos que fueron desarrollados por la desarrolladora dada
    juegos_de_desarrolladora = games_sample[games_sample['developer'] == desarrolladora]

    result = developer_reviews(desarrolladora, reviews_sample, juegos_de_desarrolladora)
    
    del juegos_de_desarrolladora
    gc.collect()

    return result

# Ruta para devolver una lista con 5 juegos recomendados similares al ingresado.
@app.get("/recomendacion_juego/{item_id}")
async def recomendacion_juego(item_id: int):

    games_sample, _, _ = load_data()

    try:
        recomendaciones = get_recommendations(item_id, games_sample)
        if isinstance(recomendaciones, pd.DataFrame):
            return {"juegos_recomendados": recomendaciones.to_dict(orient='records')}
        else:
            return {"error": "No se generaron recomendaciones válidas"}

    except IndexError:
        return {"error": "No se encontró el juego con ese ID."}
    
