# Proyecto Individual 1: Machine Learning Operations.
## Introducción:
Este proyecto tiene como objetivo desarrollar un sistema de recomendación para STEAM (Plataforma de videojuegos), responder cinco consultas relacionadas con los datos y posteriormente implementar todo lo anterior mediante una API,  haciendo uso de Render para su accesibilidad. Los Dataframes entregados deben ser tratados para poder ser analizados y extraer la información necesaria para el modelo a desarrollar. 

## Consideraciones Iniciales:
Contamos con una base de datos compuesta por tres archivos JSON (steam_games, user_reviews y user_items) que deben ser procesados para su uso.

A continuación, se hace un resumen del proceso completo:

## Extracción, Carga y Transformación de los Datos (ETL):
Se tuvo que hacer un proceso de carga de datos y desanidación, así como también la eliminación de columnas que no aportaban valor y traían complejidad al proyecto. Además, se hizo un manejo de valores nulos, faltantes, duplicados y cambios de tipo de dato. Fue necesario también hacer un análisis de sentimiento para posteriores análisis. Todo esto fue almacenado en archivos tipo .parquet, que será la nueva fuente de los datos solicitados por las funciones (consultas).

## Desarrollo API:
Se disponibilizan los datos de la empresa usando el framework FastAPI, proponiendo las siguientes consultas:
* def developer( desarrollador : str ): Cantidad de items y porcentaje de contenido Free por año según empresa desarrolladora.
* def userdata( User_id : str ): Debe devolver cantidad de dinero gastado por el usuario, el porcentaje de recomendación en base a reviews.recommend y cantidad de items.
* def UserForGenre( genero : str ): Debe devolver el usuario que acumula más horas jugadas para el género dado y una lista de la acumulación de horas jugadas por año de lanzamiento.
* def best_developer_year( año : int ): Devuelve el top 3 de desarrolladores con juegos MÁS recomendados por usuarios para el año dado. (reviews.recommend = True y comentarios positivos)
* def developer_reviews_analysis( desarrolladora : str ): Según el desarrollador, se devuelve un diccionario con el nombre del desarrollador como llave y una lista con la cantidad total de registros de reseñas de usuarios que se encuentren categorizados con un análisis de sentimiento como valor positivo o negativo.

## Deployment:
Se usa el servicio de Render que permite que la API pueda ser consumida desde la web.

## Análisis Exploratorio de Datos (EDA):
Se realizó una serie de pasos para comprender mejor la información contenida en los datasets y poder extraer algunas primeras aproximaciones y conclusiones.

## Aprendizaje Automático(Machine Learning):
Se desarrolló un modelo de recomendación item-item en un entorno de aprendizaje automático. Se calculó una matriz de similitud del coseno para determinar la similitud entre juegos basada en características como la fecha de lanzamiento y los tags. Se implementó una función 'get_recommendations' que toma el ID de un juego y devuelve una lista de 5 juegos recomendados similares al juego ingresado.

## Autora
Sarita Vallejo Ramírez
