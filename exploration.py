import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
import math

movies = pd.read_csv('../Data/movies.csv').sort_values('movieId')

# Removing movies that have no genre listed
movies = movies[movies['genres'] != '(no genres listed)']

# Add year released as a feature
movie_age = []
for movie in movies['title']:
	if '(199' in movie:
		pos = movie.find('(199') + 1
	elif '(200' in movie:
		pos = movie.find('(200') + 1
	elif '(201' in movie:
		pos = movie.find('(201') + 1
	else:
		movie_age.append(None)
		continue

	year = movie[pos:pos+4]

	if 1995 > int(year):
		movie_age.append(None)
		continue

	movie_age.append(2018 - int(year))

# only keep movies released after 1995
movies['age'] = movie_age
movies.dropna(inplace=True)

# only keep movies that have ratings
ratings = pd.read_csv('../Data/ratings.csv')[['rating', 'movieId']].sort_values('movieId')
movies = movies[movies['movieId'].isin(set(ratings['movieId']))]

# Reduce ratings to only include movies of the clean movies dataframe
ratings = ratings[ratings['movieId'].isin(movies['movieId'])]

## Convert ratings to a 1-10 scale and update data types
ratings['rating'] *= 2
ratings['rating'] = ratings['rating'].astype(np.int8)
ratings['movieId'] = ratings['movieId'].astype(np.int32)

# # before
# len(movies) # 27278
# len(ratings) # 20000263
# # after
# len(movies) # 15033
# len(ratings) # 10653093

movies['rating average'] = ratings.groupby('movieId')['rating'].mean().tolist()
movies['rating count'] = ratings.groupby('movieId')['rating'].count().tolist()
movies['ratings per year'] = movies['rating count'] / movies['age']

# movies['rating count'].describe()
# # count	15033.000000
# # mean	708.647176
# # std	2814.624231
# # min	1.000000
# # 25%	3.000000
# # 50%	18.000000
# # 75%	215.000000
# # max	53769.000000
# len(ratings_cnt[ratings_cnt['rating'] < 5]) / len(ratings_cnt) # 29.8% movies have less than 5 ratings
# len(ratings_cnt[ratings_cnt['rating'] == 1]) # 2054 movies have 1 rating


# Make a column for each genre
genre_groups = [movie.split('|') for movie in movies['genres']]

genre_set = set()
[[genre_set.add(genre) for genre in movie] for movie in genre_groups]
genre_set = list(genre_set)


for genre in genre_set:
	movies[genre] = 0

# populate genres
for i, genre in enumerate(genre_groups):
	movies.loc[i, genre] = 1

movies.dropna(inplace=True)
movies['rating average'] = movies['rating average'].astype(np.int8)

y = movies['rating average']
x = movies.drop(['rating average','movieId','title','genres'], axis=1)
x = movies[genre_set]
