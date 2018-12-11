import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
import math

movies = pd.read_csv('../Data/movies.csv')

# Removing movies that have no genre listed
movies = movies[movies['genres'] != '(no genres listed)']

# Add year released as a feature
movie_year = []
for movie in movies['title']:
	if '(199' in movie:
		pos = movie.find('(199') + 1
	elif '(200' in movie:
		pos = movie.find('(200') + 1
	elif '(201' in movie:
		pos = movie.find('(201') + 1
	else:
		movie_year.append(None)
		continue

	year = movie[pos:pos+4]

	if 1995 > int(year):
		movie_year.append(None)
		continue

	movie_year.append(year)

# only keep movies released after 1995
movies['year'] = movie_year
movies.dropna(inplace=True)

# only keep movies that have ratings
ratings = pd.read_csv('../Data/ratings.csv')[['rating', 'movieId']]
movies = movies[movies['movieId'].isin(set(ratings['movieId']))]

# Reduce ratings to only include movies of the clean movies dataframe
ratings = ratings[ratings['movieId'].isin(movies['movieId'])]

## Convert ratings to a 1-10 scale and update data types
ratings['rating'] *= 2
ratings['rating'] = ratings['rating'].astype(np.int8)
ratings['movieId'] = ratings['movieId'].astype(np.int32)

# before
len(movies) # 27278
len(ratings) # 20000263
# after
len(movies) # 15033
len(ratings) # 10653093

ratings_avg = ratings.groupby('movieId')[['movieId','rating']].mean().reset_index(drop=True)
ratings_cnt = ratings.groupby('movieId').count()

ratings_cnt.describe()
# rating
# count	15033.000000
# mean	708.647176
# std	2814.624231
# min	1.000000
# 25%	3.000000
# 50%	18.000000
# 75%	215.000000
# max	53769.000000
len(ratings_cnt[ratings_cnt['rating'] < 5]) / len(ratings_cnt) # 29.8% movies have less than 5 ratings
len(ratings_cnt[ratings_cnt['rating'] == 1]) # 2054 movies have 1 rating

movies.head()
ratings_avg

ratings_dict = {ratings_avg['movieId'][i]:ratings_avg['rating'][i] for i in range(len(ratings_avg))}

def find_movie_rating(movie):
    return ratings_dict[movie]

movies['average rating'] = list(map(find_movie_rating, movies['movieId']))

# Sorting by average value
movies.sort_values('average rating', inplace=True)
movies = movies.reset_index(drop=True)

print('Average movie rating:  {0:.2f}'.format(movies['average rating'].mean()))
print('Median movie rating:   {0:.2f}'.format(movies['average rating'].median()))

# Since our ratings are sorted, find how many records are in every 10%
movies_10_pct = int(len(movies)/10)

# Column of movie popularity
movies['popularity'] = None
bottom_20 = movies['popularity'].iloc[:movies_10_pct*2] = 1 # 'Worst'
middle_20 = movies['popularity'].iloc[movies_10_pct*4:movies_10_pct*6] = 2 # 'OK'
top_20 = movies['popularity'].iloc[movies_10_pct*8:] = 3 # 'Best'

# Remove movies that are not in top, middle, or bottom 20%
movies = movies.dropna().reset_index(drop=True)

# Add year of movie
movies['title'] = movies['title'].str.strip()
def extract_year(title):
    if title.find('(') < 0:
        return int(title[:4])

    return extract_year(title[title.find('(')+1:])

# Column of movie age
movies['year'] = [extract_year(movie) for movie in movies['title']]

movies['age'] = None
movies.loc[movies['year'] >= 1960, 'age'] = 1 # 'Old'
movies.loc[movies['year'] >= 1970, 'age'] = 2 # 'Medium'
movies.loc[movies['year'] >= 1990, 'age'] = 3 # 'New'

movies = movies.dropna().reset_index(drop=True)

# Make a column for each genre
genre_groups = [movie.split('|') for movie in movies['genres']]
genre_set = set()
[[genre_set.add(genre) for genre in movie] for movie in genre_groups]
genre_set = list(genre_set)

df = pd.DataFrame(columns=[genre_set])

df['average rating'] = movies['average rating']
df['popularity'] = movies['popularity']
df['age'] = movies['age']

df.fillna(0, inplace=True)

# populate dataframe
for i in range(len(genre_groups)):
    for genres in genre_groups[i]:
        df.loc[i, genres] = 1
