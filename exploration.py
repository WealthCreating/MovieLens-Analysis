import pandas as pd
import numpy as np
import datetime
import re
import scipy
import matplotlib.pyplot as plt

def extract_year(title):
	if title.find('(') < 0:
		return int(title[:4])

	return extract_year(title[title.find('(')+1:])

movies = pd.read_csv('../Data/movies.csv')
ratings = pd.read_csv('../Data/ratings.csv')

ratings['rating'] *= 2
ratings['rating'] = ratings['rating'].astype(np.int8)
ratings['movieId'] = ratings['movieId'].astype(np.int32)
ratings['timestamp'] = ratings['timestamp'].astype(np.int32)
ratings = ratings[['rating', 'movieId', 'timestamp']]

# Average rating - NOTE add to jupyter
ratings_avg = ratings.groupby('movieId')[['movieId','rating']].mean().reset_index(drop=True)

# Get list of movie names to convert movieId
movie_dict = {movies['movieId'][i]:movies['title'][i] for i in range(len(movies))}

def find_movie_name(movie):
	return movie_dict[movie]

ratings_avg['movieId'] = list(map(find_year, ratings_avg['movieId']))
ratings_avg.columns = ['movie', 'average rating']

genre_groups = [movie.split('|') for movie in movies['genres']]

test_set = set()
for movie in genre_groups:
	for genre in movie:
		test_set.add(genre)

[[test_set.add(genre) for genre in movie] for movie in genre_groups]













movies_to_remove = [movies['movieId'][i] for i in range(len(movies)) if not ('18' in movies['title'][i] or '19' in movies['title'][i] or '20' in movies['title'][i])]
ratings = ratings.loc[~ratings['movieId'].isin(movies_to_remove)].reset_index(drop=True)
movies = movies.loc[~movies['movieId'].isin(movies_to_remove)].reset_index(drop=True)

movies_to_remove = movies.loc[movies['genres'] == '(no genres listed)']['movieId'].tolist()
ratings = ratings.loc[~ratings['movieId'].isin(movies_to_remove)].reset_index(drop=True)
movies = movies.loc[~movies['movieId'].isin(movies_to_remove)].reset_index(drop=True)


movies['title'] = movies['title'].str.strip()
# NOTE: clean age from movie title
# NOTE: email says 1960 and before, Raj said 1960 and after...
movies['year'] = [extract_year(movie) for movie in movies['title']]
movies['age'] = 'Old'
movies.loc[movies['year'] >= 1970, 'age'] = 'Medium'
movies.loc[movies['year'] >= 1990, 'age'] = 'New'




# ------------------------------------------------------------------------------
# All about rating counts below -- I'll come back to it

ratings_count = ratings.groupby('movieId').count()
movies_count = movies.groupby('movieId').count()
ratings_count.reset_index(inplace=True)
movies_count = movies_count.reset_index()[['movieId', 'title']]

print('Count of unique "movieId" in ratings: {}'.format(ratings_count.count().values[0]))
print('Count of unique "movieId" in movies: {}'.format(movies_count.count().values[0]))

'''
timestamps of reviews to analyze sentiment
1. fix original dataset to include timestamps
2. convert timestamps to datetime
3. only include reviews within a certain time after the movie was released
	- What is the average time difference between a review of a movie and when
		the movie was released?
	- Do I need to remove any movies that came out recently?
'''
# This is to find movies without genres and remove them from ratings/movies
ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')
new_movies = movies.loc[movies['year'] >= 1995]
movies = movies.loc[movies['year'] >= 1995]
ratings = ratings.loc[ratings['timestamp'] >= datetime.datetime(1995,1,1)]

# Ratings max is '2015-03-31' so we have account for that and remove 2015 movies
# 	because we don't know WHEN in 2015 those movies premiered.  For all we know,
#	those movies may have aired after the end of the ratings data

# Now let's find average time difference between movie and review
# 1. make a column for year delta
#		- how do I link year of movie release with movieId so that
#			ratings will include year of release?
#		- The # will be years after movie premiered

# dict of movieId and year

movie_dict = {movies['movieId'][i]:movies['year'][i] for i in range(len(movies))}

test_df = movies[:1000]
def find_movie_name(movie):
	return movie_dict[movie]

test_list = list(map(find_year, ratings['movieId']))

movies['movieId'][180:200]

# NOTE: convert ratings datetime into year only

# Question: how can I compare years?  How do I know what time of year the movie
# came out?  if a movie came out in december, and i'm analyzing reviews
ratings.loc[ratings['movieId'] == 223]
movies[200:240]

# Convert ratings so that all movies in it are from at least 1995
ratings_test = ratings.loc[ratings['timestamp'] >= datetime.datetime(1995,1,1)]
