import pandas as pd
import numpy as np
import datetime
import re

movies = pd.read_csv('../Data/movies.csv')
ratings = pd.read_csv('../Data/ratings.csv')

print('Ratings:')
ratings.head()

print('\n\nMovies:')
movies.head()

print('Ratings: original data')
print(ratings.info(memory_usage='deep'))

# Convert ratings to a 1-10 scale and finish cleaning table
ratings['rating'] *= 2
ratings['rating'] = ratings['rating'].astype(np.int8)
ratings['movieId'] = ratings['movieId'].astype(np.int32)
ratings = ratings[['rating', 'movieId']]

print('Ratings: optimized data')
print(ratings.info(memory_usage='deep'))

movies.head()

for movie in movies['title']:
	try:
		a = re.findall('(\d+)', movie)[0]
	except:
		print(movie)

movies_to_remove = {movies['movieId'][i]:movies['title'][i]
					for i in range(len(movies))
					if len(re.findall('(\d+)', movies['title'][i])) == 0
					or len(re.findall('(\d+)', movies['title'][i])) != 4}

print(movies_to_remove)

for i in range(100):
	test_age = re.findall('(\d+)', movies['title'][i])
	print(test_age)



ratings = ratings.loc[~ratings['movieId'].isin(movies_to_remove.keys())]
movies = movies.loc[~ratings['movieId'].isin(movies_to_remove.keys())]


movies['release_year'] = None
for i in range(len(movies)):
	movies['release_year'][i] = int(re.findall('(\d+)',movies['title'][i])[0])


movies['release_year'] = [int(re.findall('(\d+)', i)[0] for i in movies['title']]

for movie in movies['title']:
	if not ('18' in movie or '19' in movie or '20' in movie):
		print(movie)








test = [int(re.findall('(\d+)', i)[0] for i in movies['title']]



test = []
test_df = movies[:100]
for i in range(len(test_df['title'])):
	try:
		test_age = re.findall('(\d+)', test_df['title'][i])[0]

		if len(test_age) != 4:
			print(test_age)
	except:
		print(test_df['title'][i])


test
movie_ages = {
'New':1990,
'Medium': 1970}





movies['movie_age'] = None



test = movies['title'][10]
int(re.findall('(\d+)',test)[0])



movies['movie_age'] = ['New'
					   for i in range(len(ratings))
					   if movies[]]


movie_ages.values()
