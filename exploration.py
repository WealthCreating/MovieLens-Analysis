import pandas as pd
import numpy as np
import datetime
import re
import scipy
import matplotlib.pyplot as plt

movies = pd.read_csv('../Data/movies.csv')
ratings = pd.read_csv('../Data/ratings.csv')

print('Ratings:')
ratings.head()

print('\n\nMovies:')
movies.head()

print('Ratings: original data')
print(ratings.info(memory_usage='deep'))

#-------------------------------------------------------------------------------
## Convert ratings to a 1-10 scale and finish cleaning table
ratings['rating'] *= 2
ratings['rating'] = ratings['rating'].astype(np.int8)
ratings['movieId'] = ratings['movieId'].astype(np.int32)
ratings = ratings[['rating', 'movieId']]

print('Ratings: optimized data')
print(ratings.info(memory_usage='deep'))

#-------------------------------------------------------------------------------
# NOTE: I understand that this runs the risk of wrongly adding a movie and might
# have outliers.  I have not been able to find an exception

# List of movies that don't have a year in their  name
movies_to_remove = [movies['movieId'][i]
					for i in range(len(movies))
					if not ('18' in movies['title'][i]
					or '19' in movies['title'][i]
					or '20' in movies['title'][i])]

# Remove said movies from our datasets
ratings = ratings.loc[~ratings['movieId'].isin(movies_to_remove)].reset_index(drop=True)
movies = movies.loc[~movies['movieId'].isin(movies_to_remove)].reset_index(drop=True)


# Extracting the year from the title
movies['title'] = movies['title'].str.strip()


def extract_year(title):
	if title.find('(') < 0:
		return int(title[:4])

	return extract_year(title[title.find('(')+1:])


movies['year'] = [extract_year(movie) for movie in movies['title']]

movies['age'] = 'Old'
movies.loc[movies['year'] >= 1970, 'age'] = 'Medium'
movies.loc[movies['year'] >= 1990, 'age'] = 'New'

# Two ways to find top, middle, and bottom 20%
# 1. Use z-scores and standard deviation
# 2. Order by rating, find 20% of the total record count
movies.head()

# Wait a second.... are there movies that are in our ratings dataframe that
# are not in the movies dataframe, or vise versa?
ratings_count = ratings.groupby('movieId').count()
movies_count = movies.groupby('movieId').count()
ratings_count.reset_index(inplace=True)
movies_count = movies_count.reset_index()[['movieId', 'title']]

most_rated_movie_id = ratings_count.loc[ratings_count['rating'] == ratings_count['rating'].max()]['movieId'].values[0]
movies.loc[movies['movieId'] == most_rated_movie_id]


# It would be interesting to see if the movies with fewer ratings scew
# 	predictions in a way.  For example there are ~26,000 different movies, and over
# 	60% of movies have less than 50 reviews, even though the average number of
#	reviews for a movie is over 700.  Which means we'll have a high sdev.
print('# of reviews:  {}'.format(ratings_count.count()[0]))
print('# of movies with less than 50 reviews:  {}'.format(ratings_count.loc[ratings_count['rating'] < 50].count()[0]))
print('Average number of ratings for a movie:  {}'.format(int(ratings_count['rating'].mean())))
print('Standard deviation of ratings:  {}'.format(int(ratings_count['rating'].std())))

# large sdev compared to the mean means that the majority of the
# distribution of the review count will around the mean, with a small
# number of movies having a significantly larger review count.


plt.figure(figsize=(13, 5))
plt.hist(ratings_count['rating'], bins=20)
plt.title('Count of ratings')
plt.show()


print('Count of unique "movieId" in ratings: {}'.format(ratings_count.count().values[0]))
print('Count of unique "movieId" in movies: {}'.format(movies_count.count().values[0]))
# What we're going to do: compare the movieId's in both dataframes and see what
# we find






# NOTE: from graph above, and looking at list of ratings_count,
# Are there two different datasets?


# Question: Can we use the count of reviews for a movie to predict ratings?
# Theory: movies with more reviews are higher rated.  I have this theory
# because people are more likely to watch a movie if a friend recommended it,
# and less likely to watch a movie if a friend didn't like it.  Also, not always
# (but generally), movies with a larger budget and top producers/actors will
# have a better quality than a movie created by aspiring producers/actors.

# Therefore,




ratings_count.loc[ratings_count['rating'] > 50]

ratings_count.count()

ratings_count.mean()
'''
