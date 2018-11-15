import pandas as pd
import numpy as np
import datetime
import re
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import math

from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

movies = pd.read_csv('../Data/movies.csv')
ratings = pd.read_csv('../Data/ratings.csv')

ratings['rating'] *= 2
ratings['rating'] = ratings['rating'].astype(np.int8)
ratings['movieId'] = ratings['movieId'].astype(np.int32)
ratings['timestamp'] = ratings['timestamp'].astype(np.int32)
ratings = ratings[['rating', 'movieId', 'timestamp']]


# Removing movies that have no genre listed
no_genre = movies.loc[movies['genres'] == '(no genres listed)']['movieId'].tolist()

# Only taking movies that have 18, 19, or 20
year_in_title = [movies['movieId'][i] for i in range(len(movies)) if not ('18' in movies['title'][i] or '19' in movies['title'][i] or '20' in movies['title'][i])]

movies_to_remove = no_genre + year_in_title
ratings = ratings.loc[~ratings['movieId'].isin(movies_to_remove)].reset_index(drop=True)

# Removing movies that are in movies but not in ratings
movie_set = set(movies['movieId'])
ratings_set = set(ratings['movieId'])
missing_movies = list(movie_set - ratings_set)
movies = movies.loc[~movies['movieId'].isin(missing_movies)].reset_index(drop=True)

# ------------------------------------------------------------------------------
# Average rating - NOTE add to jupyter
ratings_avg = ratings.groupby('movieId')[['movieId','rating']].mean().reset_index(drop=True)

# Add ratings avg to movies
ratings_dict = {ratings_avg['movieId'][i]:ratings_avg['rating'][i] for i in range(len(ratings_avg))}
def find_movie_rating(movie):
	return ratings_dict[movie]

movies['average rating'] = list(map(find_movie_rating, movies['movieId']))

# Sorting by average value
movies.sort_values('average rating', inplace=True)
movies = movies.reset_index(drop=True)

# Median rating
movies['average rating'].median()

# ------------------------------------------------------------------------------
# Cleaning for ML

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
# NOTE: email says 1960 and before, Raj said 1960 and after...
movies['title'] = movies['title'].str.strip()
def extract_year(title):
	if title.find('(') < 0:
		return int(title[:4])

	return extract_year(title[title.find('(')+1:])

# Column of movie age
movies['year'] = [extract_year(movie) for movie in movies['title']]
movies['age'] = 1 # 'Old'
movies.loc[movies['year'] >= 1970, 'age'] = 2 # 'Medium'
movies.loc[movies['year'] >= 1990, 'age'] = 3 # 'New'

# columns of genre groups
genre_groups = [movie.split('|') for movie in movies['genres']]
genre_set = set()
[[genre_set.add(genre) for genre in movie] for movie in genre_groups]
genre_set = list(genre_set)

df = pd.DataFrame(columns=[genre_set])

df['average rating'] = movies['average rating']
df['popularity'] = movies['popularity']
df['age'] = movies['age']

df.fillna(0, inplace=True)

for i in range(len(genre_groups)):
	for genres in genre_groups[i]:
		df.loc[i, genres] = 1

# ------------------------------------------------------------------------------
# ML

x1 = df[genre_set]
x2 = pd.get_dummies(df['age'])
x3 = pd.concat([x1, x2], axis=1)
y = df['popularity']

logr = LogisticRegression()
bnb = BernoulliNB()
tree = RandomForestClassifier()

# Collection of parameters to test
logr_param = {'tol': [math.exp(-5), math.exp(-4), math.exp(-3)]}

bnb_param = {'alpha': [0.01, 0.5, 1, 2]}

tree_param = {'n_estimators': [10, 100, 200],
              'n_jobs': [-1],
              'max_depth': [2, 5, 10]}


# Determine the best parameters
logr_best = GridSearchCV(estimator=logr, param_grid=logr_param, cv=5)
bnb_best = GridSearchCV(estimator=bnb, param_grid=bnb_param, cv=5)
tree_best = GridSearchCV(estimator=tree, param_grid=tree_param, cv=5)

classifiers = [logr_best, bnb_best, tree_best]
predictors = [x1, x2, x3]
input_columns = ['Genres', 'Age', 'Genres and Age']
classifier_name = ['Logistic Regression', 'Bernoulli Naive Bayes', 'Random Forest']

print('Predicting movie popularity')
for a in range(3):
	print('\n\nInput Columns: {}\n'.format(input_columns[a]))
	x = predictors[a]

	for i in range(3):
		classifier = classifiers[i]
		classifier.fit(x, y)
		y_pred = classifier.predict(x)
		print('{}   -    accuracy: {}'.format(classifier_name[i], accuracy_score(y_pred, y)))

print('Predicting movie rating')
y = df['average rating']
for a in range(3):
	print('\n\nInput Columns: {}\n'.format(input_columns[a]))
	x = predictors[a]

	for i in range(3):
		classifier = classifiers[i]
		classifier.fit(x, y)
		y_pred = classifier.predict(x)
		print('{}   -    accuracy: {}'.format(classifier_name[i], accuracy_score(y_pred, y)))






for a in range(3):
	print('Input Columns: {}\n'.format(input_columns[a]))
	x = predictors[a]

	for i in range(3):
	    classifier = classifiers[i]
	    classifier.fit(x, y)
	    y_pred = classifier.predict(x)
	    matrix = confusion_matrix(y, y_pred)

	    print(classifier_name[i])
	    print('\nTrue positive: ', matrix[0][0])
	    print('False positive: ', matrix[0][1])
	    print('True Negative: ', matrix[1][1])
	    print('False Negative: ', matrix[1][0])
	    print('-------------------------------------------------------')

	print('\n\n\n')




















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
