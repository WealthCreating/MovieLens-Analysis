## MovieLens Analysis
### By: Carter Carlson


### Instructions

1. Pre-process the data to center around movies:
   - Compute the average rating for each movie
   -  Normalize the Year of Movie Release so that 1990-later is "New", 1970s/1980s is "Medium" and anything in 1960s or before is "Old"
   - Assign genres to each movie.
   - Ignore "links", "tags" and "tags genome" for this project

2. Sort the movies by their average rating. Find the median of average rating. Pick top 20%, bottom 20% and middle 20% of the movies. You can call them "Best", "Worst and "OK" respectively.

3. Run a ML algorithm to show if/how a movie's Genre and/or Age determines its rating ("Best", "Worst", "OK").


### Analysis Process
* Load in CSV's and verify/optimize column datatypes
* Create a 'movie_age' column from movie release date
* Strip movie titles of release date
* Calculate average rating of each movie
   * Group ratings by 'movieId'
   * Dictionary from movies with 'movieId' as key, 'title' as value
   * Map dictionary to grouped ratings
   * Sort
* Calculate median rating
* Calculate standard deviation
   * Use z-scores to determine movies in the top 20%, bottom 20%, and middle 20% by ratings
   * Create a 'average_rating' column 
   * Create a new DataFrame from selected movies
* Split genre column into individual genre columns
* Use Logistic Regression, Naive Bayes, and Decision Tree to generate classification reports
* Bonus: Decision Tree correlation heatmap of genres
