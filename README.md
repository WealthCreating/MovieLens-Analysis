## MovieLens Analysis
### By: Carter Carlson

### Introduction

This [dataset](http://files.grouplens.org/datasets/movielens/ml-20m-README.html) contains 20 million 5-start ratings applied to 27,000 movies by 138,000
users.  Data was created from users between January 09, 1995 and March 31, 2015.

---

### Analysis Details

* Load in CSV's and verify/optimize column datatypes.
* Determine if there's a relationship between movie genres, release date, and ratings.
* Use ML techniques to predict the average rating of a movie.

---

### Techniques Used

* Data aggregation/cleansing
* Data visualization
* Data standardization
* Machine Learning analysis
* Feature engineering
* Feature importance

---

### Challenges

While we have twenty years of review data, some of the reviews are for movies that
were released before review data was collected.  The reviews of older movies may
have a bias, as all of their reviews are several years or decades after the movie
release.  To that aspect, I will be removing all movies released before 1995, which
is when our first reviews are documented.
