import pandas as pd
ratings = pd.read_csv(
    "data/ml-1m/ratings.dat",
    sep="::",
    engine="python",
    names=["UserID", "MovieID", "Rating", "Timestamp"]
)

print(ratings.head())
print("Total ratings:", len(ratings))

movies = ratings = pd.read_csv(
    "data/ml-1m/movies.dat",
    sep="::",
    engine="python",
    names=["UserID", "MovieID", "Rating", "Timestamp"]
)
print(movies.head())
print("Total ratings:", len(ratings))