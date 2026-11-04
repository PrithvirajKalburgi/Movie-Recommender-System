import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt
#from pathlib import Path
ratings = pd.read_csv(
    "data/ml-1m/ratings.dat",
    sep="::",
    engine="python",
    names=["user_id", "movie_id", "rating", "timestamp"]
)

print(ratings.head())
print("Total ratings:", len(ratings))

movies = pd.read_csv(
    'data/ml-1m/movies.dat',
    sep="::",
    engine="python",
    names=["movie_id", "title", "genres"],encoding="latin-1"
)

print(movies.head())
print("Total ratings:", len(movies))