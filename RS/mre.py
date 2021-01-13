import pandas as pd
import numpy as np
from time import sleep
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from pathlib import Path

cf = pd.read_csv("movies_and_ratings_small_dataset.csv")

train, test = train_test_split(cf, test_size=0.2, random_state=42)

n_items = cf.index.unique().shape[0]