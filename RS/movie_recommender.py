import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
###### helper functions. Use them when needed #######
def get_title_from_index(index):
	return df[df.index == index]["title"].values[0]

def get_index_from_title(title):
	return df[df.title == title]["index"].values[0]
##################################################
##Content Based Recommender System
#Take something from the user and recommend based on that

##Step 1: Read CSV File
df = pd.read_csv("movie_dataset.csv")

#Reads the first rows of the dataset
#print(df.head())

#Reads features of the dataset
#print(df.columns)

##Step 2: Select Features
features = ["keywords","cast","genres","director"]


##Step 3: Create a column in DF which combines all selected features

for feature in features:
	df[feature] = df[feature].fillna('') #fills NaN with ''

def combined_features(row):
	try:
		return row["keywords"] + " " +row["cast"] + " " +row["genres"]+ " " +row["director"]
	except:
		print( "Error:" , row)
df["combined_features"] = df.apply(combined_features, axis=1) #axis=1 passes as rows and not columns

print("Combined features \n", df["combined_features"].head())


##Step 4: Create count matrix from this new combined column

cv = CountVectorizer()
#counts the frequecny of the words
count_matrix = cv.fit_transform(df["combined_features"])

##Step 5: Compute the Cosine Similarity based on the count_matrix

#calculates similarity between points
sim_scores = cosine_similarity(count_matrix)

print(sim_scores)

movie_user_likes = "Pulp Fiction"

## Step 6: Get index of this movie from its title

movie_index = get_index_from_title(movie_user_likes)

# Find Similar Movies
# Converts matrix into a list and gives us inside a set of tuples
similar_movies = list(enumerate(sim_scores[movie_index]))

## Step 7: Get a list of similar movies in descending order of similarity score

# Sort similar
sorted_similar_movies = sorted(similar_movies, key= lambda x:x[1], reverse=True) #Key = decide the order, sort by x of 1 (Cosine Similarity), reverse= True gives us descending order

## Step 8: Print titles of first 50 movies

i= 0
for movie in sorted_similar_movies:
	print (get_title_from_index(movie[0]))
	i = i+1
	if i >50:
		break