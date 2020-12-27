import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
###### helper functions. Use them when needed #######
def get_title_from_index(index):
	return df[df.index == index]["name"].values[0]

def get_index_from_title(name):
	return df[df.name == name]["id"].values[0]
##################################################
##Content Based Recommender System
#Take something from the user and recommend based on that

##Step 1: Read CSV File
df = pd.read_csv("artists_and_tags_lessened_dataset.csv", encoding='latin-1')

#Reads the first rows of the dataset
#print(df.head())

#Reads features of the dataset
#print(df.columns)

##Step 2: Select Features
features = ["tagValue"] # Select features to take in acount


##Step 3: Create a column in DF which combines all selected features

# for feature in features:
# 	df[feature] = df[feature].fillna('') #fills NaN with ''

def combined_features(row):
	try:
		return row["tagValue"]
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

artist_user_likes = "Hocico"

## Step 6: Get id of this artist from its title

artist_index = get_index_from_title(artist_user_likes)

# Find Similar artists
# Converts matrix into a list and gives us inside a set of tuples
similar_artists = list(enumerate(sim_scores[artist_index]))

## Step 7: Get a list of similar artists in descending order of similarity score

# Sort similar
sorted_similar_artists = sorted(similar_artists, key= lambda x:x[1], reverse=True) #Key = decide the order, sort by x of 1 (Cosine Similarity), reverse= True gives us descending order

## Step 8: Print titles of first 50 artists

i= 0
for artist in sorted_similar_artists:
	print (get_title_from_index(artist[0]))
	i = i+1
	if i >50:
		break