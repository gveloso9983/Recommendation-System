import pandas as pd

ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv')
movie_ratings = pd.merge(movies, ratings).drop(['genres','timestamp'],axis=1)

user_ratings =  movie_ratings.pivot_table(index=['userId'],columns=['title'], values='rating')


#Remove movies who have less than 10 users who rated it
user_ratings = user_ratings.dropna(thresh=15, axis=1).fillna(0)

# print(user_ratings.head())

#Similarity Matrix
item_similarity_df = user_ratings.corr(method='pearson')
#print(item_similarity_df.head(10))

#Make recommendation

def get_similar_movies(movie_name, user_rating):
    similar_score = item_similarity_df[movie_name]*(user_rating-2.5) #subtracts the rating by the mean in order to correct the low values
    similar_score = similar_score.sort_values(ascending=False)

    return similar_score

action_lover = [("(500) Days of Summer (2009)",5), 
                ("Zodiac (2007)",4), 
                ("2012 (2009)",3)]

similar_movies = pd.DataFrame()

for movie,rating in action_lover:
        similar_movies = similar_movies.append(get_similar_movies(movie, rating), ignore_index=True)

print(similar_movies.sum().sort_values(ascending=False))
