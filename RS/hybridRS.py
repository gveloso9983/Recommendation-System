import pandas as pd
import numpy as np
from time import sleep
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#Read CSV File
#df = pd.read_csv("movies_and_ratings_small_dataset.csv")
cf = pd.read_csv("movies_and_ratings_small_dataset.csv")
df = cf.drop_duplicates(subset = ["title"], keep = 'first', ignore_index = True)

###### helper functions. Use them when needed #######
def get_title_from_index(index):
	return df[df.index == index]["title"].values[0]

def get_index_from_title(title):
	return df[df.title == title].index[0]

def get_director_from_title(title):
	return df[df.title == title]["director"].values[0]
##################################################

###### Interface functions #######
#Welcome message
def welcome():
    print('##########################################')
    print('#          Movies Recomendation          #')
    print('##########################################')
    print('#           Welcome to the best          #')
    print('#    recomendation platform ever made!   #')
    print('##########################################')
    input('#              Press Enter to continue...')
    print('\n \n')

#Login
def login():
    print('##########################################')
    print('#                  Login                 #')
    print('##########################################')
    id = int(input('# Insert an userId please: '))
    print('\n \n')
    return id

#Menu
def menu():
    print('##########################################')
    print('#                  Menu                  #')
    print('##########################################')
    print('# 0 - Top 10 movies                      #')
    print('# 1 - Content based                      #') # recommendatin based on a movie
    print('# 2 - Colaborative based                 #') # recommendation based on your likings
    print('# 3 - RMSE                               #') # rmse
    print('# 4 - Logout                             #')
    print('# 5 - Close                              #')
    option = int(input('# Please select one option: '))
    print('\n \n')
    return option

###### APP functionalities #######
#Create top10
def top10():
    no_dups = df.drop_duplicates("title", keep="first")
    df_sorted = no_dups.sort_values("vote_average", ignore_index=True, ascending = False)
    print("Top 10 movies: ")
    print(df_sorted[["title","vote_average"]].head(10))
    print('\n \n')
    sleep(5)
    return True

def movie_likes():
    
    print('movies_likes')

    features = ["keywords", "cast", "genres", "director"]
    for feature in features:
        df[feature] = df[feature].fillna('')
    
    def combine_features(row):
        try:
            return row["keywords"] + " " + row["cast"] + " " + row["genres"] + " " + row["director"]
        except:
            print( "Error:" , row)
    
    df["combined_features"]= df.apply(combine_features, axis = 1) #axis=1 passes as rows and not columns
    #print("Combined features \n ", df["combined_features"].head())
    ##Step 4: Create count matrix from this new combined column
    cv = CountVectorizer()
    #counts the frequecy of the words
    count_matrix = cv.fit_transform(df["combined_features"])
    ##Step 5: Compute the Cosine Similarity based on the count_matrix
    #calculates similarity between points
    sim_scores = cosine_similarity(count_matrix)
    #print("Cosine Similarity : \n")
    #print(sim_scores)
    #print("\n")
    titles = sorted(cf['title'].unique().tolist())
    print('##########################################')
    print('#        Select what film you like       #')
    print('##########################################')
    i = 0
    for t in titles:
        print('# ' + repr(i) + ' - ' + t)
        i += 1
    print('##########################################')

    movie_user_likes_index = int(input('# Select the movie index: '))
    movie_user_likes = titles[movie_user_likes_index]
    ## Step 6: Get id of this movie from its title
    movie_index = get_index_from_title(movie_user_likes)
    movie_director = get_director_from_title(movie_user_likes)
    print("Because you like the director " + movie_director + " : \n")
    # Find Similar movies
    # Converts matrix into a list and gives us inside a set of tuples
    similar_movies = list(enumerate(sim_scores[movie_index]))
    ## Step 7: Get a list of similar movies in descending order of similarity score
    # Sort similar
    print(sim_scores[movie_index])
    sorted_similar_movies = sorted(similar_movies, key=lambda x:x[1], reverse=True) #Key = decide the order, sort by x of 1 (Cosine Similarity), reverse= True gives us descending order
    ## Step 8: Print titles of first 50 artists
    print("Recommended movies: \n")
    i = 0
    for movie in sorted_similar_movies:
        print (get_title_from_index(movie[0]))
        i = i+1
        if i >10:
            break
    return True

#Collaborative Based
def colaborative_based():
    print('colaborative_based')
    user_ratings = cf.pivot_table(index=['userId'],columns=['title'], values='rating')
    #Remove movies who have less than 20 users who rated it
    user_ratings = user_ratings.dropna(thresh=20, axis=1).fillna(0)
    #print("Pivoted table")
    #print(user_ratings.head())
    item_similarity_df = user_ratings.corr(method='pearson')

    rows = cf.loc[cf['userId'] == userId]

    action_lover = [tuple(l) for l in rows[['title', 'rating']].values.tolist()]

    out_tup = [i for i in action_lover if i[0] in user_ratings]

    action_lover = out_tup

    
    def get_similar_movies(movie_name, user_rating):
        similar_score = item_similarity_df[movie_name]*(user_rating-2.5) #subtracts the rating by the mean in order to correct the low values
        similar_score = similar_score.sort_values(ascending=False)
        return similar_score
    
    similar_movies = pd.DataFrame()
    for movie,rating in action_lover:
            similar_movies = similar_movies.append(get_similar_movies(movie, rating), ignore_index=True)
    print(similar_movies.sum().sort_values(ascending=False).head(10))
    return True

def rmse():
    train, test = train_test_split(cf, test_size=0.3, random_state=42)

    features = ["keywords", "cast", "genres", "director"]
    for feature in features:
        train[feature] = train[feature].fillna('')
        test[feature] = test[feature].fillna('')
    
    def combine_features(row):
        try:
            return row["keywords"] + " " + row["cast"] + " " + row["genres"] + " " + row["director"]
        except:
            print( "Error:" , row)
    
    train["combined_features"] = train.apply(combine_features, axis = 1) #axis=1 passes as rows and not columns
    test["combined_features"] = test.apply(combine_features, axis = 1) #axis=1 passes as rows and not columns

    ##Step 4: Create count matrix from this new combined column
    cv = CountVectorizer()
    #counts the frequecy of the words
    count_matrix_train = cv.fit_transform(train["combined_features"])
    count_matrix_test = cv.transform(test["combined_features"])
    ##Step 5: Compute the Cosine Similarity based on the count_matrix
    #calculates similarity between points
    sim_scores_train = cosine_similarity(count_matrix_train)
    sim_scores_test = cosine_similarity(count_matrix_test)

    predict = count_matrix_test.dot(sim_scores_test) / np.array([np.abs(sim_scores_test).sum(axis=1)])
    print(predict)

    rmse = mean_squared_error(sim_scores_train, predict)
    print(rmse)
    return True

def close():
    return False
##################################################

#ratings_train = pd.read_csv('', sep='\t', names=r_cols, encoding='latin-1')
#ratings_test = pd.read_csv('', sep='\t', names=r_cols, encoding='latin-1')


welcome()
userId = login()
options = { 
    0: top10,
    1: movie_likes,
    2: colaborative_based,
    3: rmse,
}
flag = True
while flag:
    option = menu()
    if option == 4:
        userId = login()
    else:
        func = options.get(option, close)
        flag = func()