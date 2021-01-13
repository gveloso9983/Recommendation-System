import pandas as pd
import numpy as np
from time import sleep
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

###### helper functions. Use them when needed #######
def get_title_from_index(index):
	return df[df.index == index]["title"].values[0]

def get_index_from_title(title):
	return df[df.title == title]["index"].values[0]


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
    input('# Insert an userId please: ')
    print('\n \n')

#Menu
def menu():
    print('##########################################')
    print('#                  Menu                  #')
    print('##########################################')
    print('# 0 - Top 10 movies                      #')
    print('# 1 - Content based                      #') # recommendatin based on a movie
    print('# 2 - Movies you like                    #') #n sei se e necessario
    print('# 3 - Colaborative based                 #') # recommendation based on your likings
    print('# 4 - Welcome                            #') # Adicionar rating
    print('# 5 - Logout                             #')
    print('# Other - Close                          #')
    option = int(input('# Please select one option: '))
    print('\n \n')
    return option

#Features menu
def features_menu():
    print('##########################################')
    print('#                Features                #')
    print('##########################################')
    print('# 0 - Keywords                           #')
    print('# 1 - Cast                               #')
    print('# 2 - Genres                             #')
    print('# 3 - Director                           #')
    print('# Other - Close                          #')
    option = input('# Please select the features you want (just type the numbers and Enter): ')
    print('\n \n')
    return option
##################################################

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

#Content Based Filltering
# def content_based():
#     print('content_based')
    #Select Features
    
    # features_selected_int = features_menu()
    # features_selected_str = []
    # for f in features_selected_int:
    #     features_selected_str.append(features[int(f)])
    

def movie_likes():
    print('movies_likes')
    features = ["keywords","cast","genres","director"] # Select features to take in acount
    
    for feature in features:
        df[feature] = df[feature].fillna('') #fills NaN with ''
    

    def combine_features(row):
        try:
            return row["keywords"] + " " + row["cast"] + " " + row["genres"] + " " + row["director"]
        except:
            print( "Error:" , row)
    
    df["combined_features"]= df.apply(combine_features, axis = 1) #axis=1 passes as rows and not columns
    print("Combined features \n", df["combined_features"].head())
    ##Step 4: Create count matrix from this new combined column
    cv = CountVectorizer()
    #counts the frequecy of the words
    count_matrix = cv.fit_transform(df["combined_features"])
    ##Step 5: Compute the Cosine Similarity based on the count_matrix
    #calculates similarity between points
    sim_scores = cosine_similarity(count_matrix)
    print(sim_scores)
    movie_user_likes = "Pulp Fiction"
    ## Step 6: Get id of this movie from its title
    movie_index = get_index_from_title(movie_user_likes)
    movie_director = get_director_from_title(movie_user_likes)
    print("Because you like the director" + movie_director + " : \n")
    # Find Similar movies
    # Converts matrix into a list and gives us inside a set of tuples
    similar_movies = list(enumerate(sim_scores[movie_index]))
    ## Step 7: Get a list of similar movies in descending order of similarity score
    # Sort similar
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
    #Remove movies who have less than 20 users who rated it | Fills NaN with 0
    user_ratings = user_ratings.dropna(thresh=20, axis=1).fillna(0)
    #print(user_ratings.head())
    item_similarity_df = user_ratings.corr(method='pearson')
    #print(item_similarity_df)
    #TODO Change to user input
    # Lookup do user id
    # Guardar os dados do titulo e do rating num tuple
    # Guardar o tuple num array
    #Pedir ao user um input ("Title", rating) x3
    action_lover = [("Up",4), ("Interstellar",4), ("Guardians of the Galaxy",3)]
    
    def get_similar_movies(movie_name, user_rating):
        similar_score = item_similarity_df[movie_name]*(user_rating-2.5) #subtracts the rating by the mean in order to correct the low values
        similar_score = similar_score.sort_values(ascending=False)
        return similar_score
    
    similar_movies = pd.DataFrame()
    for movie,rating in action_lover:
            similar_movies = similar_movies.append(get_similar_movies(movie, rating), ignore_index=True)
    print(similar_movies.sum().sort_values(ascending=False).head(10))
    return True

def logout():
    print('logout')
    return True

def close():
    return False
##################################################
#Read CSV File
df = pd.read_csv("tentaEste.csv")
cf = pd.read_csv("movies_and_ratings_small_dataset.csv")

welcome()
login()
options = { 
    0: top10,
    1: content_based,
    2: movie_likes,
    3: colaborative_based,
    4: welcome,
    5: logout,
}
flag = True
while flag:
    option = menu()
    func = options.get(option, close)
    flag = func()