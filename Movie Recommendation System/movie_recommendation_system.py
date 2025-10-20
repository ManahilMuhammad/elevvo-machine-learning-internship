# --> BEGINNING OF: importing libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
import kagglehub
# <-- END OF: importing libraries

# --> BEGINNING OF: loading data
@st.cache_data
def load_data():
    # load dataset
    path = kagglehub.dataset_download("prajitdatta/movielens-100k-dataset")

    # load u.item and u.data (movie and rating files)
    movies = pd.read_csv(
        f"{path}/ml-100k/u.item", sep="|", encoding="latin-1",
        header=None, usecols=[0, 1], names=["movie_id", "title"]
    )
    ratings = pd.read_csv(
        f"{path}/ml-100k/u.data", sep="\t",
        names=["user_id", "movie_id", "rating", "timestamp"]
    )
    
    return movies, ratings

movies, ratings = load_data()
# <-- END OF: loading data

# --> BEGINNING OF: creating user-item matrix
ui_matrix = ratings.pivot_table(index="user_id", columns="movie_id", values="rating")
# <-- END OF: creating user-item matrix

# --> BEGINNING OF: function computing cosine similarity b/w users
def user_sim(matrix):
    return pd.DataFrame(cosine_similarity(matrix.fillna(0)), index=matrix.index, columns=matrix.index)
# <-- END OF: function computing cosine similarity b/w users

# --> BEGINNING OF: function computing cosine similarity b/w items
def item_sim(matrix):
    return pd.DataFrame(cosine_similarity(matrix.fillna(0).T), index=matrix.columns, columns=matrix.columns)
# <-- END OF: function computing cosine similarity b/w items

# --> BEGINNING OF: function making predictions using user-based collaborative filtering
def user_based(user_id, matrix, user_sim):
    # compute each user's mean rating
    mean_ratings = matrix.mean(axis=1)
    
    # normalise each user's mean rating
    normalised_ratings = (matrix.T - mean_ratings).T
    
    # compute similarity scores for relevant user
    sim_scores = user_sim[user_id]
    
    # compute weighted sum of other users' ratings
    # where weight corresponds to similarity w/ relevant user
    weighted_sum = normalised_ratings.T.dot(sim_scores)
    
    # normalisation
    sim_sum = np.abs(sim_scores).sum()

    predictions = mean_ratings[user_id] + weighted_sum / sim_sum
    
    return predictions
# <-- END OF: function making predictions using user-based collaborative filtering

# --> BEGINNING OF: function making predictions using item-based collaborative filtering
def item_based(user_id, matrix, item_sim):
    # get all ratings of relevant user
    user_ratings = matrix.loc[user_id]
    
    # identify all items rated by user
    rated_items = user_ratings.dropna().index
    
    # return empty series if user has no ratings
    if len(rated_items) == 0:
        return pd.Series(dtype=float)
    
    # compute weighted scores
    # where weight corresponds to similarity of an unseen movie w/ one the user rated
    weighted_scores = item_sim[rated_items].dot(user_ratings[rated_items])
    
    # normalisation
    sim_sums = np.abs(item_sim[rated_items]).sum(axis=1)
    
    predictions = weighted_scores / sim_sums
    
    return predictions
# <-- END OF: function making predictions using item-based collaborative filtering

# --> BEGINNING OF: function computing precision at K (default K is 10)
def k_precision(recs_dict, test_data, k=10, threshold=4.0):
    # initialise counter keeping track of correct recommendations as 0
    relevant_recs = 0
    
    for user_id in recs_dict.keys():
        # find top K recommendations for a user
        recs = recs_dict[user_id][:k]
        
        # get user's actual ratings
        all_ratings = test_data[test_data["user_id"] == user_id]
        
        # find movies the user liked
        liked_items = all_ratings[all_ratings["rating"] >= threshold]["movie_id"].values
        
        # count number of relevant recommendations 
        relevant_recs += len(set(recs).intersection(set(liked_items)))
    return relevant_recs / (len(recs_dict) * k)
# <-- END OF: function computing precision at K (default K is 10)

# --> BEGINNING OF: UI
# title
st.title("📽 Movie Recommendation System")

# instructions
st.markdown("Select and rate movies to get personalized recommendations!")

# ask user to select movies
selected_movies = st.multiselect("Choose movies you’ve seen:", movies["title"].values)

# initialise ratings to an empty dictionary
user_ratings = {}

# ask user to rate each selected movie using a slider
for movie in selected_movies:
    # slider values range from 1 to 5 and default is 3
    rating = st.slider(f"Rate '{movie}'", 1, 5, 3)
    
    # add movie and rating as key-value pair to the dictionary
    user_ratings[movie] = rating

# ask user to select the method of prediction to use
method = st.selectbox(
    "Prediction Method",
    ["User-Based Collaborative Filtering", "Item-Based Collaborative Filtering", "Matrix Factorisation"]
)

# button to generate recommendations
if st.button("Get Recommendations"):
    # handle the case in which the user doesn't select any movie
    if not user_ratings:
        st.warning("Please select and rate at least one movie.")
        
    else:
        # assign a temporary user ID 
        temp_user_id = 9999
        
        # temporary dataframe created as a copy of the ratings dataframe
        # new user's ratings will be appended to this copy
        temp_df = ratings.copy()
        
        # add the temporary user's ratings to the temporary dataframe
        for movie, rating in user_ratings.items():
            movie_id = movies[movies["title"] == movie]["movie_id"].values[0]
            temp_df.loc[len(temp_df)] = [temp_user_id, movie_id, rating, 0]

        # create a temporary user-item matrix 
        # by converting the temporary dataframe into a pivot table
        temp_matrix = temp_df.pivot_table(index="user_id", columns="movie_id", values="rating")

        # if the chosen method is user-based collaborative filtering
        if method == "User-Based Collaborative Filtering":
            # compute user similarity
            user_sim = user_sim(temp_matrix)
            
            # compute predictions
            preds = user_based(temp_user_id, temp_matrix, user_sim)

        # if the chosen method is item-based collaborative filtering
        elif method == "Item-Based Collaborative Filtering":
            # compute item similarity
            item_sim = item_sim(temp_matrix)
            
            # compute predictions
            preds = item_based(temp_user_id, temp_matrix, item_sim)

        else:
            # fill any missing values in the temporary matrix w/ 0
            matrix_filled = temp_matrix.fillna(0)
            
            svd = TruncatedSVD(n_components=20, random_state=42)
            
            # factorise matrix
            latent_matrix = svd.fit_transform(matrix_filled)
            
            # reconstruct matrix of predicted ratings
            reconstructed = np.dot(latent_matrix, svd.components_)
            
            # extract predictions
            preds = pd.Series(reconstructed[temp_matrix.index.get_loc(temp_user_id)], index=temp_matrix.columns)

        # filter movies so that only unseen ones are recommended
        rated_movies = [movies[movies["title"] == m]["movie_id"].values[0] for m in user_ratings.keys()]
        preds = preds.drop(rated_movies, errors="ignore")
        
        # show top 10 recommended movies
        top_movies = preds.sort_values(ascending=False).head(10).index
        recommended = movies[movies["movie_id"].isin(top_movies)]["title"].values

        # show recommended movies
        st.subheader("⏭ Recommended Movies for You:")
        for m in recommended:
            st.write(f"- {m}")

# --> BEGINNING OF: section to evaluate precision at K
st.markdown("---")
st.subheader("✍︎ Evaluate Performance")

# button to evaluate performance
if st.button("Evaluate Precision at K"):
    # split data into training and testing sets
    # training is 80% and testing is 20%
    train, test = train_test_split(ratings, test_size=0.2, random_state=42)
    
    # create user-item matrix for training
    train_matrix = train.pivot_table(index="user_id", columns="movie_id", values="rating")

    # compute similarity matrices
    user_sim = user_sim(train_matrix)
    item_sim = item_sim(train_matrix)

    # select the fist 20 users (for speed purposes)
    # to use for evaluation
    all_users = train_matrix.index[:20]
    
    # initialise an empty dictionary of recommendations
    recommendations = {}

    for uid in all_users:
        # --> BEGINNING OF: computing predictions based on prediction method
        if method == "User-Based Collaborative Filtering":
            preds = user_based(uid, train_matrix, user_sim)
        elif method == "Item-Based Collaborative Filtering":
            preds = item_based(uid, train_matrix, item_sim)
        else:
            matrix_filled = train_matrix.fillna(0)
            svd = TruncatedSVD(n_components=20, random_state=42)
            latent_matrix = svd.fit_transform(matrix_filled)
            reconstructed = np.dot(latent_matrix, svd.components_)
            preds = pd.Series(reconstructed[train_matrix.index.get_loc(uid)], index=train_matrix.columns)
        # <-- END OF: computing predictions based on prediction method

        # filter out seen movies
        seen = train_matrix.loc[uid].dropna().index
        preds = preds.drop(seen, errors="ignore")
        
        # choose top 10 recommendations
        top_recs = preds.sort_values(ascending=False).head(10).index
        recommendations[uid] = top_recs.tolist()

    # evaluate precision at k
    precision = k_precision(recommendations, test, k=10)
    
    # display computed precision
    st.success(f"✔︎ Precision at 10: {precision:.4f}")

# <-- END OF: section to evaluate precision at K
# <-- END OF: UI
