import pandas as pd
import streamlit as st
import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import faiss  # Faster nearest neighbors

# TMDb API Key (Replace with your own key)
TMDB_API_KEY = "5169792254a831e7aabdb4402d1f27ec"

# Load datasets from GitHub
GITHUB_RAW_URL = "https://raw.githubusercontent.com/surya077-mj/movie_recommendation/main/movie_recommender/"

def load_csv(file_name, dtype=None):
    return pd.read_csv(GITHUB_RAW_URL + file_name, dtype=dtype, low_memory=False)

movies = load_csv("movies.csv")
ratings = load_csv("ratings.csv", dtype={"userId": "int32", "movieId": "int32", "rating": "float32"})
tags = load_csv("tags.csv")
links = load_csv("links.csv", dtype={"movieId": "int32", "tmdbId": "Int32", "imdbId": "Int32"})

# Display number of movies loaded
st.write(f"Loaded {len(movies)} movies")

# Merge TMDb IDs with movies dataset
movies = movies.merge(links[["movieId", "tmdbId", "imdbId"]], on="movieId", how="left")

# Handle missing values
movies["genres"].fillna("", inplace=True)
movies["tmdbId"].fillna(0, inplace=True)
movies["imdbId"].fillna(0, inplace=True)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(movies["genres"]).toarray().astype("float32")  # Convert to dense and float32

# Use Faiss for faster nearest neighbors
index = faiss.IndexFlatL2(tfidf_matrix.shape[1])
index.add(tfidf_matrix)

# Preprocess movie titles
def clean_title(title):
    return title.lower().strip().split(" (", 1)[0]

movies["title_clean"] = movies["title"].apply(clean_title)

# Compute average ratings
avg_ratings = ratings.groupby("movieId")["rating"].mean().reset_index()
movies = movies.merge(avg_ratings, on="movieId", how="left")
movies["rating"].fillna(movies["rating"].mean(), inplace=True)

# Fetch movie poster from TMDb
def get_movie_poster(tmdb_id):
    if tmdb_id == 0 or pd.isna(tmdb_id):
        return None
    url = f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}?api_key={TMDB_API_KEY}"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        poster_path = data.get("poster_path")
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500{poster_path}"
        else:
            st.write(f"No poster found for TMDb ID: {tmdb_id}")
            return None
    except requests.RequestException as e:
        st.write(f"Error fetching poster for TMDb ID {tmdb_id}: {e}")
        return None

# Get movie recommendations
def get_recommendations(title, movies, index, tfidf_matrix, num_recommendations=5, min_rating=0):
    title = clean_title(title)
    idx = movies.index[movies["title_clean"] == title].tolist()
    if not idx:
        return []
    idx = idx[0]
    
    distances, indices = index.search(np.array([tfidf_matrix[idx]]).astype("float32"), num_recommendations + 1)
    recommended_movies = movies.iloc[indices[0][1:]].copy()
    recommended_movies = recommended_movies[recommended_movies["rating"] >= min_rating]
    
    if len(recommended_movies) < num_recommendations:
        top_movies = movies.sort_values(by="rating", ascending=False).head(num_recommendations - len(recommended_movies))
        recommended_movies = pd.concat([recommended_movies, top_movies])
    
    return recommended_movies.sort_values(by="rating", ascending=False)[["title", "rating", "tmdbId", "imdbId"]].head(num_recommendations).values.tolist()

# Streamlit UI
st.title("🎬 Movie Recommender")

movie_title = st.text_input("Enter a movie title:")
num_recommendations = st.slider("How many recommendations do you want?", 1, 10, 5)
min_rating = st.slider("Enter minimum rating for recommendations:", 0.0, 5.0, 3.0, 0.1)

if st.button("Get Recommendations"):
    if movie_title:
        with st.spinner("Fetching recommendations..."):
            recommendations = get_recommendations(movie_title, movies, index, tfidf_matrix, num_recommendations, min_rating)
        
        if recommendations:
            st.subheader(f"Movies similar to '{movie_title}':")
            for rec in recommendations:
                title, rating, tmdb_id, imdb_id = rec
                poster_url = get_movie_poster(tmdb_id) if tmdb_id else None
                imdb_url = f"https://www.imdb.com/title/tt{str(int(imdb_id)).zfill(7)}/" if imdb_id and imdb_id != 0 else None
                
                col1, col2 = st.columns([1, 3])
                with col1:
                    if poster_url:
                        st.image(poster_url, width=100)
                    else:
                        st.write(f"No Poster Available (TMDb ID: {tmdb_id})")
                with col2:
                    if imdb_url:
                        st.markdown(f"**{title}** (⭐ {round(rating, 1)}) - [IMDb Link]({imdb_url})")
                    else:
                        st.markdown(f"**{title}** (⭐ {round(rating, 1)})")
        else:
            st.warning(f"No recommendations found for '{movie_title}'. Showing top-rated movies instead.")
            top_movies = movies.sort_values(by="rating", ascending=False).head(num_recommendations)
            for _, row in top_movies.iterrows():
                imdb_url = f"https://www.imdb.com/title/tt{str(int(row['imdbId'])).zfill(7)}/" if pd.notnull(row['imdbId']) and row['imdbId'] != 0 else "No IMDB Link Available"
                st.markdown(f"- **{row['title']}** (⭐ {round(row['rating'], 1)}) - [IMDb Link]({imdb_url})")