import pandas as pd
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import time
import json

# Configurations
api_key = '274b3d14'  # Replace this with your actual OMDb API key
data_path = "/Users/rainfalls/Downloads/ml-32m/"
cache_path = "overview_cache.json"  # Cache file path
max_requests = 1000  # Limit the number of API requests

# Load cache
try:
    with open(cache_path, "r") as file:
        cache = json.load(file)
except FileNotFoundError:
    cache = {}

def get_movie_overview(imdb_id):
    if imdb_id in cache:
        return cache[imdb_id]
    if get_movie_overview.requests_made >= max_requests:
        return 'Request limit reached, no more API calls.'

    url = f"http://www.omdbapi.com/?i={imdb_id}&apikey={api_key}"
    response = requests.get(url)
    get_movie_overview.requests_made += 1

    if response.status_code == 200:
        data = response.json()
        if 'Error' in data:
            return 'No overview available'
        cache[imdb_id] = data.get('Plot', 'No overview available')
        return data.get('Plot', 'No overview available')
    elif response.status_code == 401 and 'Request limit reached' in response.text:
        time.sleep(60)
        return get_movie_overview(imdb_id)
    else:
        return 'No overview available'

get_movie_overview.requests_made = 0  # Initialize the counter for API requests

def save_cache():
    with open(cache_path, "w") as file:
        json.dump(cache, file)

print("Loading movie data...")
movies = pd.read_csv(f"{data_path}movies.csv")
links = pd.read_csv(f"{data_path}links.csv")
print("Data loaded successfully.")

print("Processing IMDb IDs and fetching overviews...")
for index, row in links.iterrows():
    if index >= max_requests:  # Stop processing if the max request limit is reached
        break
    links.at[index, 'overview'] = get_movie_overview(str(row['imdbId']))
    time.sleep(2)  # Adjust or remove sleep as needed based on your rate limits

save_cache()

movie_descriptions = movies.merge(links[['movieId', 'overview']], on='movieId')
movie_descriptions['description'] = movie_descriptions['genres'] + " " + movie_descriptions['overview']

print("Converting descriptions to embeddings...")
model = SentenceTransformer('all-MiniLM-L6-v2')
descriptions = movie_descriptions['description'].tolist()
embeddings = model.encode(descriptions, normalize_embeddings=True)
print("Embeddings created.")

print("Calculating cosine similarity...")
similarity_matrix = cosine_similarity(embeddings)
print("Cosine similarity matrix calculated.")

def recommend_movies(movie_idx, top_n=5):
    print(f"Generating recommendations for movie index: {movie_idx}")
    similarity_scores = list(enumerate(similarity_matrix[movie_idx]))
    sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    recommended_indices = [i[0] for i in sorted_scores[1:top_n+1]]
    recommended_movies = movie_descriptions.iloc[recommended_indices]['title'].tolist()
    return recommended_movies

recommended_movies = recommend_movies(10, 5)
print("Recommended Movies:", recommended_movies)
