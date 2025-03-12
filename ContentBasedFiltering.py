import pandas as pd

# Not sure what other path to use for now 
data_path = "/Users/rainfalls/Downloads/ml-32m/"

movies = pd.read_csv(data_path + "movies.csv")
tags = pd.read_csv(data_path + "tags.csv")
ratings = pd.read_csv(data_path + "ratings.csv")
links = pd.read_csv(data_path + "links.csv")

print(movies.head()) 

# Generate Text Descriptions for each movie (merging Genre & Tags)
# Fill missing values with empty strings
movies["genres"] = movies["genres"].fillna("")
tags["tag"] = tags["tag"].fillna("")

# Merge tags with movies
movie_tags = tags.groupby("movieId")["tag"].apply(lambda x: " ".join(x)).reset_index()
movies = movies.merge(movie_tags, on="movieId", how="left")

# Fill missing tags
movies["tag"] = movies["tag"].fillna("")

# Create movie descriptions
movies["description"] = movies["title"] + " " + movies["genres"] + " " + movies["tag"]
print(movies[["title", "description"]].head())

# Changing this into sentence embeddings 
from sentence_transformers import SentenceTransformer
import numpy as np
model = SentenceTransformer('all-MiniLM-L6-v2') 

# Convert descriptions to embeddings
embeddings = model.encode(movies["description"].tolist(), normalize_embeddings=True)

# Save embeddings for later use
np.save("movie_embeddings.npy", embeddings)
print(f"Embeddings shape: {embeddings.shape}")
