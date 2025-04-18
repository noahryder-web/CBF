import pandas as pd
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses

# Load the data
movies_df = pd.read_csv("../data/movies.csv")
tags_df = pd.read_csv("../data/tags.csv")

# Print a sample to check initial data
print("Initial movies data:")
print(movies_df.head())
print("Initial tags data:")
print(tags_df.head())

# Data cleaning for 'tags_df'
tags_df.dropna(subset=["tag"], inplace=True)  # Drop NaN values from 'tag' column
tags_df["tag"] = tags_df["tag"].astype(str)  # Convert 'tag' column to string

# Group tags by movieId and concatenate unique tags, ensuring order
tags_grouped = tags_df.groupby("movieId")["tag"].agg(lambda x: " ".join(sorted(set(x)))).reset_index()

# Ensure movieId types match before merging
movies_df["movieId"] = movies_df["movieId"].astype(int)
tags_grouped["movieId"] = tags_grouped["movieId"].astype(int)

# Merge movies and tags dataframes
movies_df = movies_df.merge(tags_grouped, on="movieId", how="left")
movies_df["tag"] = movies_df["tag"].fillna("")  # Fill NaN tags with empty string

# Create movie descriptions by combining genres and tags
movies_df["description"] = movies_df["genres"] + " " + movies_df["tag"]

# Ensure 'description' column exists
if "description" not in movies_df.columns:
    raise ValueError("The 'description' column is missing from movies_df.")

# Generate embeddings for movie descriptions
model = SentenceTransformer('all-MiniLM-L6-v2')
try:
    movie_descriptions = movies_df["description"].astype(str).tolist()
    embeddings = model.encode(movie_descriptions, normalize_embeddings=True)
    print("Embeddings generated successfully!")
except Exception as e:
    print(f"Error generating embeddings: {e}")

# Example data and model fine-tuning
train_data = [
    ("Toy Story is a great animation", "Finding Nemo is also amazing", 1),  # Similar
    ("Horror movies are scary", "Romantic comedies are funny", 0)  # Dissimilar
]
train_examples = [InputExample(texts=[a, b], label=float(score)) for a, b, score in train_data]
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=8)
loss_function = losses.ContrastiveLoss(model)

# Train the model
try:
    model.fit(train_objectives=[(train_dataloader, loss_function)], epochs=1)
    print("Fine-tuning complete!")
except Exception as e:
    print(f"Error during model training: {e}")

