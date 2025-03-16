
# Content Based Filtering

Content-based filtering (CBF) is a method of recommendation in which items are recommended based on a comparison between the content of the items and a user profile. The content of each item is represented as a set of descriptors or terms, typically the words that describe the items' attributes, and these are compared with the user profile, which describes the types of items this user likes.

In content-based filtering, items are represented as vectors in a multi-dimensional space, where each dimension corresponds to a feature of the items. The similarity between items is calculated using metrics like cosine similarity, Euclidean distance, or the dot product, depending on the nature of the vector space. For instance, cosine similarity measures the cosine of the angle between two item vectors, indicating how close they are directionally, regardless of their magnitude​.

This approach has several advantages, such as handling new items that have not yet been rated by users, since it doesn't rely on user interactions but rather on the features of the items themselves. It also provides transparency in the recommendation process by making it clear why items are being recommended based on their features. However, one of its disadvantages is the limitation in the features used. If the features don't capture all the reasons a user may like an item, the recommendations may not be accurate. Furthermore, this method tends to recommend items similar to those already rated by the user, potentially leading to a lack of diversity in the recommendations​.
​
In the context of our project, which uses sentence transformers to generate semantic embeddings of movie descriptions, content-based filtering fits well. We can enhance the descriptive power of our item profiles by incorporating rich semantic features derived from text descriptions using advanced NLP models. By doing so, we're able to capture a deeper understanding of the content, which should lead to more accurate recommendations based on the actual content of the movies rather than just surface-level metadata. This approach can help us overcome some of the traditional limitations of content-based filtering by providing a more nuanced view of what each movie is about, thus potentially increasing the diversity and accuracy of our recommendations.

# Content-Based Movie Recommendation System

This project implements a content-based filtering system to recommend movies based on semantic similarities in movie descriptions. It leverages the OMDb API to enhance movie metadata with detailed descriptions fetched from IMDb.

## Features

- Fetching movie overviews using the OMDb API.
- Generating semantic embeddings from movie descriptions.
- Calculating cosine similarity between movies to find similarities.
- Recommending movies based on similarity scores.

## Requirements

To run this project, you will need Python 3.8 or higher and several libraries which can be installed via pip:

```python
pip install pandas requests sentence-transformers scikit-learn

```

## API Key
You need an API key from OMDb to access IMDb data: 50f4a4bd

```python
api_key = '50f4a4bd'
data_path = "/path/to/your/dataset/"
```

## Implementation Details
### Data Preparation
Load and merge movie data using pandas, then fetch movie overviews from the OMDb API:

```bash
import pandas as pd
import requests
```

## Load datasets
```bash
movies = pd.read_csv(f"{data_path}movies.csv")
links = pd.read_csv(f"{data_path}links.csv")

def get_movie_overview(imdb_id):
    url = f'http://www.omdbapi.com/?i=tt{imdb_id}&apikey={api_key}'
    response = requests.get(url)
    data = response.json()
    return data.get('Plot', 'No overview available')

# Fetch and store overviews
links['imdbId'] = links['imdbId'].apply(lambda x: f"{int(x):07d}")
links['overview'] = links['imdbId'].apply(get_movie_overview)
movie_descriptions = movies.merge(links[['movieId', 'overview']], on='movieId')
movie_descriptions['description'] = movie_descriptions['genres'] + " " + movie_descriptions['overview']

```

## Generating Embeddings
Convert descriptions to embeddings:
```bash
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
descriptions = movie_descriptions['description'].tolist()
embeddings = model.encode(descriptions, normalize_embeddings=True)
```

## Cosine Similarity Calculation
Calculate the cosine similarity between movie embeddings to determine similarity:

```bash
from sklearn.metrics.pairwise import cosine_similarity

similarity_matrix = cosine_similarity(embeddings)
```

## Recommending Movies
Function to recommend movies based on a movie index:

```bash
def recommend_movies(movie_idx, top_n=5):
    similarity_scores = list(enumerate(similarity_matrix[movie_idx]))
    sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    recommended_indices = [i[0] for i in sorted_scores[1:top_n+1]]
    return movie_descriptions.iloc[recommended_indices]['title'].tolist()

# Example: Recommend top 5 movies similar to movie at index 10
print(recommend_movies(10, 5))
Running the Project
Execute the provided code snippets in a Python environment set up as described. Adjust the movie index in recommend_movies to test different recommendations.

```

### Instructions for Usage
1. Ensure the dataset paths in the script match where your `movies.csv` and `links.csv` files are stored.
2. Run the code in a Python environment to see the recommendations.







