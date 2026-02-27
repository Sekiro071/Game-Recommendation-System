#Respawned – NLP-Based Game Recommendation System

Respawned is a content-based game recommendation system built using Natural Language Processing (NLP).  
It analyzes game metadata and recommends similar games based on user input.

Built using a Kaggle dataset containing **26,000+ games**, the system leverages TF-IDF vectorization and cosine similarity to generate relevant recommendations.

#Features

- Content-based recommendation engine
- TF-IDF vectorization on structured metadata
- Cosine similarity for similarity ranking
- Handles 26,000+ games efficiently
- Interactive web interface built using Gradio
- Dynamic game image display with fallback support
- Custom gaming-themed UI with advanced CSS styling

#How It Works

1. Game metadata (genres, categories, tags, developer) is combined into a structured text format.
2. A custom semicolon-based analyzer is used with `TfidfVectorizer`.
3. Cosine similarity matrix is computed across all games.
4. When a user enters a game title:
   - The best match is found using partial name matching.
   - Similarity scores are ranked.
   - Top-N similar games are displayed with images.

#Tech Stack

- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorizer
- Cosine Similarity
- Gradio (UI)
- HTML & Custom CSS

#Dataset

- Source: Kaggle
- Size: 26,000+ games
- Includes:
  - Game Name
  - Genres
  - Categories
  - SteamSpy Tags
  - Developer
  - Header Image URLs
