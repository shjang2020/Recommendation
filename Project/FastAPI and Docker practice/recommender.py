from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertModel
import torch
import random

class BertBasedRecommender:
    def __init__(self):
        # Load pre-trained DistilBERT model and tokenizer
        model_path = './model'
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = DistilBertModel.from_pretrained(model_path, local_files_only=True)
        print(f"Models loaded from {model_path}")
        self.movies_df = None
        self.genre_embeddings = None
        self.plot_embeddings = None
        
    def get_bert_embedding(self, text):
        """Get BERT embedding for a text"""
        # Tokenize and prepare input
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        # Get BERT embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Use [CLS] token embedding as sentence embedding
        embeddings = outputs.last_hidden_state[:, 0, :].numpy()
        return embeddings
    
    def fit(self, movies_df, genre_embeddings=None, plot_embeddings=None):
        """Fit the BERT model on movie genres or use pre-computed embeddings"""
        self.movies_df = movies_df
        
        if genre_embeddings is not None:
            self.genre_embeddings = genre_embeddings
        else:
            # Get BERT embeddings for all movies genres
            print("Generating BERT embeddings for movies genres...")
            self.genre_embeddings = np.vstack([
                self.get_bert_embedding(genres) for genres in movies_df['genres_str']
            ])
            print("Genre embeddings generation completed!")
        
        if plot_embeddings is not None:
            self.plot_embeddings = plot_embeddings
        else:
            # Get BERT embeddings for all movies plot
            print("Generating BERT embeddings for movies plot...")
            self.plot_embeddings = np.vstack([
                self.get_bert_embedding(plot) for plot in movies_df['plot']
            ])
            print("Plot embeddings generation completed!")
        return self
    
    def recommend(self, genres=None, keywords=None, n_recommendations=1, start_year=1900, end_year=2020):
        """Recommend movies based on genres and/or keywords"""
        # Create a filtered copy of the dataframe
        filtered_df = self.movies_df.copy()
        
        # limit the recommendation to a specific year range
        if start_year and end_year:
            filtered_df = filtered_df[
                (filtered_df['year'] >= start_year) & (filtered_df['year'] <= end_year)
            ]
        elif start_year:
            filtered_df = filtered_df[
                (filtered_df['year'] >= start_year)
            ]
        elif end_year:
            filtered_df = filtered_df[
                (filtered_df['year'] <= end_year)
            ] 
            
        print(f"Debug - Filtered DataFrame size: {len(filtered_df)}")
        print(f"Debug - Original DataFrame size: {len(self.movies_df)}")
            
        # If no input is provided, return random recommendations
        if not genres and not keywords:
            random_indices = random.sample(range(len(filtered_df)), n_recommendations)
            recommendations = filtered_df.iloc[random_indices][
                ['clean_title', 'year', 'genres', 'avg_rating', 'rating_count', 'poster_path', 'backdrop_path', 'plot']
            ]
            recommendations['similarity_score'] = 0.0
            return recommendations
            
        # Get embeddings for genres and keywords
        genre_embedding = None
        keyword_embedding = None
        
        # Get the indices of filtered movies in the original dataframe
        filtered_indices = filtered_df.index
        print(f"Debug - Filtered indices: {filtered_indices}")
        print(f"Debug - Genre embeddings shape: {self.genre_embeddings.shape}")
        print(f"Debug - Plot embeddings shape: {self.plot_embeddings.shape}")
        
        if genres:
            genre_embedding = self.get_bert_embedding(' '.join(genres))
            # Get similarities only for filtered movies
            genre_similarities = cosine_similarity(genre_embedding, self.genre_embeddings[filtered_indices]).flatten()
            print(f"Debug - Genre similarities shape: {genre_similarities.shape}")
        else:
            genre_similarities = np.ones(len(filtered_df))
            
        if keywords:
            keyword_embedding = self.get_bert_embedding(keywords)
            # Get similarities only for filtered movies
            keyword_similarities = cosine_similarity(keyword_embedding, self.plot_embeddings[filtered_indices]).flatten()
            print(f"Debug - Keyword similarities shape: {keyword_similarities.shape}")
        else:
            keyword_similarities = np.ones(len(filtered_df))
        
        # Combine similarities (geometric mean)
        combined_similarities = np.sqrt(genre_similarities * keyword_similarities)
        
        # Get top N similar movies (ensure n_recommendations doesn't exceed available movies)
        n_recommendations = min(n_recommendations, len(filtered_df))
        similar_indices = combined_similarities.argsort()[-n_recommendations:][::-1]
        
        # Get movie details with enhanced features
        recommendations = filtered_df.iloc[similar_indices][
            ['clean_title', 'year', 'genres', 'avg_rating', 'rating_count', 'poster_path', 'backdrop_path', 'plot']
        ]
        recommendations['similarity_score'] = combined_similarities[similar_indices]
        
        return recommendations