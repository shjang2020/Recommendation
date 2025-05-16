import pandas as pd
import requests
import zipfile
import io
import os
import pickle
import numpy as np
import re
from typing import Tuple, Optional, Dict
from datetime import datetime
import logging
import time
from tqdm import tqdm

app_logger = logging.getLogger(__name__)

class MovieLensDataLoader:
    def __init__(self, data_dir='data', tmdb_api_key=None):
        self.data_dir = data_dir
        self.tmdb_api_key = tmdb_api_key
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
    
    def download_data(self):
        """Download MovieLens small dataset"""
        url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
        # Disable SSL verification due to expired certificate
        response = requests.get(url, verify=False)
        
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            zip_ref.extractall(self.data_dir)
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load movies and ratings data"""
        movies_path = os.path.join(self.data_dir, 'ml-latest-small', 'movies.csv')
        ratings_path = os.path.join(self.data_dir, 'ml-latest-small', 'ratings.csv')
        
        movies_df = pd.read_csv(movies_path)
        ratings_df = pd.read_csv(ratings_path)
        
        return movies_df, ratings_df
    
    def preprocess_title(self, title: str) -> Tuple[str, Optional[int]]:
        """Extract year from title and clean the title"""
        # Extract year from title (format: "Title (Year)")
        year_match = re.search(r'\((\d{4})\)', title)
        year = int(year_match.group(1)) if year_match else None
        
        # Clean title
        clean_title = re.sub(r'\(\d{4}\)', '', title).strip()
        
        return clean_title, year
    
    def normalize_genre(self, genre: str) -> str:
        """Normalize genre text"""
        # Convert to lowercase and remove special characters
        genre = genre.lower().strip()
        # Replace special characters with spaces
        genre = re.sub(r'[^a-z0-9\s]', ' ', genre)
        # Remove extra spaces
        genre = re.sub(r'\s+', ' ', genre).strip()
        return genre
    
    def get_tmdb_movie_info(self, title: str, year: Optional[int] = None) -> Optional[Dict]:
        """Get movie information from TMDB API"""
        if not self.tmdb_api_key:
            app_logger.warning("TMDB API key not provided. Skipping movie plot fetching.")
            return None
            
        try:
            # Search for the movie
            search_url = f"https://api.themoviedb.org/3/search/movie"
            params = {
                'api_key': self.tmdb_api_key,
                'query': title,
                'year': year
            }
            
            response = requests.get(search_url, params=params)
            response.raise_for_status()
            
            results = response.json().get('results', [])
            if not results:
                return None
                
            # Get the first result
            movie_id = results[0]['id']
            
            # Get movie details
            details_url = f"https://api.themoviedb.org/3/movie/{movie_id}"
            params = {'api_key': self.tmdb_api_key}
            
            response = requests.get(details_url, params=params)
            response.raise_for_status()
            
            movie_data = response.json()
            
            # Rate limiting
            time.sleep(0.25)  # TMDB API rate limit: 40 requests per 10 seconds
            
            return {
                'plot': movie_data.get('overview'),
                'poster_path': movie_data.get('poster_path'),
                'backdrop_path': movie_data.get('backdrop_path'),
                'tmdb_id': movie_id
            }
            
        except Exception as e:
            app_logger.error(f"Error fetching movie info from TMDB: {str(e)}")
            return None
    
    def prepare_movie_features(self, movies_df: pd.DataFrame, ratings_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare movie features with enhanced preprocessing"""
        # Create a copy to avoid modifying the original
        movies_df = movies_df.copy()
        
        # Extract year and clean title
        title_year = movies_df['title'].apply(self.preprocess_title)
        movies_df['clean_title'] = [t[0] for t in title_year]
        movies_df['year'] = [t[1] for t in title_year]
        
        # Process genres
        movies_df['genres'] = movies_df['genres'].str.split('|')
        movies_df['genres'] = movies_df['genres'].apply(
            lambda x: [self.normalize_genre(g) for g in x] if isinstance(x, list) else []
        )
        movies_df['genres_str'] = movies_df['genres'].apply(lambda x: ' '.join(x))
        
        # Add rating statistics
        rating_stats = ratings_df.groupby('movieId').agg({
            'rating': ['mean', 'std', 'count']
        }).reset_index()
        rating_stats.columns = ['movieId', 'avg_rating', 'rating_std', 'rating_count']
        
        movies_df = movies_df.merge(rating_stats, on='movieId', how='left')
        
        # Fill missing values
        movies_df['avg_rating'] = movies_df['avg_rating'].fillna(0)
        movies_df['rating_std'] = movies_df['rating_std'].fillna(0)
        movies_df['rating_count'] = movies_df['rating_count'].fillna(0)
        
        # Get movie plots from TMDB
        if self.tmdb_api_key:
            app_logger.info("Fetching movie plots from TMDB...")
            print("\nFetching movie plots from TMDB...")
            movie_plots = []
            valid_movie_indices = []  # Store indices of movies with valid TMDB info
            
            # Try to load existing progress
            progress_file = os.path.join(self.data_dir, 'movie_plots_progress.pkl')
            if os.path.exists(progress_file):
                with open(progress_file, 'rb') as f:
                    movie_plots = pickle.load(f)
                app_logger.info(f"Loaded {len(movie_plots)} existing movie plots")
            
            for idx, row in tqdm(movies_df.iterrows(), total=len(movies_df), desc="Fetching movie info"):
                # Skip if we already have this movie's data
                if idx < len(movie_plots):
                    if movie_plots[idx] is not None:  # Only keep movies with valid TMDB info
                        valid_movie_indices.append(idx)
                    continue
                    
                plot_info = self.get_tmdb_movie_info(row['clean_title'], row['year'])
                if plot_info:
                    movie_plots.append(plot_info)
                    valid_movie_indices.append(idx)
                else:
                    movie_plots.append(None)
                
                # Save progress every 1000 movies
                if (idx + 1) % 1000 == 0:
                    with open(progress_file, 'wb') as f:
                        pickle.dump(movie_plots, f)
                    app_logger.info(f"Saved progress: {idx + 1} movies processed")
            
            # Filter movies_df to only include movies with valid TMDB info
            movies_df = movies_df.iloc[valid_movie_indices].reset_index(drop=True)
            
            # Add plot information to movies_df
            movies_df['plot'] = [p['plot'] for p in movie_plots if p is not None]
            movies_df['poster_path'] = [p['poster_path'] for p in movie_plots if p is not None]
            movies_df['backdrop_path'] = [p['backdrop_path'] for p in movie_plots if p is not None]
            movies_df['tmdb_id'] = [p['tmdb_id'] for p in movie_plots if p is not None]
            
            # Clean up progress file after successful completion
            if os.path.exists(progress_file):
                os.remove(progress_file)
            
            app_logger.info(f"Movie plot fetching completed! {len(movies_df)} movies with valid TMDB info")
        
        return movies_df
    
    def validate_data(self, movies_df: pd.DataFrame, ratings_df: pd.DataFrame) -> bool:
        """Validate data quality"""
        try:
            # Check for missing values
            missing_movies = movies_df.isnull().sum()
            missing_ratings = ratings_df.isnull().sum()
            
            if missing_movies.sum() > 0 or missing_ratings.sum() > 0:
                app_logger.warning("Missing values found:")
                app_logger.warning(f"Movies: {missing_movies[missing_movies > 0]}")
                app_logger.warning(f"Ratings: {missing_ratings[missing_ratings > 0]}")
            
            # Check for duplicate movie IDs
            duplicate_movies = movies_df['movieId'].duplicated().sum()
            if duplicate_movies > 0:
                app_logger.warning(f"{duplicate_movies} duplicate movie IDs found")
            
            # Check rating range
            invalid_ratings = ratings_df[
                (ratings_df['rating'] < 0.5) | (ratings_df['rating'] > 5.0)
            ]
            if len(invalid_ratings) > 0:
                app_logger.warning(f"{len(invalid_ratings)} invalid ratings found")
            
            # Check year range (only for non-null values)
            if 'year' in movies_df.columns:
                invalid_years = movies_df[
                    movies_df['year'].notna() & 
                    ((movies_df['year'] < 1900) | (movies_df['year'] > datetime.now().year))
                ]
                if len(invalid_years) > 0:
                    app_logger.warning(f"{len(invalid_years)} invalid years found")
            
            return True
            
        except Exception as e:
            app_logger.error(f"Error during data validation: {str(e)}")
            return False
    
    def save_processed_data(self, movies_df: pd.DataFrame, ratings_df: pd.DataFrame):
        """Save processed data to files"""
        processed_data_path = os.path.join(self.data_dir, 'processed_data.pkl')
        with open(processed_data_path, 'wb') as f:
            pickle.dump({
                'movies_df': movies_df,
                'ratings_df': ratings_df
            }, f)
        print(f"Processed data saved to {processed_data_path}")
    
    def load_processed_data(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Load processed data from files"""
        processed_data_path = os.path.join(self.data_dir, 'processed_data.pkl')
        if os.path.exists(processed_data_path):
            with open(processed_data_path, 'rb') as f:
                data = pickle.load(f)
            print(f"Processed data loaded from {processed_data_path}")
            return data['movies_df'], data['ratings_df']
        return None, None
    
    def save_embeddings(self, genre_embeddings: np.ndarray, plot_embeddings: np.ndarray):
        """Save BERT embeddings to file"""
        genre_embeddings_path = os.path.join(self.data_dir, 'genre_embeddings.npy')
        plot_embeddings_path = os.path.join(self.data_dir, 'plot_embeddings.npy')
        np.save(genre_embeddings_path, genre_embeddings)
        np.save(plot_embeddings_path, plot_embeddings)
        print(f"Embeddings saved to {genre_embeddings_path} and {plot_embeddings_path}")
    
    def load_embeddings(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Load BERT embeddings from file"""
        genre_embeddings_path = os.path.join(self.data_dir, 'genre_embeddings.npy')
        plot_embeddings_path = os.path.join(self.data_dir, 'plot_embeddings.npy')
        if os.path.exists(genre_embeddings_path):
            genre_embeddings = np.load(genre_embeddings_path)
            print(f"Embeddings loaded from {genre_embeddings_path}")
        else:
            genre_embeddings = None
        if os.path.exists(plot_embeddings_path):
            plot_embeddings = np.load(plot_embeddings_path)
            print(f"Embeddings loaded from {plot_embeddings_path}")
        else:
            plot_embeddings = None
        return genre_embeddings, plot_embeddings

if __name__ == "__main__":
    # Test the data loader
    loader = MovieLensDataLoader()
    loader.download_data()
    movies_df, ratings_df = loader.load_data()
    
    # Validate raw data
    print("\nValidating raw data...")
    loader.validate_data(movies_df, ratings_df)
    
    # Process data
    movies_df = loader.prepare_movie_features(movies_df, ratings_df)
    
    # Validate processed data
    print("\nValidating processed data...")
    loader.validate_data(movies_df, ratings_df)
    
    print("\nProcessed Movies shape:", movies_df.shape)
    print("\nSample movies with enhanced features:")
    print(movies_df[['movieId', 'clean_title', 'year', 'genres', 'avg_rating', 'rating_count']].head()) 