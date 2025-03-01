import torch
from torch.utils.data import Dataset
import pandas as pd
from datetime import datetime
from src.utils import timestamp_to_month_index


class MovieRatingDataset(Dataset):
    def __init__(self, dataframe, user_dict, movie_dict, genre_dict, max_genres=5):
        """
        Custom dataset for Reward Model.

        Parameters:
        - dataframe: DataFrame containing 'userId', 'movieId', 'rating', 'timestamp', 'genres'.
        - user_dict: Mapping from userId to numerical ID.
        - movie_dict: Mapping from movieId to numerical ID.
        - genre_dict: Mapping from genre name to numerical ID.
        - max_genres: Maximum number of genres to include (padded if fewer).
        """
        self.df = dataframe
        self.user_dict = user_dict
        self.movie_dict = movie_dict
        self.genre_dict = genre_dict
        self.max_genres = max_genres
        self.reference_date = dataframe["timestamp"].min().strftime("%Y-%m-%d %H:%M:%S") 

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        user_id = self.user_dict.get(row["userId"], 0)  
        movie_id = self.movie_dict.get(row["movieId"], 0)

        genres = row["genres"].split("|")
        genre_ids = [self.genre_dict.get(g, 0) for g in genres]

        genre_ids = genre_ids[: self.max_genres]
        while len(genre_ids) < self.max_genres:
            genre_ids.append(0)  

        month_index = timestamp_to_month_index(row["timestamp"], self.reference_date)
        normalized_month = (month_index - 1) / 240  
        label_id = float(row["rating"])

        # Convert to tensors
        item = {
            "user_ids": torch.tensor(user_id, dtype=torch.long),
            "movie_ids": torch.tensor(movie_id, dtype=torch.long),
            "genre_ids": torch.tensor(genre_ids, dtype=torch.long),
            "timestamps": torch.tensor(normalized_month, dtype=torch.float),
            "labels": torch.tensor(label_id, dtype=torch.long),
        }
        return item
