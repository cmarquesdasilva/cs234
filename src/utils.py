"""
This module provides utility functions for loading, merging, and processing movie rating data.
It includes functions to load and merge datasets, generate user profile prompts, and split data by time.
"""
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import os
import torch

# Model Traning Utilities
def load_and_merge_data() -> pd.DataFrame:
    """
    Load and merge movie rating data with metadata.
    
    Returns:
        pd.DataFrame: Merged dataset with user ratings and movie titles.
    """
    print("Loading datasets...")
    ratings = pd.read_csv("src/data/ratings.csv")
    movies = pd.read_csv("src/data/movies.csv")

    # Convert timestamp to datetime
    ratings["timestamp"] = pd.to_datetime(ratings["timestamp"], unit="s")
    print(ratings["timestamp"].min(), ratings["timestamp"].max())
    # Merge movie titles
    data = ratings.merge(movies, on="movieId", how="left")
    return data

def generate_prompt_for_user_profile(data: pd.DataFrame, type: str = "train") -> pd.DataFrame:
    """
    Generate user profile prompts from a dataset efficiently without an explicit double loop.
    
    Parameters:
    - data: DataFrame with columns ["userId", "title", "genres", "rating"]
    - type: "train" for generating prompts, otherwise returns original data.
    
    Returns:
    - DataFrame with user profile prompts
    """
    user_prompts = []
    collection = []
    accumulated_length = 0
    data.sort_values(by="timestamp", ascending=False, inplace=True)
    user_id = data.iloc[0]["userId"]
    
    for i, row in data.iterrows():
        user_data_str = "".join(
            f"{row['title']} ({row['genres']}): {row['rating']}."
        )

        # Count number of tokens:
        length = len(user_data_str.split())
        accumulated_length += length
        if accumulated_length >= 8000:
            break
        collection.append(user_data_str)
    user_data_str = ",".join(collection)
    user_prompts.append({"userId": user_id, "prompt": f"User {user_id} has rated the following movies:\n{user_data_str}\n"})

    return pd.DataFrame(user_prompts)


def split_data_by_time(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits data into:
    1) **Training Set** 
    2) **Validation Set** 
    
    **Rules:**
    - Each user's training & validation set must follow temporal order.
    - Validation starts strictly after the last training timestamp.

    Returns:
        - `train_df`: Training dataset.
        - `val_df`: Validation dataset.
    """
    train_list, val_list = [], []

    for user, user_data in tqdm(data.groupby("userId"), desc="Processing users"):
        user_data_size = user_data.shape[0]
        if user_data_size < 4:
            continue  # Skip users with very few interactions

        # Sort user data by timestamp
        user_data = user_data.sort_values("timestamp", ascending=True)

        # **Train set: Earliest 80% of user’s data**
        split_idx = int(user_data_size * 0.8)
        train_data = user_data.iloc[:split_idx]

        # **Validation set: All timestamps AFTER last training timestamp**
        last_train_timestamp = train_data["timestamp"].max()
        val_data = user_data[user_data["timestamp"] > last_train_timestamp]
        if val_data.empty:
            continue  # Skip users with no valid validation data

        # Generate training data
        train_list.append(train_data)
        val_list.append(val_data)

    train_df = pd.concat(train_list)
    val_df = pd.concat(val_list)

    return train_df, val_df

# Model Training Utilities #
def save_model(model, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
    print(f"Model saved to {output_dir}")

def normalize_month(month_index: int, max_month=60) -> float:
    """
    Normalize month index to range [0,1].
    """
    return (month_index - 1) / (max_month - 1)


def timestamp_to_month_index(timestamp: str, reference_date: str) -> int:
    """
    Convert timestamp (datetime format) to a month index.
    
    Parameters:
    - timestamp: The timestamp to be converted.
    - reference_date: The reference date for the conversion.
    
    Returns:
    - int: The month index.
    """
    timestamp = timestamp.strftime("%Y-%m-%d %H:%M:%S") 
    reference_date = datetime.strptime(reference_date, "%Y-%m-%d %H:%M:%S")
    date = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")

    year_diff = date.year - reference_date.year
    month_diff = date.month - reference_date.month
    month_index = year_diff * 12 + month_diff + 1
    return month_index

# Analytics Utilities #
def convert_string(row: any) -> int:
    """
    Convert a string to an integer if it is a digit, otherwise return -1 for strings.
    
    Parameters:
    - row: The value to be converted.
    
    Returns:
    - int: The converted integer or -1 if the input is a non-digit string.
    """
    if isinstance(row, str) and row.isdigit():
        return int(float(row))
    elif isinstance(row, str):
        return -1
    return row