import torch
from torch.utils.data import Dataset
import pandas as pd
from datetime import datetime
from typing import Dict
from src.utils import timestamp_to_month_index, load_and_merge_data

LABEL_MAP = {
    0.5: 0,
    1.0: 1,
    1.5: 2,
    2.0: 3,
    2.5: 4,
    3.0: 5,
    3.5: 6,
    4.0: 7,
    4.5: 8,
    5.0: 9,
}
BINARY_LABEL_MAP = {rating: 0 if rating < 3 else 1 for rating in LABEL_MAP.keys()}


def create_movie_id_map() -> Dict[int, int]:
    """
    Build a dictionary to map sparse/large movieIds to a compact range 0..(num_movies-1).
    """
    df = load_and_merge_data()
    unique_movie_ids = sorted(df["movieId"].unique())
    return {m: i for i, m in enumerate(unique_movie_ids)}


class MovieRatingDataset(Dataset):
    """
    A simplified dataset class:
      - Uses the original userId directly for embedding (assuming userId is 1..605).
      - Remaps movieId to a compact index for embedding via movie2idx.
      - Treats genres as a single text field for a text embedder.
      - Optionally uses Tag, Plot, Description as text fields.
      - Normalizes timestamps using a reference date.
      - Converts ratings to either full 10-class labels or binary labels.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        label_map: str = "full",
        inference_mode: bool = False
    ):
        self.df = dataframe.copy()
        self.movie2idx = create_movie_id_map()
        self.label_map = label_map
        self.inference_mode = inference_mode
        # Compute earliest timestamp for month normalization
        self.reference_date = self.df["timestamp"].min().strftime("%Y-%m-%d %H:%M:%S")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        # We assume userId is already a small integer range [1..605].
        # For embedding, we need 0-based indexing => userId - 1 if userId starts at 1
        user_id = int(row["userId"])  # or row["userId"] - 1 if that suits your embedding
        movie_id = row["movieId"]
        movie_idx = self.movie2idx[movie_id]

        # Convert text fields safely
        plot_text = str(row["Plot"]) if pd.notna(row["Plot"]) else ""
        genres_text = row["genres"].replace("|", " ") if pd.notna(row["genres"]) else ""

        # Normalize time
        month_index = timestamp_to_month_index(row["timestamp"], self.reference_date)
        normalized_month = (month_index - 1) / 240.0

        if self.inference_mode:
          return {
            # For logging/analysis, original user/movie IDs:
            "real_user_id": user_id,
            "real_movie_id": movie_id,

            # Tensors for embedding
            "user_ids": torch.tensor(user_id, dtype=torch.long),
            "movie_ids": torch.tensor(movie_idx, dtype=torch.long),

            # Text fields
            "genres_text": genres_text,
            "plot": plot_text,
            "timestamps": torch.tensor(normalized_month, dtype=torch.float)
          }
        else:
          # Convert rating to label
          rating_value = float(row["rating"])
          if self.label_map == "full":
              label_id = LABEL_MAP[rating_value]
          elif self.label_map == "binary":
              label_id = BINARY_LABEL_MAP[rating_value]
          else:
              raise ValueError(f"Unknown label_map type: {self.label_map}")

          return {
              # For logging/analysis, original user/movie IDs:
              "real_user_id": user_id,
              "real_movie_id": movie_id,

              # Tensors for embedding
              "user_ids": torch.tensor(user_id, dtype=torch.long),
              "movie_ids": torch.tensor(movie_idx, dtype=torch.long),

              # Text fields
              "genres_text": genres_text,
              "plot": plot_text,
              "timestamps": torch.tensor(normalized_month, dtype=torch.float),
              "labels": torch.tensor(label_id, dtype=torch.long),
          }
