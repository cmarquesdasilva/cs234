import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import os
import argparse
from tqdm import tqdm
from typing import List, Tuple, Dict, Any, Optional
from src.reward_model import RewardModel
from src.movie_dataset import MovieRatingDataset
from src.utils import save_model, load_and_merge_data, split_data_by_time

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_dicts(df_list: List[pd.DataFrame]) -> Tuple[Dict[int, int], Dict[int, int], Dict[str, int]]:
    USER_TO_ID: Dict[int, int] = {}
    MOVIE_TO_ID: Dict[int, int] = {}
    GENRE_TO_ID: Dict[str, int] = {}

    for df in df_list:
        for _, row in df.iterrows():
            # Map userId to integer
            user_id = row["userId"]
            if user_id not in USER_TO_ID:
                USER_TO_ID[user_id] = len(USER_TO_ID) + 1

            # Map movieId to integer
            movie_id = row["movieId"]
            if movie_id not in MOVIE_TO_ID:
                MOVIE_TO_ID[movie_id] = len(MOVIE_TO_ID) + 1

            # Map genre names to integer
            genres = row["genres"].split("|")
            for g in genres:
                if g not in GENRE_TO_ID:
                    GENRE_TO_ID[g] = len(GENRE_TO_ID) + 1

    return USER_TO_ID, MOVIE_TO_ID, GENRE_TO_ID

def train_one_epoch(model: nn.Module, dataloader: DataLoader, optimizer: optim.Optimizer, loss_fn: nn.Module) -> float:
    model.train()
    total_loss = 0.0

    # Wrap dataloader with tqdm progress bar
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for batch in progress_bar:
        user_ids = batch["user_ids"].to(DEVICE)
        movie_ids = batch["movie_ids"].to(DEVICE)
        genre_ids = batch["genre_ids"].to(DEVICE)
        timestamps = batch["timestamps"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        optimizer.zero_grad()
        logits = model(user_ids, movie_ids, genre_ids, timestamps)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(dataloader)

def train_reward_model(
    train_dataset: MovieRatingDataset, 
    val_dataset: MovieRatingDataset,        
    output_dir: str, 
    user_vocab_size: int, 
    movie_vocab_size: int, 
    genre_vocab_size: int, 
    num_labels: int = 10,
    embed_dim: int = 64,
    hidden_dim: int = 128,
    num_layers: int = 4,
    num_heads: int = 4,
    epochs: int = 5, 
    batch_size: int = 32, 
    lr: float = 1e-4, 
) -> None:
    """
    Train a Reward Model using the provided datasets.
    params:
        - train_dataset: Training dataset
        - val_dataset: Validation dataset
        - user_vocab_size: Number of unique users
        - movie_vocab_size: Number of unique movies
        - genre_vocab_size: Number of unique genres
        - output_dir: Directory to save model checkpoints
        - epochs: Number of training epochs
        - batch_size: Training batch size
        - lr: Learning rate
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Get vocab sizes
    user_vocab_size = user_vocab_size + 1
    movie_vocab_size = movie_vocab_size + 1
    genre_vocab_size = genre_vocab_size + 1

    # Initialize model
    model = RewardModel(user_vocab_size,
                        movie_vocab_size,
                        genre_vocab_size,
                        embed_dim=embed_dim,
                        hidden_dim=hidden_dim,
                        num_labels=num_labels,
                        num_layers=num_layers,
                        num_heads=num_heads
                        ).to(DEVICE)

    # Define loss function & optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    best_val_acc = 0.0
    best_train_predictions: List[List[Any]] = []
    best_val_predictions: List[List[Any]] = []

    # Training loop
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn)
        _, train_acc, train_predictions = evaluate(model, train_loader, loss_fn, save_predictions=True)
        val_loss, val_acc, val_predictions = evaluate(model, val_loader, loss_fn, save_predictions=True)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Save the best model and predictions for analysis
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_train_predictions = train_predictions
            best_val_predictions = val_predictions
            save_model(model, os.path.join(output_dir, "best"))
            print("Best model saved.")

    # Save final version of the model
    save_model(model, os.path.join(output_dir, "final"))
    print("Best val acc: ", best_val_acc)

    # Save best model predictions
    if best_train_predictions:
        train_df = pd.DataFrame(best_train_predictions, columns=["user_id", "movie_id", "predicted_label", "true_label"])
        train_df.to_csv(os.path.join(output_dir, "best_train_predictions.csv"), index=False)
        print(f"Best model training predictions saved to {output_dir}/best_train_predictions.csv")

    if best_val_predictions:
        val_df = pd.DataFrame(best_val_predictions, columns=["user_id", "movie_id", "predicted_label", "true_label"])
        val_df.to_csv(os.path.join(output_dir, "best_val_predictions.csv"), index=False)
        print(f"Best model validation predictions saved to {output_dir}/best_val_predictions.csv")

def evaluate(model: nn.Module, dataloader: DataLoader, loss_fn: nn.Module, save_predictions: bool = False) -> Tuple[float, float, Optional[List[List[Any]]]]:
    """ Evaluate the model on the provided dataset.
    params:
        - model: Reward Model
        - dataloader: DataLoader for the dataset
        - loss_fn: Loss function
        - save_predictions: Whether to save predictions
    
    returns:
        - Average loss
        - Accuracy
        - Predictions (if save_predictions=True)
    """
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    predictions: List[List[Any]] = []

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)
        for batch in progress_bar:
            user_ids = batch["user_ids"].to(DEVICE)
            movie_ids = batch["movie_ids"].to(DEVICE)
            genre_ids = batch["genre_ids"].to(DEVICE)
            timestamps = batch["timestamps"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            logits = model(user_ids, movie_ids, genre_ids, timestamps)
            loss = loss_fn(logits, labels)
            preds = torch.argmax(logits, dim=-1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

            # Save predictions if needed
            if save_predictions:
                for i in range(len(user_ids)):
                    predictions.append([user_ids[i].item(), movie_ids[i].item(), preds[i].item(), labels[i].item()])

    accuracy = correct / total if total > 0 else 0.0
    return total_loss / len(dataloader), accuracy, predictions if save_predictions else None

def main():
    parser = argparse.ArgumentParser(description="Train Reward Model")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--output_dir", type=str, default="vanilla_reward_model", help="Where to save checkpoints")
    parser.add_argument("--model_path", type=str, default=None, help="Path to checkpoint if continuing training")
    parser.add_argument("--embed_dim", type=int, default=64, help="Size of user & genre embeddings")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of transformer heads")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension of transformer layers")
    parser.add_argument("--num_labels", type=int, default=10, help="Number of labels for classification")
    args = parser.parse_args()

    # Load data
    df = load_and_merge_data()
    train_df, val_df = split_data_by_time(df)

    # Build dictionaries
    USER_TO_ID, MOVIE_TO_ID, GENRE_TO_ID = build_dicts([train_df, val_df])
    user_vocab_size = len(USER_TO_ID)
    movie_vocab_size = len(MOVIE_TO_ID)
    genre_vocab_size = len(GENRE_TO_ID)
    print(genre_vocab_size)

    print(f"User vocab size: {user_vocab_size}, Movie vocab size: {movie_vocab_size}, Genre vocab size: {genre_vocab_size}")

    # Create datasets
    train_dataset = MovieRatingDataset(train_df, USER_TO_ID, MOVIE_TO_ID, GENRE_TO_ID)
    val_dataset = MovieRatingDataset(val_df, USER_TO_ID, MOVIE_TO_ID, GENRE_TO_ID)
    print(f"Train dataset size: {len(train_dataset)}, Val dataset size: {len(val_dataset)}")

    # Train model
    train_reward_model(train_dataset,
                       val_dataset,
                       args.output_dir,
                       user_vocab_size,
                       movie_vocab_size,
                       genre_vocab_size,
                       num_labels=args.num_labels,
                       embed_dim=args.embed_dim,
                       hidden_dim=args.hidden_dim,
                       num_layers=args.num_layers,
                       num_heads = args.num_heads, 
                       epochs=args.epochs,
                       batch_size=args.batch_size,
                       lr=args.lr)
        
if __name__ == "__main__":
    main()
