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
from src.movie_dataset import MovieRatingDataset, LABEL_MAP, BINARY_LABEL_MAP
from src.utils import save_model, load_and_merge_data, split_data_by_time

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_one_epoch(model: nn.Module, dataloader: DataLoader, optimizer: optim.Optimizer, loss_fn: nn.Module) -> float:
    model.train()
    total_loss = 0.0

    # Wrap dataloader with tqdm progress bar
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for batch in progress_bar:
        user_ids = batch["user_ids"].to(DEVICE)
        movie_ids = batch["movie_ids"].to(DEVICE)
        genre_ids = batch["genres_text"]
        plot_ids = batch["plot"]
        timestamps = batch["timestamps"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        optimizer.zero_grad()
        logits = model(user_ids, movie_ids, genre_ids, plot_ids, timestamps)
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
    use_time_encoding: bool,
    use_plot_embedder: bool,
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

    # Initialize model
    model = RewardModel(user_vocab_size,
                        movie_vocab_size,
                        embed_dim=embed_dim,
                        hidden_dim=hidden_dim,
                        num_labels=num_labels,
                        num_layers=num_layers,
                        num_heads=num_heads,
                        use_time_encoding=use_time_encoding,
                        use_plot_embedder=use_plot_embedder,
                        ).to(DEVICE)

    if num_labels == 10:
        inv_label_map = {v: k for k, v in LABEL_MAP.items()}
    elif num_labels == 2:
        inv_label_map = {v: k for k, v in BINARY_LABEL_MAP.items()}        

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
        train_df = pd.DataFrame(best_train_predictions, columns=["userId", "movieId", "predicted_label", "true_label"])
        # Remap movie_id and user_id to original values
        
        train_df["predicted_label"] = train_df["predicted_label"].map(inv_label_map)
        train_df["true_label"] = train_df["true_label"].map(inv_label_map)
        train_df.to_csv(os.path.join(output_dir, "best_train_predictions.csv"), index=False)
        print(f"Best model training predictions saved to {output_dir}/best_train_predictions.csv")

    if best_val_predictions:
        val_df = pd.DataFrame(best_val_predictions, columns=["userId", "movieId", "predicted_label", "true_label"])
        # Remap movie_id and user_id to original values
        val_df["predicted_label"] = val_df["predicted_label"].map(inv_label_map)
        val_df["true_label"] = val_df["true_label"].map(inv_label_map)
        val_df.to_csv(os.path.join(output_dir, "best_val_predictions.csv"), index=False)
        print(f"Best model validation predictions saved to {output_dir}/best_val_predictions.csv")

@torch.no_grad()
def evaluate(
    model: nn.Module, 
    dataloader: DataLoader, 
    loss_fn: nn.Module,
    save_predictions: bool = False
) -> Tuple[float, float, Optional[List[List[Any]]]]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    predictions = []

    progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)
    for batch in progress_bar:
        user_ids = batch["user_ids"].to(DEVICE)
        movie_ids = batch["movie_ids"].to(DEVICE)
        genres_text = batch["genres_text"]   
        plot_text = batch["plot"]            #
        timestamps = batch["timestamps"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        logits = model(
            user_ids=user_ids,
            movie_ids=movie_ids,
            genres=genres_text,
            plot=plot_text,
            timestamps=timestamps
        )

        loss = loss_fn(logits, labels)
        preds = torch.argmax(logits, dim=-1)

        total_loss += loss.item()
        correct += (preds == labels).sum().item()
        total += len(labels)

        if save_predictions:
            real_uids = batch["real_user_id"].tolist()
            real_mids = batch["real_movie_id"].tolist()

            for i in range(len(labels)):
                predictions.append([
                    real_uids[i],
                    real_mids[i],
                    preds[i].item(),
                    labels[i].item()
                ])

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total if total > 0 else 0
    return avg_loss, accuracy, predictions if save_predictions else None

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
    parser.add_argument("--use_time_encoding", action="store_true", default=False, help="Setting time encoding embedding")
    parser.add_argument("--use_plot_embedder", action="store_true", default=False, help="Setting plot encoding embedding")
    args = parser.parse_args()

    # Processing use_* labels
    if args.use_time_encoding:
        use_time_encoding = True
    else:
        use_time_encoding = False

    if args.use_plot_embedder:
        use_plot_embedder= True
    else:
        use_plot_embedder = False

    if args.num_labels == 10:
        label_map = "full"
    elif args.num_labels == 2:
        label_map = "binary"
    else:
        # Create an exception error
        raise ValueError(f"Unknown label_map type: {label_map}")

    # Load data
    df = load_and_merge_data()

    user_vocab_size = df["userId"].nunique() + 1
    movie_vocab_size = df["movieId"].nunique() + 1

    train_df, val_df = split_data_by_time(df)
    print(f"User vocab size: {user_vocab_size}, Movie vocab size: {movie_vocab_size}")

    # Create datasets
    train_dataset = MovieRatingDataset(train_df, label_map=label_map)
    val_dataset = MovieRatingDataset(val_df, label_map=label_map)
    print(f"Train dataset size: {len(train_dataset)}, Val dataset size: {len(val_dataset)}")

    # Train model
    train_reward_model(train_dataset,
                       val_dataset,
                       args.output_dir,
                       user_vocab_size,
                       movie_vocab_size,
                       use_time_encoding=use_time_encoding,
                       use_plot_embedder=use_plot_embedder,
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
