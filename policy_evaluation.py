import os
import random
from types import SimpleNamespace

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.policy import PolicyModel, MovieRankingEnv, RankZero

def load_policy(policy_path: str,
                device: str = "cpu",
                embedding_dim: int = 384,
                action_dim: int = 25,
                hidden_dim: int = 128) -> PolicyModel:
    """
    Loads a trained PolicyModel from the specified directory.
    
    Args:
        policy_path: Directory containing the saved model file (model.bin).
        device: The device to load the model onto ("cpu" or "cuda").
        embedding_dim: Dimension of the movie embeddings expected by the policy.
        action_dim: Number of possible actions (e.g. 25 for Top-25 movies).
        hidden_dim: Dimension used for the hidden layers in the policy/value networks.
        
    Returns:
        A PolicyModel instance with loaded weights.
        
    Raises:
        FileNotFoundError: If the model file doesn't exist at the specified path.
    """
    # Instantiate the policy model architecture
    policy_model = PolicyModel(
        embedding_dim=embedding_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim
    )

    # Construct full path to the saved model file
    model_file = os.path.join(policy_path, "baseline_pytorch_model.bin")
    if not os.path.isfile(model_file):
        raise FileNotFoundError(f"Model file not found at: {model_file}")

    # Load the state dictionary and move it to the specified device
    state_dict = torch.load(model_file, map_location=device)
    policy_model.load_state_dict(state_dict)
    policy_model.to(device)
    return policy_model

def evaluate_policy_comparison(movies_path: str, 
                               ratings_path: str, 
                               policy_path: str) -> None:
    """
    Compare the trained PPO policy against a random policy by running
    the random selection multiple times for each user to compute an average reward.
    
    Args:
        movies_path: Path to CSV containing movie information.
        ratings_path: Path to CSV containing user ratings.
        policy_path: Directory where the trained policy model is saved (model.bin).
        num_random_samples: Number of times to sample a random action per user for averaging.
        
    Returns:
        None. Generates and saves a comparison plot as 'random_vs_policy_model.png'.
    """
    # Environment configuration
    env_config = SimpleNamespace(
        device="cuda" if torch.cuda.is_available() else "cpu",
        user_vocab_size=611,
        movie_vocab_size=9743
    )

    # Initialize RankZero and environment
    ranker = RankZero(movies_path, ratings_path, use_gpu=False)
    env = MovieRankingEnv(ranker, env_config)

    # Load trained policy model
    policy_model = load_policy(
        policy_path,
        env_config.device)
    policy_model.eval()

    # Prepare lists to store rewards
    all_rewards_policy = []

    # Retrieve user IDs
    user_ids = sorted(ranker.user_fav_genres["userId"].unique())

    # Evaluate for each user, showing progress with tqdm
    for uid in tqdm(user_ids, desc="Evaluating Policies"):
        # Step 1: Generate Top-25 candidate movies
        top25 = env.get_top25_movies(uid)
        state = np.stack(top25['embedding'].values)
        state_tensor = torch.tensor(state, dtype=torch.float32).to(env_config.device)

        # ==== Policy-based action ====
        with torch.no_grad():
            action_probs, _ = policy_model(state_tensor)
        policy_dist = torch.distributions.Categorical(action_probs)
        policy_action = policy_dist.sample()
        policy_reward = env.get_rewards(uid, top25.iloc[[policy_action.item()]], env_config)

        # Collect rewards
        all_rewards_policy.append(policy_reward)

    # ==== Plot the comparison as a line plot ====
    plt.figure(figsize=(10, 5))

    # Plot policy rewards
    plt.plot(user_ids, all_rewards_policy, label="Policy", color="blue", marker='o')

    # Plot average random rewards
    plt.axhline(y=4.5, color='red', linestyle='--', label=" Random Policy (4.5)")

    plt.xlabel("User ID")
    plt.ylabel("Reward")
    plt.title("Comparison: Policy vs. Random (Averaged) Rewards")
    plt.legend()
    plt.tight_layout()
    plt.savefig("random_vs_policy_model.png")

if __name__ == "__main__":
    evaluate_policy_comparison(
        movies_path="src/data/movies_spec.csv",
        ratings_path="src/data/ratings.csv",
        policy_path="src/policy_model"
    )
