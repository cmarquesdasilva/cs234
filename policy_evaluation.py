import torch
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace

from src.policy import PolicyModel, MovieRankingEnv, RankZero

def load_policy(policy_path: str,
                device: str = "cpu",
                embedding_dim: int = 384,
                action_dim: int = 25,
                hidden_dim: int = 128) -> PolicyModel:
    """
    Loads a trained PolicyModel from the specified directory.

    :param policy_path: Directory containing the saved model file (e.g. "model.bin").
    :param device: The device to load the model onto (e.g. "cpu" or "cuda").
    :param embedding_dim: Dimension of the movie embeddings expected by the policy.
    :param action_dim: Number of possible actions (e.g. 25 for Top-25 movies).
    :param hidden_dim: Dimension used for the hidden layers in the policy/value networks.
    :return: A PolicyModel instance with loaded weights.
    """
    # Instantiate the policy model architecture
    policy_model = PolicyModel(
        embedding_dim=embedding_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim
    )

def evaluate_policy(movies_path, ratings_path, policy_path):
    # Initialize RankZero
    env_config = SimpleNamespace(
        device="cuda" if torch.cuda.is_available() else "cpu",
        user_vocab_size=611,
        movie_vocab_size=9725
    )

    ranker = RankZero(movies_path, ratings_path, use_gpu=(env_config.device=="cuda"))
    env = MovieRankingEnv(ranker, env_config)

    # Load trained policy
    policy_model = load_policy(policy_path,
                              env_config.device,
                              embedding_dim=384,
                              action_dim=25)
    policy_model.eval()

    all_rewards = []
    user_ids = sorted(ranker.user_fav_genres["userId"].unique())

    # Loop through each user
    for uid in user_ids:
        # Retrieve top-25 candidate movies
        top25 = env.get_top25_movies(uid)
        state = np.stack(top25['embedding'].values)
        state_tensor = torch.tensor(state, dtype=torch.float32).to(env_config.device)

        # Get action probabilities from policy
        with torch.no_grad():
            action_probs, _ = policy_model(state_tensor)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()

        # Compute reward
        reward = env.get_rewards(uid, top25.iloc[[action.item()]], env_config)
        all_rewards.append(reward)

    # Plot userId vs reward
    plt.figure(figsize=(10, 4))
    plt.scatter(user_ids, all_rewards, alpha=0.7)
    plt.xlabel("User ID")
    plt.ylabel("Reward")
    plt.title("Policy Rewards per User")
    plt.savefig("policy_rewards.png")

if __name__ == "__main__":
    evaluate_policy(
        movies_path="src/data/movies_spec.csv",
        ratings_path="src/data/ratings.csv",
        policy_path="src/policy_model/baseline"  # Folder where the policy is saved
    )