import pandas as pd
from src.policy import MovieRankingEnv, RankZero, PPOTrainer 
import torch
import torch.nn as nn
from types import SimpleNamespace

def main():
    # Load environment
    movies_path = "src/data/movies_spec.csv"
    ratings_path = "src/data/ratings.csv"
    env_config = SimpleNamespace(device="cuda" if torch.cuda.is_available() else "cpu",
    user_vocab_size=611,
    movie_vocab_size=9725)
    
    ranker = RankZero(movies_path, ratings_path, use_gpu=False)
    env = MovieRankingEnv(ranker, config=env_config)
    
    # Define state & action dimensions
    action_dim = 25  # Top-25 movies per user
    
    # Initialize PPO trainer
    trainer = PPOTrainer(env, action_dim)
    
    # Train the policy
    trainer.train(num_episodes=1000)
    
if __name__ == "__main__":
    main()