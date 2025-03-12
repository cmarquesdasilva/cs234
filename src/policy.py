# Ranker Zero
import pandas as pd
import numpy as np
import os

import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import faiss
from transformers import AutoTokenizer, AutoModel

from tqdm import tqdm
from src.movie_dataset import MovieRatingDataset, LABEL_MAP
from torch.utils.data import DataLoader
from src.utils import load_model, inference, save_model
from src.monitoring import log_training

import numpy as np
from torch.distributions import Categorical

class RankZero:
    def __init__(self, movies_path, ratings_path,
    processed_movie_catalog_path="src/data/movie_catalog.csv",
    processed_user_fav_path="src/data/user_fav_genres.csv",
    use_gpu=False, 
    seed=42):
        """Initialize RankZero and retrieve top 25 movies for user"""
        
        self.use_gpu = use_gpu
        self.rating_df = pd.read_csv(ratings_path)
        self.device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
        self.seed = seed  # Set a static seed
        
        # Load or generate movie catalog
        if os.path.exists(processed_movie_catalog_path):
            self.movie_catalog = pd.read_csv(processed_movie_catalog_path)
        else:
            self.movie_catalog = self.create_movie_catalog(movies_path)
            self.movie_catalog.to_csv(processed_movie_catalog_path, index=False)

        # Load or generate user favorite genres
        if os.path.exists(processed_user_fav_path):
            self.user_fav_genres = pd.read_csv(processed_user_fav_path)
        else:
            self.user_fav_genres = self.create_user_fav_genres(movies_path)
            self.user_fav_genres.to_csv(processed_user_fav_path, index=False)

        # Load tokenizer and model for embedding generation
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.embedding_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(self.device)
        
        # Cache for user recommendations
        self.user_top25_cache = {}
        
        # Generate and store embeddings
        self.generate_movie_embeddings()
    
    def create_movie_catalog(self, movies_path):
        """Create the movie catalog with popularity scores."""
        movies = pd.read_csv(movies_path)  # movieId, genres, plot, imdbVotes
        
        # Compute IMDb popularity score
        movies['imdbVotes'] = movies['imdbVotes'].replace(',', '', regex=True)
        movies['imdbVotes'] = pd.to_numeric(movies['imdbVotes'], errors='coerce')
        total_votes = movies['imdbVotes'].sum()
        movies['popularityScore'] = movies['imdbVotes'] / total_votes
        
        # Improve popularity score range using log transformation
        movies['popularityScore'] = np.log1p(movies['popularityScore'] * 1e6)
        return movies
    
    def create_user_fav_genres(self, movies_path):
        """Create user favorite genres dataset."""
        movies = pd.read_csv(movies_path)  # movieId, title, genres
        
        # Merge ratings with movie genres
        user_movie_genres = self.rating_df.merge(movies[['movieId', 'genres']], on='movieId', how='left')
        
        # Expand genres into separate rows
        user_movie_genres['genres'] = user_movie_genres['genres'].str.split('|')
        user_movie_genres = user_movie_genres.explode('genres')
        
        # Get top-rated genres per user
        top_genres = (user_movie_genres.groupby(['userId', 'genres'])
                      .agg({'rating': 'mean'})
                      .reset_index()
                      .sort_values(['userId', 'rating'], ascending=[True, False]))
        
        # Select top 5 genres per user
        user_fav_genres = top_genres.groupby('userId')['genres'].apply(lambda x: x.head(5).tolist()).reset_index()
        user_fav_genres[['genre1', 'genre2', 'genre3', 'genre4', 'genre5']] = pd.DataFrame(user_fav_genres['genres'].tolist(), index=user_fav_genres.index)
        user_fav_genres = user_fav_genres.drop(columns=['genres'])
        
        return user_fav_genres

    def generate_movie_embeddings(self):
        """Generate and store embeddings for all movies."""
        movie_texts = self.movie_catalog.apply(lambda row: row['Plot'] + " " + row['genres'], axis=1).tolist()
        
        inputs = self.tokenizer(movie_texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            movie_embeddings = self.embedding_model(**inputs).last_hidden_state[:, 0, :].cpu().numpy()
        
        self.movie_catalog['embedding'] = list(movie_embeddings)
    
    def get_top25_movies(self, user_id):
        """Retrieve the top 25 movies based on user preferences and similarity."""
        
        if user_id in self.user_top25_cache:
            return self.user_top25_cache[user_id]
        
        np.random.seed(self.seed)  # Ensure consistent Top-100 selection
        
        rated_movies = set(self.rating_df[self.rating_df['userId'] == user_id]['movieId'].tolist())
        unseen_movies = self.movie_catalog[~self.movie_catalog['movieId'].isin(rated_movies)]
        
        # Select top 100 movies based on genre preferences
        user_prefs = self.user_fav_genres[self.user_fav_genres['userId'] == user_id].iloc[0]
        favorite_genres = user_prefs[1:].dropna().tolist()
        
        genre_filtered_movies = unseen_movies[
            unseen_movies['genres'].apply(lambda x: any(g in x.split("|") for g in favorite_genres))
        ]

        top_100_movies = genre_filtered_movies.sample(n=min(100, len(genre_filtered_movies)), random_state=self.seed)
        
        # Compute user preference embedding
        user_movie_spec = self.rating_df[self.rating_df['userId'] == user_id].merge(self.movie_catalog, on='movieId')
        user_movie_spec = user_movie_spec.sort_values(by='rating', ascending=False)
        highly_rated_movies = user_movie_spec.iloc[:25]
        
        highly_rated_texts = highly_rated_movies.apply(lambda row: row['Plot'] + " " + row['genres'], axis=1).tolist()
        inputs = self.tokenizer(highly_rated_texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            highly_rated_embeddings = self.embedding_model(**inputs).last_hidden_state[:, 0, :]
        
        user_embedding = highly_rated_embeddings.mean(dim=0).cpu().numpy()
        
        # Compute similarity using FAISS
        top_100_embeddings = np.stack(top_100_movies['embedding'].values)
        
        dimension = top_100_embeddings.shape[1]
        if self.use_gpu:
            res = faiss.StandardGpuResources()
            index = faiss.IndexFlatL2(dimension)
            index = faiss.index_cpu_to_gpu(res, 0, index)
        else:
            index = faiss.IndexFlatL2(dimension)
        
        index.add(top_100_embeddings)
        _, top_indices = index.search(np.expand_dims(user_embedding, axis=0), 25)
        
        top_25_movies = top_100_movies.iloc[top_indices[0]]

        # Cache results
        self.user_top25_cache[user_id] = top_25_movies
        
        return top_25_movies


class MovieRankingEnv:
    def __init__(self, ranker, config):
        """Environment to interact with RankZero for policy training."""
        self.ranker = ranker
        self.config = config
    
    def sample_user(self):
        """Randomly sample a user from the dataset."""
        return self.ranker.user_fav_genres['userId'].sample().values[0]
    
    def get_top25_movies(self, user_id):
        """Retrieve top 25 movies for the user."""
        return self.ranker.get_top25_movies(user_id)
    
    def get_rewards(self, user_id:int, movies: pd.DataFrame, config: dict, reward_type='tf'):
        """Generate reward scores for ranking."""
        
        # Build Datasetmovies = 
        movies.copy()
        movies.loc[:, "userId"] = user_id
        movies.loc[:, "timestamp"] = pd.to_datetime("now")
   
        movie_dataset = MovieRatingDataset(
            movies,
            inference_mode=True
        )
        device = self.config.device
        movie_dataloader = DataLoader(movie_dataset, batch_size=8, shuffle=False)
        
        # Load Model
        model = load_model("vanilla_reward_model/best/",
                self.config.user_vocab_size,
                self.config.movie_vocab_size,
                device,
                use_time_encoding=False,
                use_plot_embedder=True)
        
        # Run Model
        predictions = inference(model=model,
        dataloader=movie_dataloader,
        device=device)
        score = predictions[0][-1]
        inv_label = {v: k for k,v in LABEL_MAP.items()}
        return inv_label[score]


class PolicyModel(nn.Module):
    """
    Processes a (25, embedding_dim) movie-embedding matrix by
    aggregating into a single state vector and then applying
    a fully connected neural net.
    """
    def __init__(self, embedding_dim=384, action_dim=25, hidden_dim=128):
        super(PolicyModel, self).__init__()
        
        # Policy network
        self.policy = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Value network
        self.value = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Single scalar value
        )

    def forward(self, state: torch.Tensor):
        """
        state: shape (25, embedding_dim)
               i.e., 25 movie embeddings for a given user or situation.
        
        We do a simple mean-pool so that we have a single
        embedding vector (embedding_dim,).
        """
        # Aggregate 25 embeddings into a single vector
        # shape becomes (embedding_dim,)
        aggregated = state.mean(dim=0)

        # Pass aggregated embedding through the policy network
        action_probs = self.policy(aggregated)  # shape: (25,)

        # Also pass through the value network for the baseline
        state_value = self.value(aggregated)    # shape: (1,)

        return action_probs, state_value


class PPOTrainer:
    def __init__(self, env, action_dim=25, lr=3e-4, gamma=0.99, epsilon=0.2, epochs=10, batch_size=64):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.epochs = epochs
        self.batch_size = batch_size
        
        self.policy_model = PolicyModel().to(env.config.device)
        self.optimizer = optim.Adam(self.policy_model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        
    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(self.env.config.device)
        action_probs, _ = self.policy_model(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)
    
    def compute_returns(self, rewards):
        returns = []
        G = 0
        for r in reversed(rewards):
          G = r + self.gamma * G
          returns.insert(0, G)
        return torch.tensor(returns, dtype=torch.float32)
    
    def train(self, num_episodes=1000):
      
      for episode in range(num_episodes):
        user_id = self.env.sample_user()
        top_movies = self.env.get_top25_movies(user_id)

        # Send all 25 movie embeddings as input to the policy
        movie_features = np.stack(top_movies['embedding'].values)

        # Get action probabilities and sample an action
        action, log_prob = self.select_action(movie_features)
        
        # Compute reward for the selected action
        reward = self.env.get_rewards(user_id, top_movies.iloc[[action]], self.env.config)
        
        # Convert to tensors
        state = torch.tensor(movie_features, dtype=torch.float32).to(self.env.config.device)
        action = torch.tensor([action], dtype=torch.int64).to(self.env.config.device)
        log_prob = torch.tensor([log_prob], dtype=torch.float32).to(self.env.config.device)
        return_value = torch.tensor([reward], dtype=torch.float32).to(self.env.config.device)
        
        # PPO optimization step
        for _ in range(self.epochs):
            action_probs, state_value = self.policy_model(state)
            
            dist = Categorical(action_probs)
            new_log_prob = dist.log_prob(action)
            
            ratio = torch.exp(new_log_prob - log_prob.detach())
            advantage = return_value - state_value.squeeze()
            
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage
            
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = self.loss_fn(state_value.view(-1), return_value.view(-1))
            
            loss = policy_loss + 0.5 * value_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        if episode % 10 == 0:
            print(f"Episode {episode}: Loss {loss.item()} Reward {reward}")
            log_training(episode, loss.item(), reward)
      save_model(self.policy_model, "src/policy_model","baseline")

