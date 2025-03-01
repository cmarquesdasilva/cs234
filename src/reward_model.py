import torch
import torch.nn as nn

class TimeEncoding(nn.Module):
    """
    Fourier-based encoding for continuous time representation.
    Uses multiple frequencies to capture different time scales.
    """
    def __init__(self, embed_dim: int, num_frequencies: int = 4) -> None:
        super().__init__()
        self.num_frequencies = num_frequencies
        self.freqs = nn.Parameter(torch.randn(num_frequencies) * 2 * torch.pi)
        self.phases = nn.Parameter(torch.randn(num_frequencies))  
        self.linear = nn.Linear(num_frequencies * 2, embed_dim)

    def forward(self, normalized_months: torch.Tensor) -> torch.Tensor:
        """
        normalized_months: (B,) - Normalized month indices in range [0,1].
        Returns: Fourier-based time encoding (B, embed_dim)
        """
        time_matrix = normalized_months.unsqueeze(-1) * self.freqs
        fourier_features = torch.cat([torch.sin(time_matrix + self.phases), torch.cos(time_matrix + self.phases)], dim=-1)
        return self.linear(fourier_features)


class LearnableGenreAggregation(nn.Module):
    """
    Learns importance weights for multiple genres dynamically.
    Uses attention-like mechanism to aggregate genre embeddings.
    """
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.weight_fc = nn.Linear(embed_dim, 1)

    def forward(self, genre_embeds: torch.Tensor) -> torch.Tensor:
        """
        genre_embeds: (B, num_genres, embed_dim)
        Returns: weighted genre representation (B, embed_dim)
        """
        attn_scores = self.weight_fc(genre_embeds).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)
        weighted_genre_representation = (attn_weights * genre_embeds).sum(dim=1)
        return weighted_genre_representation


class RewardModel(nn.Module):
    def __init__(self, user_vocab_size: int, movie_vocab_size: int, genre_vocab_size: int, embed_dim: int = 64, hidden_dim: int = 128, num_labels: int = 10, num_layers: int = 4, num_heads: int = 4) -> None:
        super().__init__()

        self.user_embedding = nn.Embedding(user_vocab_size, embed_dim)
        self.movie_embedding = nn.Embedding(movie_vocab_size, embed_dim)
        self.genre_embedding = nn.Embedding(genre_vocab_size, embed_dim)

        self.genre_aggregation = LearnableGenreAggregation(embed_dim)

        self.time_encoding = TimeEncoding(embed_dim)

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,   
            nhead=num_heads,     
            dim_feedforward=hidden_dim,  
            batch_first=True 
        )

        self.encoder = nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=num_layers,  
            norm=nn.LayerNorm(embed_dim)
        )

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_labels)
        )

    def forward(self, user_ids: torch.Tensor, movie_ids: torch.Tensor, genre_ids: torch.Tensor, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Reward Model.
        - user_ids: (B,)
        - movie_ids: (B,)
        - genre_ids: (B, num_genres) - Multiple genres per movie
        - timestamps: (B,) - Time when rating was given (normalized)
        """
        user_embed = self.user_embedding(user_ids)
        movie_embed = self.movie_embedding(movie_ids)
        genre_embeds = self.genre_embedding(genre_ids)

        # Sum movie and genre embeddings
        genre_representation = self.genre_aggregation(genre_embeds)  
        movie_genre_representation = movie_embed + genre_representation

        # Compute Fourier-Based Time Encoding
        time_embedding = self.time_encoding(timestamps)

        # Create sequence for Transformer Encoder
        transformer_input = torch.stack([user_embed, movie_genre_representation, time_embedding], dim=1)

        # Pass through Transformer Encoder
        transformer_output = self.encoder(transformer_input)

        # Compute the mean
        attn_output = transformer_output.mean(dim=1)

        # Pass through FFN to predict rating class
        logits = self.ffn(attn_output)
        
        return logits