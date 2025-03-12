import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# Load MiniLM model and tokenizer
class TextEmbedder(nn.Module):
    def __init__(self, embedding_dim=64, mlp_hidden_dim=64):
        super(TextEmbedder, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.encoder = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

        # MLP to reduce dimensionality from 384 (MiniLM) to embedding_dim
        self.mlp = nn.Sequential(
            nn.Linear(384, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, embedding_dim)
        )

    def forward(self, text):
        """
        :param tags: List[str] or single string
        :return: Tensor of shape (batch_size, embedding_dim)
        """
        if isinstance(text, str):
            text = [text]  

        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            outputs = self.encoder(**inputs)

        embeddings = outputs.last_hidden_state[:, 0, :]
        return self.mlp(embeddings)


class RewardModel(nn.Module):
    def __init__(
        self, 
        user_vocab_size: int, 
        movie_vocab_size: int, 
        embed_dim: int = 64, 
        hidden_dim: int = 128, 
        num_labels: int = 10, 
        num_layers: int = 8, 
        num_heads: int = 4,
        use_time_encoding: bool = False,
        use_plot_embedder: bool = False,
    ) -> None:
        super().__init__()

        self.num_labels = num_labels
        self.user_embedding = nn.Embedding(user_vocab_size, embed_dim)
        self.movie_embedding = nn.Embedding(movie_vocab_size, embed_dim)

        self.use_plot_embedder = use_plot_embedder

        if self.use_plot_embedder:
            print("Model with plot embedder")
            self.plot_embedder = TextEmbedder(embedding_dim=embed_dim)
        else:   
            print("Model without plot embedder")

        self.genre_embedder = TextEmbedder(embedding_dim=embed_dim)

        self.use_time_encoding = use_time_encoding

        if self.use_time_encoding:
            print("Using Time encoding...")
            self.time_encoding = TimeEncoding(embed_dim)
        else:
            print("Model without time encoding....")

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
            nn.Linear(hidden_dim, self.num_labels)
        )

    def forward(self, user_ids: torch.Tensor,
                movie_ids: torch.Tensor,
                genres: list,
                plot: list,
                timestamps: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for Reward Model.
        - user_ids: (B,)
        - movie_ids: (B,)
        - genre_ids: (B, num_genres) - Multiple genres per movie
        - timestamps: (B,) - Time when rating was given (normalized), optional if use_time_encoding is False
        """
        user_embed = self.user_embedding(user_ids)
        movie_embed = self.movie_embedding(movie_ids)
        
        genre_embeds = self.genre_embedder(genres)

        if self.use_plot_embedder:
            plot_embeds = self.plot_embedder(plot)
            movie_embed += plot_embeds

        # Sum movie and genre embeddings
        movie_genre_representation = movie_embed + genre_embeds

        if self.use_time_encoding:
            time_embedding = self.time_encoding(timestamps)
            transformer_input = torch.stack([user_embed, movie_genre_representation, time_embedding], dim=1)
        else:
            transformer_input = torch.stack([user_embed, movie_genre_representation], dim=1)

        # Pass through Transformer Encoder
        transformer_output = self.encoder(transformer_input)

        # Compute the mean
        attn_output = transformer_output.mean(dim=1)

        # Pass through FFN to predict rating class
        logits = self.ffn(attn_output)
        
        return logits
