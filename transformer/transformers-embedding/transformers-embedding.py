import torch
import torch.nn as nn
import math

def create_embedding_layer(vocab_size: int, d_model: int) -> nn.Embedding:
    embedding = nn.Embedding(
        num_embeddings=vocab_size,
        embedding_dim=d_model
    )
    return embedding

def embed_tokens(embedding: nn.Embedding, tokens: torch.Tensor, d_model: int) -> torch.Tensor:
    scaling_factor = math.sqrt(d_model)
    x = embedding(tokens)
    x.shape
    x = x * scaling_factor
    return x