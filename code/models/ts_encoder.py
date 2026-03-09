from __future__ import annotations

import torch
from torch import nn


class TransformerEncoderBlock(nn.Module):
    """Pre-norm transformer block for time-series tokens."""

    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class TimeSeriesEncoder(nn.Module):
    """Stacked transformer encoder with learnable positions."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        max_tokens: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive.")
        self.max_tokens = int(max_tokens)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.max_tokens, d_model))
        self.layers = nn.ModuleList(
            [
                TransformerEncoderBlock(d_model, n_heads, mlp_ratio=4.0, dropout=dropout)
                for _ in range(n_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) > self.max_tokens:
            raise ValueError(f"Sequence length {x.size(1)} exceeds max_tokens {self.max_tokens}.")
        x = x + self.pos_embed[:, : x.size(1), :]
        for layer in self.layers:
            x = layer(x)
        return x
