from __future__ import annotations

import torch
from torch import nn


class ICDClassifier(nn.Module):
    """MLP head for pooled embeddings."""

    def __init__(self, d_model: int, num_labels: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_labels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)
