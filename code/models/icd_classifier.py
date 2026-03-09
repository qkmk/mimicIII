from __future__ import annotations

import torch
from torch import nn


class ICDClassifier(nn.Module):
    """Pool token embeddings and predict ICD labels."""

    def __init__(self, d_model: int, num_labels: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(d_model, num_labels)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = tokens.transpose(1, 2)
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)
