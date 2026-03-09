from __future__ import annotations

import torch
from torch import nn

from .icd_classifier import ICDClassifier
from .patch_embed import TimeSeriesPatchEmbedding
from .ts_encoder import TimeSeriesEncoder


class TimeSeriesModel(nn.Module):
    """Patch-based time-series encoder with ICD classifier head."""

    def __init__(
        self,
        num_channels: int,
        num_labels: int,
        patch_len: int = 128,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        seq_len: int = 4096,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if seq_len <= 0:
            raise ValueError("seq_len must be positive.")
        self.num_channels = int(num_channels)
        self.patch_len = int(patch_len)
        self.patch_embed = TimeSeriesPatchEmbedding(
            in_channels=num_channels,
            d_model=d_model,
            patch_len=self.patch_len,
            stride=self.patch_len,
        )
        max_tokens = max(1, seq_len // self.patch_len)
        self.channel_embed = nn.Embedding(self.num_channels, d_model)
        self.encoder = TimeSeriesEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            max_tokens=max_tokens,
            dropout=dropout,
        )
        self.classifier = ICDClassifier(d_model=d_model, num_labels=num_labels)

    def _add_channel_embedding(self, tokens: torch.Tensor, channel_mask: torch.Tensor | None) -> torch.Tensor:
        if channel_mask is None:
            channel_mask = torch.ones(
                tokens.size(0),
                self.num_channels,
                device=tokens.device,
                dtype=tokens.dtype,
            )
        else:
            channel_mask = channel_mask.to(dtype=tokens.dtype, device=tokens.device)
        channel_ids = torch.arange(self.num_channels, device=tokens.device)
        channel_vectors = self.channel_embed(channel_ids)
        denom = channel_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = (channel_mask @ channel_vectors) / denom
        return tokens + pooled[:, None, :]

    def forward(
        self,
        x: torch.Tensor,
        channel_mask: torch.Tensor | None = None,
        return_embeddings: bool = False,
    ) -> torch.Tensor:
        tokens = self.patch_embed(x)
        tokens = self._add_channel_embedding(tokens, channel_mask)
        tokens = self.encoder(tokens)
        if return_embeddings:
            return tokens
        return self.classifier(tokens)
