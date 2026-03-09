from __future__ import annotations

import torch
from torch import nn


class TimeSeriesPatchEmbedding(nn.Module):
    """Patchify time-series with a Conv1d projection."""

    def __init__(self, in_channels: int, d_model: int, patch_len: int, stride: int | None = None) -> None:
        super().__init__()
        if patch_len <= 0:
            raise ValueError("patch_len must be positive.")
        stride = stride or patch_len
        self.patch_len = int(patch_len)
        self.stride = int(stride)
        self.proj = nn.Conv1d(
            in_channels=in_channels,
            out_channels=d_model,
            kernel_size=self.patch_len,
            stride=self.stride,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return tokens as [B, N, d_model]."""
        x = self.proj(x)
        return x.transpose(1, 2)
