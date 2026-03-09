from __future__ import annotations

import torch
from torch import nn


class SimpleCnn(nn.Module):
    """Basic 1D CNN for multi-label classification."""

    def __init__(self, in_channels: int, num_labels: int, hidden_channels: list[int] | None = None) -> None:
        super().__init__()
        hidden_channels = hidden_channels or [32, 64, 128]

        layers: list[nn.Module] = []
        prev = in_channels
        kernels = [7, 5, 3]
        for idx, out_ch in enumerate(hidden_channels):
            kernel = kernels[idx] if idx < len(kernels) else 3
            padding = kernel // 2
            layers.append(nn.Conv1d(prev, out_ch, kernel_size=kernel, padding=padding))
            layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.ReLU())
            prev = out_ch

        self.encoder = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(prev, num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)
