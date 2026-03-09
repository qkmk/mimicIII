from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import torch
import wfdb
from torch.utils.data import Dataset


CANONICAL_CHANNELS = [
    "II",
    "V",
    "AVR",
    "RESP",
    "PLETH",
    "ABP",
    "CVP",
    "I",
    "III",
    "AVL",
    "AVF",
    "MCL1",
    "PAP",
    "ART",
    "CO2",
]


def _normalize_channel(name: str | None) -> str:
    if not name:
        return ""
    text = str(name).strip().upper()
    return re.sub(r"[^A-Z0-9]+", "", text)


def _coerce_int(value, default: int | None = None) -> int | None:
    if value is None:
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _coerce_codes(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip().upper() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    codes = re.findall(r"[A-Z]?\d{3,5}(?:\.\d+)?", text, flags=re.IGNORECASE)
    if not codes:
        codes = re.split(r"[;,\s]+", text)
    output: list[str] = []
    seen: set[str] = set()
    for code in codes:
        cleaned = str(code).strip().strip(".").upper()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        output.append(cleaned)
    return output


def _load_metadata(path: Path) -> list[dict]:
    items: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            try:
                items.append(json.loads(text))
            except json.JSONDecodeError:
                continue
    return items


def _load_vocab(path: Path) -> dict[str, int]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "code2idx" in data:
        return {str(k).upper(): int(v) for k, v in data["code2idx"].items()}
    return {str(k).upper(): int(v) for k, v in data.items()}


def _pad_truncate(x: np.ndarray, seq_len: int) -> np.ndarray:
    if seq_len <= 0:
        return x
    channels, length = x.shape
    if length >= seq_len:
        return x[:, :seq_len]
    output = np.zeros((channels, seq_len), dtype=x.dtype)
    if length > 0:
        output[:, :length] = x
    return output


def _zscore(x: np.ndarray) -> np.ndarray:
    means = np.nanmean(x, axis=1, keepdims=True)
    stds = np.nanstd(x, axis=1, keepdims=True)
    means = np.where(np.isfinite(means), means, 0.0)
    stds = np.where(np.isfinite(stds) & (stds > 0), stds, 1.0)
    x = (x - means) / stds
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


class WFDBDataset(Dataset):
    """WFDB waveform dataset with channel alignment and multi-hot labels."""

    def __init__(
        self,
        metadata_path: str | Path,
        label_vocab_path: str | Path,
        wfdb_root: str | Path,
        seq_len: int,
        canonical_channels: list[str] | None = None,
    ) -> None:
        self.metadata_path = Path(metadata_path)
        self.label_vocab_path = Path(label_vocab_path)
        self.wfdb_root = Path(wfdb_root)
        self.seq_len = int(seq_len)

        self.metadata = _load_metadata(self.metadata_path)
        self.code2idx = _load_vocab(self.label_vocab_path)
        self.num_labels = len(self.code2idx)

        self.canonical_channels = canonical_channels or CANONICAL_CHANNELS
        self.num_channels = len(self.canonical_channels)
        self._canonical_norm = [_normalize_channel(ch) for ch in self.canonical_channels]
        self._canonical_map = {name: idx for idx, name in enumerate(self._canonical_norm) if name}

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int):
        item = self.metadata[idx]
        record = str(item.get("record", "") or "")
        sampfrom = _coerce_int(item.get("sampfrom"), 0) or 0
        sampto = _coerce_int(item.get("sampto"))

        x, channel_mask = self._load_signal(record, sampfrom, sampto)
        x = _pad_truncate(x, self.seq_len)
        finite = np.isfinite(x)
        if not finite.any():
            x = np.zeros_like(x)
        else:
            missing = ~finite.any(axis=1)
            if missing.any():
                x = x.copy()
                x[missing] = 0.0
            x = _zscore(x)

        y = np.zeros(self.num_labels, dtype=np.float32)
        for code in _coerce_codes(item.get("icd9")):
            label_idx = self.code2idx.get(code)
            if label_idx is not None:
                y[label_idx] = 1.0

        return (
            torch.from_numpy(x).float(),
            torch.from_numpy(y).float(),
            torch.from_numpy(channel_mask).float(),
        )

    def _resolve_record_path(self, record: str) -> str:
        text = record.strip()
        if not text:
            return ""
        if text.lower().endswith(".hea") or text.lower().endswith(".dat"):
            text = text[:-4]
        path = Path(text)
        if path.is_absolute() or re.match(r"^[a-zA-Z]:", text):
            return str(path)
        return str(self.wfdb_root / path)

    def _load_signal(
        self,
        record: str,
        sampfrom: int,
        sampto: int | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load a WFDB segment aligned to canonical channels."""
        empty = np.zeros((self.num_channels, 0), dtype=np.float32)
        mask = np.zeros(self.num_channels, dtype=np.float32)
        if not record:
            return empty, mask

        record_path = self._resolve_record_path(record)
        sampfrom = max(sampfrom, 0)
        sampto_read = sampto + 1 if sampto is not None and sampto >= sampfrom else None

        try:
            record_obj = wfdb.rdrecord(record_path, sampfrom=sampfrom, sampto=sampto_read)
        except Exception:
            return empty, mask

        sig = record_obj.p_signal if record_obj.p_signal is not None else record_obj.d_signal
        if sig is None:
            return empty, mask

        sig = np.asarray(sig, dtype=np.float32)
        if sig.ndim == 1:
            sig = sig[:, None]
        sig = sig.T

        sig_names = record_obj.sig_name or []
        sig_norm = [_normalize_channel(name) for name in sig_names]
        name_to_idx: dict[str, int] = {}
        for i, name in enumerate(sig_norm):
            if name and name not in name_to_idx:
                name_to_idx[name] = i

        aligned = np.zeros((self.num_channels, sig.shape[1]), dtype=np.float32)
        for canonical_name, channel_idx in self._canonical_map.items():
            src_idx = name_to_idx.get(canonical_name)
            if src_idx is None or src_idx >= sig.shape[0]:
                continue
            aligned[channel_idx] = sig[src_idx]
            mask[channel_idx] = 1.0

        return aligned, mask
