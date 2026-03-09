from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm as _tqdm
except Exception:
    _tqdm = None

from models.ts_model import TimeSeriesModel
from wfdb_dataset import WFDBDataset


def _str2bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a multi-label WFDB time-series classifier.")
    parser.add_argument("--train_metadata", required=True, help="Training metadata JSONL path.")
    parser.add_argument("--val_metadata", required=True, help="Validation metadata JSONL path.")
    parser.add_argument("--label_vocab", required=True, help="Label vocab JSON path.")
    parser.add_argument("--wfdb_root", required=True, help="WFDB root directory.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=30, help="Epochs.")
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping max_norm (0 to disable).")
    parser.add_argument(
        "--gradient_clip_norm",
        type=float,
        default=None,
        help="Alias for --grad_clip (max_norm).",
    )
    parser.add_argument("--scheduler", choices=["none", "plateau", "cosine"], default="plateau", help="LR scheduler.")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum LR for cosine/plateau.")
    parser.add_argument("--t_max", type=int, default=50, help="T_max for cosine annealing.")
    parser.add_argument("--warmup_epochs", type=int, default=0, help="Warmup epochs.")
    parser.add_argument("--seq_len", type=int, default=4096, help="Fixed sequence length.")
    parser.add_argument("--patch_len", type=int, default=128, help="Patch length for Conv1d patchify.")
    parser.add_argument("--d_model", type=int, default=256, help="Embedding dimension.")
    parser.add_argument("--n_heads", type=int, default=8, help="Attention heads.")
    parser.add_argument("--n_layers", type=int, default=4, help="Transformer layers.")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--output", default="best.pt", help="Checkpoint output path.")
    parser.add_argument("--last_output", default=None, help="Last checkpoint path (default: run_dir/last.pt).")
    parser.add_argument("--run_dir", default="runs/ts_transformer", help="Base directory for logs.")
    parser.add_argument("--run_name", default=None, help="Run name (default: timestamp).")
    parser.add_argument("--min_label_freq", type=int, default=20, help="Minimum label frequency to keep.")
    parser.add_argument("--use_pos_weight", type=_str2bool, default=True, help="Use pos_weight for BCE loss.")
    parser.add_argument("--pos_weight_clip_max", type=float, default=10.0, help="Clip pos_weight max value.")
    parser.add_argument("--k", type=int, default=5, help="Top-k for precision/recall.")
    parser.add_argument("--threshold_min", type=float, default=0.1, help="Minimum threshold for search.")
    parser.add_argument("--threshold_max", type=float, default=0.9, help="Maximum threshold for search.")
    parser.add_argument("--threshold_step", type=float, default=0.05, help="Threshold search step.")
    parser.add_argument("--device", default=None, help="Device override, e.g. cpu or cuda.")
    return parser.parse_args()


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


def _load_label_vocab(path: Path) -> tuple[dict[str, int], list[str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "code2idx" in data:
        code2idx = {str(k).upper(): int(v) for k, v in data["code2idx"].items()}
        idx2code = [str(code) for code in data.get("idx2code", [])]
        if not idx2code:
            idx2code = [None] * len(code2idx)
            for code, idx in code2idx.items():
                if 0 <= idx < len(idx2code):
                    idx2code[idx] = code
            idx2code = [code or "" for code in idx2code]
        return code2idx, idx2code
    code2idx = {str(k).upper(): int(v) for k, v in data.items()}
    idx2code = [""] * len(code2idx)
    for code, idx in code2idx.items():
        if 0 <= idx < len(idx2code):
            idx2code[idx] = code
    return code2idx, idx2code


def _scan_metadata(
    metadata_path: Path,
) -> tuple[dict[str, int], int, int, float]:
    counts: dict[str, int] = {}
    total_samples = 0
    samples_with_any = 0
    total_labels = 0
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            try:
                item = json.loads(text)
            except json.JSONDecodeError:
                continue
            total_samples += 1
            codes = _coerce_codes(item.get("icd9") if isinstance(item, dict) else None)
            unique = set(codes)
            if unique:
                samples_with_any += 1
            total_labels += len(unique)
            for code in unique:
                counts[code] = counts.get(code, 0) + 1
    avg_labels = total_labels / total_samples if total_samples > 0 else 0.0
    return counts, total_samples, samples_with_any, avg_labels


def _compute_filtered_sample_stats(metadata_path: Path, keep_set: set[str]) -> tuple[int, float]:
    total_samples = 0
    covered_samples = 0
    kept_labels = 0
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            try:
                item = json.loads(text)
            except json.JSONDecodeError:
                continue
            total_samples += 1
            codes = _coerce_codes(item.get("icd9") if isinstance(item, dict) else None)
            unique = set(codes)
            kept = [code for code in unique if code in keep_set]
            if kept:
                covered_samples += 1
            kept_labels += len(kept)
    avg_kept = kept_labels / total_samples if total_samples > 0 else 0.0
    return covered_samples, avg_kept


def _build_filtered_vocab(
    label_vocab_path: Path,
    counts: dict[str, int],
    min_label_freq: int,
    output_path: Path,
) -> tuple[dict[str, int], list[str]]:
    _, idx2code = _load_label_vocab(label_vocab_path)
    kept_codes = [code for code in idx2code if counts.get(code, 0) >= min_label_freq]
    if not kept_codes:
        raise ValueError("No labels left after filtering. Lower min_label_freq.")
    code2idx = {code: idx for idx, code in enumerate(kept_codes)}
    vocab = {"code2idx": code2idx, "idx2code": kept_codes}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=True, indent=2)
    return code2idx, kept_codes


def _log_label_stats(counts: np.ndarray, idx2code: list[str], total: int) -> None:
    if total <= 0:
        return
    pairs = [(idx2code[i] if i < len(idx2code) else str(i), int(counts[i])) for i in range(len(counts))]
    pairs.sort(key=lambda x: x[1], reverse=True)
    head = pairs[:10]
    tail = [item for item in pairs if item[1] > 0][-10:]
    print("Label frequency head:", ", ".join(f"{code}:{cnt}" for code, cnt in head))
    print("Label frequency tail:", ", ".join(f"{code}:{cnt}" for code, cnt in tail))


def _save_label_stats(
    path: Path,
    counts: np.ndarray,
    pos_weight: np.ndarray | None,
    idx2code: list[str],
    total: int,
    min_label_freq: int,
) -> None:
    label_counts = {}
    for idx, count in enumerate(counts):
        code = idx2code[idx] if idx < len(idx2code) else str(idx)
        label_counts[code] = int(count)
    payload = {
        "total_samples": int(total),
        "min_label_freq": int(min_label_freq),
        "label_counts": label_counts,
    }
    if pos_weight is not None:
        payload["pos_weight"] = {
            idx2code[idx] if idx < len(idx2code) else str(idx): float(pos_weight[idx])
            for idx in range(len(counts))
        }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)


def _iter_with_progress(loader: Iterable, desc: str) -> Iterator:
    if _tqdm is not None:
        return _tqdm(loader, desc=desc, leave=False)
    try:
        total = len(loader)
    except TypeError:
        total = None
    return _simple_progress(loader, desc, total)


def _simple_progress(iterable: Iterable, desc: str, total: int | None) -> Iterator:
    if not total:
        for item in iterable:
            yield item
        return
    width = 30
    for idx, item in enumerate(iterable, 1):
        filled = int(width * idx / total)
        bar = "#" * filled + "-" * (width - filled)
        print(f"\r{desc} [{bar}] {idx}/{total}", end="", flush=True)
        yield item
    print()


def _micro_f1(tp: float, fp: float, fn: float) -> float:
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom > 0 else 0.0


def _f1_from_preds(preds: torch.Tensor, targets: torch.Tensor) -> tuple[float, float, list[float]]:
    tp = (preds & targets).sum(dim=0).float()
    fp = (preds & (~targets)).sum(dim=0).float()
    fn = ((~preds) & targets).sum(dim=0).float()
    denom = 2 * tp + fp + fn
    per_label = torch.where(denom > 0, (2 * tp / denom), torch.zeros_like(denom))
    micro = _micro_f1(tp.sum().item(), fp.sum().item(), fn.sum().item())
    macro = per_label.mean().item() if per_label.numel() else 0.0
    return micro, macro, per_label.cpu().tolist()


def _precision_recall_at_k(probs: torch.Tensor, targets: torch.Tensor, k: int) -> tuple[float, float]:
    if k <= 0:
        return 0.0, 0.0
    k = min(k, probs.size(1))
    topk = torch.topk(probs, k=k, dim=1).indices
    hits = targets.gather(1, topk).float().sum(dim=1)
    precision = (hits / k).mean().item()
    denom = targets.sum(dim=1).clamp(min=1).float()
    recall = (hits / denom).mean().item()
    return precision, recall


def _auc_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int32)
    y_score = y_score.astype(np.float64)
    pos = y_true.sum()
    neg = y_true.size - pos
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(y_score, kind="mergesort")
    scores = y_score[order]
    labels = y_true[order]
    ranks = np.empty_like(scores, dtype=np.float64)
    i = 0
    n = scores.size
    while i < n:
        j = i
        while j + 1 < n and scores[j + 1] == scores[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        ranks[i : j + 1] = avg_rank
        i = j + 1
    sum_pos = ranks[labels == 1].sum()
    return (sum_pos - pos * (pos + 1) / 2.0) / (pos * neg)


def _micro_macro_auroc(probs: torch.Tensor, targets: torch.Tensor) -> tuple[float, float]:
    y_true = targets.cpu().numpy().astype(np.int32)
    y_score = probs.cpu().numpy().astype(np.float64)
    micro = _auc_binary(y_true.reshape(-1), y_score.reshape(-1))
    macro_values = []
    for idx in range(y_true.shape[1]):
        auc = _auc_binary(y_true[:, idx], y_score[:, idx])
        if not math.isnan(auc):
            macro_values.append(auc)
    macro = float(np.mean(macro_values)) if macro_values else float("nan")
    return micro, macro


def _threshold_search(
    probs: torch.Tensor,
    targets: torch.Tensor,
    t_min: float,
    t_max: float,
    t_step: float,
) -> tuple[float, float]:
    best_t = 0.5
    best_f1 = -1.0
    t = t_min
    while t <= t_max + 1e-9:
        preds = probs >= t
        micro, _, _ = _f1_from_preds(preds, targets)
        if micro > best_f1:
            best_f1 = micro
            best_t = t
        t += t_step
    return best_t, best_f1


def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    threshold_min: float,
    threshold_max: float,
    threshold_step: float,
    k: int,
) -> tuple[dict, torch.Tensor, torch.Tensor]:
    total_loss = 0.0
    total_count = 0
    all_logits: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    model.eval()
    with torch.no_grad():
        for x, y, channel_mask in _iter_with_progress(loader, desc="val"):
            x = x.to(device)
            y = y.to(device)
            channel_mask = channel_mask.to(device)
            logits = model(x, channel_mask=channel_mask)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            total_count += x.size(0)
            all_logits.append(logits.detach().cpu())
            all_targets.append(y.detach().cpu())

    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0).bool()
    probs = torch.sigmoid(logits)

    best_t, best_micro_f1 = _threshold_search(probs, targets, threshold_min, threshold_max, threshold_step)
    preds = probs >= best_t
    micro_f1, macro_f1, per_label_f1 = _f1_from_preds(preds, targets)
    micro_auroc, macro_auroc = _micro_macro_auroc(probs, targets)
    precision_k, recall_k = _precision_recall_at_k(probs, targets, k)

    metrics = {
        "val_loss": total_loss / max(total_count, 1),
        "val_micro_f1": micro_f1,
        "val_macro_f1": macro_f1,
        "val_micro_auroc": micro_auroc,
        "val_macro_auroc": macro_auroc,
        "val_precision_at_k": precision_k,
        "val_recall_at_k": recall_k,
        "val_best_threshold": best_t,
        "val_best_micro_f1": best_micro_f1,
        "val_per_label_f1": per_label_f1,
    }
    return metrics, probs, targets


def _run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
) -> float:
    total_loss = 0.0
    total_count = 0
    model.train()

    for x, y, channel_mask in _iter_with_progress(loader, desc="train"):
        x = x.to(device)
        y = y.to(device)
        channel_mask = channel_mask.to(device)
        optimizer.zero_grad()
        logits = model(x, channel_mask=channel_mask)
        loss = criterion(logits, y)
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        total_count += x.size(0)

    return total_loss / max(total_count, 1)


def _write_metrics_csv(path: Path, history: list[dict]) -> None:
    fieldnames = [
        "epoch",
        "train_loss",
        "val_loss",
        "val_micro_f1",
        "val_macro_f1",
        "val_micro_auroc",
        "val_macro_auroc",
        "val_precision_at_k",
        "val_recall_at_k",
        "val_best_threshold",
        "lr",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow({k: row.get(k) for k in fieldnames})


def _append_results_summary(path: Path, row: dict) -> None:
    fieldnames = list(row.keys())
    exists = path.exists()
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _save_curves(path: Path, history: list[dict], key_pairs: list[tuple[str, str]], title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("Warning: matplotlib not available, skipping plots.")
        return
    epochs = [row["epoch"] for row in history]
    plt.figure()
    for key, label in key_pairs:
        values = [row.get(key) for row in history]
        plt.plot(epochs, values, label=label)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _error_analysis(
    probs: torch.Tensor,
    targets: torch.Tensor,
    threshold: float,
    idx2code: list[str],
) -> dict:
    preds = probs >= threshold
    tp = (preds & targets).sum(dim=0)
    fp = (preds & (~targets)).sum(dim=0)
    fn = ((~preds) & targets).sum(dim=0)
    support = targets.sum(dim=0)
    denom = (2 * tp + fp + fn).float()
    per_label_f1 = torch.where(denom > 0, (2 * tp.float() / denom), torch.zeros_like(denom)).cpu().tolist()

    fp_counts = fp.cpu().numpy()
    fn_counts = fn.cpu().numpy()
    support_counts = support.cpu().numpy()

    def top_items(values: np.ndarray, n: int = 10) -> list[dict]:
        indices = np.argsort(values)[::-1][:n]
        return [
            {"label": idx2code[i] if i < len(idx2code) else str(i), "count": int(values[i])}
            for i in indices
        ]

    support_order = np.argsort(support_counts)[::-1]
    high_support = [
        {
            "label": idx2code[i] if i < len(idx2code) else str(i),
            "support": int(support_counts[i]),
            "f1": float(per_label_f1[i]),
        }
        for i in support_order[:10]
    ]
    low_support = [
        {
            "label": idx2code[i] if i < len(idx2code) else str(i),
            "support": int(support_counts[i]),
            "f1": float(per_label_f1[i]),
        }
        for i in support_order[::-1][:10]
    ]

    return {
        "false_positive_labels": top_items(fp_counts, 10),
        "false_negative_labels": top_items(fn_counts, 10),
        "high_support_labels": high_support,
        "low_support_labels": low_support,
        "per_label_f1": per_label_f1,
    }


def _apply_warmup(optimizer: torch.optim.Optimizer, base_lr: float, epoch: int, warmup_epochs: int) -> None:
    if warmup_epochs <= 0:
        return
    if epoch > warmup_epochs:
        return
    scale = epoch / warmup_epochs
    for group in optimizer.param_groups:
        group["lr"] = base_lr * scale


def main() -> None:
    args = parse_args()
    if args.gradient_clip_norm is not None:
        args.grad_clip = args.gradient_clip_norm
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_name = args.run_name or time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.run_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output)
    last_output_path = Path(args.last_output) if args.last_output else run_dir / "last.pt"

    counts_map, total_samples, samples_with_any, avg_labels_before = _scan_metadata(Path(args.train_metadata))
    raw_code2idx, _ = _load_label_vocab(Path(args.label_vocab))
    labels_in_train = sum(1 for code in counts_map if counts_map[code] > 0)
    filtered_vocab_path = run_dir / "label_vocab.filtered.json"
    code2idx, idx2code = _build_filtered_vocab(
        Path(args.label_vocab), counts_map, args.min_label_freq, filtered_vocab_path
    )
    keep_set = set(idx2code)
    covered_samples, avg_labels_after = _compute_filtered_sample_stats(Path(args.train_metadata), keep_set)
    print(
        "Label filtering: "
        f"vocab={len(raw_code2idx)} train_labels={labels_in_train} "
        f"after={len(code2idx)} min_label_freq={args.min_label_freq}"
    )
    print(
        f"Coverage: before={samples_with_any}/{total_samples} "
        f"after={covered_samples}/{total_samples}"
    )
    print(
        f"Avg labels per sample: before={avg_labels_before:.3f} after={avg_labels_after:.3f}"
    )

    train_dataset = WFDBDataset(
        metadata_path=args.train_metadata,
        label_vocab_path=filtered_vocab_path,
        wfdb_root=args.wfdb_root,
        seq_len=args.seq_len,
    )
    val_dataset = WFDBDataset(
        metadata_path=args.val_metadata,
        label_vocab_path=filtered_vocab_path,
        wfdb_root=args.wfdb_root,
        seq_len=args.seq_len,
    )

    if train_dataset.num_labels == 0:
        raise ValueError("Label vocab is empty.")

    counts = np.array([counts_map.get(code, 0) for code in idx2code], dtype=np.int64)
    pos_weight = (total_samples - counts) / np.clip(counts, 1, None)
    if args.use_pos_weight:
        pos_weight = np.minimum(pos_weight, args.pos_weight_clip_max)
    _log_label_stats(counts, idx2code, total_samples)
    _save_label_stats(
        run_dir / "label_frequency.json",
        counts,
        pos_weight if args.use_pos_weight else None,
        idx2code,
        total_samples,
        args.min_label_freq,
    )
    if args.use_pos_weight:
        pw_min, pw_mean, pw_max = float(pos_weight.min()), float(pos_weight.mean()), float(pos_weight.max())
        print(
            f"pos_weight stats: min={pw_min:.3f} mean={pw_mean:.3f} max={pw_max:.3f} "
            f"(clip_max={args.pos_weight_clip_max:.1f})"
        )
    else:
        print("pos_weight disabled.")

    model = TimeSeriesModel(
        num_channels=train_dataset.num_channels,
        num_labels=train_dataset.num_labels,
        patch_len=args.patch_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        seq_len=args.seq_len,
    )
    model.to(device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    pos_weight_stats = None
    if args.use_pos_weight:
        pos_weight_stats = (float(pos_weight.min()), float(pos_weight.mean()), float(pos_weight.max()))
        pos_weight_t = torch.tensor(pos_weight, dtype=torch.float32, device=device)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)
    else:
        criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = None
    if args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=2, min_lr=args.min_lr
        )
    elif args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.t_max, eta_min=args.min_lr)

    history: list[dict] = []
    best_f1 = -1.0
    best_epoch = 0
    patience_left = args.patience
    label_count_after = len(code2idx)

    metrics_jsonl = run_dir / "metrics.jsonl"
    with open(metrics_jsonl, "w", encoding="utf-8") as _:
        pass

    for epoch in range(1, args.epochs + 1):
        _apply_warmup(optimizer, args.lr, epoch, args.warmup_epochs)
        train_loss = _run_epoch(model, train_loader, criterion, optimizer, device, args.grad_clip)
        val_metrics, probs, targets = _evaluate(
            model,
            val_loader,
            criterion,
            device,
            args.threshold_min,
            args.threshold_max,
            args.threshold_step,
            args.k,
        )
        lr = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["val_loss"],
            "val_micro_f1": val_metrics["val_micro_f1"],
            "val_macro_f1": val_metrics["val_macro_f1"],
            "val_micro_auroc": val_metrics["val_micro_auroc"],
            "val_macro_auroc": val_metrics["val_macro_auroc"],
            "val_precision_at_k": val_metrics["val_precision_at_k"],
            "val_recall_at_k": val_metrics["val_recall_at_k"],
            "val_best_threshold": val_metrics["val_best_threshold"],
            "lr": lr,
        }
        history.append(row)
        with open(metrics_jsonl, "a", encoding="utf-8") as f:
            payload = dict(row)
            payload["val_best_micro_f1"] = val_metrics["val_best_micro_f1"]
            payload["val_per_label_f1"] = val_metrics["val_per_label_f1"]
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")

        print(
            f"Epoch {epoch}/{args.epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_metrics['val_loss']:.4f} "
            f"val_micro_f1={val_metrics['val_micro_f1']:.4f} val_macro_f1={val_metrics['val_macro_f1']:.4f}"
        )
        if pos_weight_stats is not None:
            pw_min, pw_mean, pw_max = pos_weight_stats
            pos_weight_text = f"{pw_min:.3f}/{pw_mean:.3f}/{pw_max:.3f}"
        else:
            pos_weight_text = "disabled"
        print(
            "Diag: "
            f"labels_after={label_count_after} avg_labels={avg_labels_after:.3f} "
            f"pos_weight(min/mean/max)={pos_weight_text} "
            f"threshold={val_metrics['val_best_threshold']:.2f} "
            f"monitor=val_micro_f1({val_metrics['val_micro_f1']:.4f})"
        )

        if val_metrics["val_micro_f1"] > best_f1:
            best_f1 = val_metrics["val_micro_f1"]
            best_epoch = epoch
            patience_left = args.patience
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "code2idx": train_dataset.code2idx,
                    "seq_len": args.seq_len,
                    "patch_len": args.patch_len,
                    "d_model": args.d_model,
                    "n_heads": args.n_heads,
                    "n_layers": args.n_layers,
                    "min_label_freq": args.min_label_freq,
                    "use_pos_weight": args.use_pos_weight,
                    "pos_weight_clip_max": args.pos_weight_clip_max,
                },
                output_path,
            )
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "code2idx": train_dataset.code2idx,
                    "seq_len": args.seq_len,
                    "patch_len": args.patch_len,
                    "d_model": args.d_model,
                    "n_heads": args.n_heads,
                    "n_layers": args.n_layers,
                    "min_label_freq": args.min_label_freq,
                    "use_pos_weight": args.use_pos_weight,
                    "pos_weight_clip_max": args.pos_weight_clip_max,
                },
                last_output_path,
            )
            error_report = _error_analysis(
                probs, targets, val_metrics["val_best_threshold"], idx2code
            )
            with open(run_dir / "error_analysis.json", "w", encoding="utf-8") as f:
                json.dump(error_report, f, ensure_ascii=True, indent=2)
        else:
            patience_left -= 1
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "code2idx": train_dataset.code2idx,
                    "seq_len": args.seq_len,
                    "patch_len": args.patch_len,
                    "d_model": args.d_model,
                    "n_heads": args.n_heads,
                    "n_layers": args.n_layers,
                    "min_label_freq": args.min_label_freq,
                    "use_pos_weight": args.use_pos_weight,
                    "pos_weight_clip_max": args.pos_weight_clip_max,
                },
                last_output_path,
            )

        if scheduler is not None and epoch > args.warmup_epochs:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics["val_micro_f1"])
            else:
                scheduler.step()

        if patience_left <= 0:
            print(f"Early stopping at epoch {epoch} (best epoch {best_epoch}).")
            break

    _write_metrics_csv(run_dir / "metrics.csv", history)
    _save_curves(
        run_dir / "loss_curve.png",
        history,
        [("train_loss", "train_loss"), ("val_loss", "val_loss")],
        "Loss Curve",
    )
    _save_curves(
        run_dir / "f1_curve.png",
        history,
        [("val_micro_f1", "val_micro_f1"), ("val_macro_f1", "val_macro_f1")],
        "F1 Curve",
    )

    results_row = {
        "run_name": run_name,
        "best_epoch": best_epoch,
        "best_val_micro_f1": best_f1,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "grad_clip": args.grad_clip,
        "min_label_freq": args.min_label_freq,
        "use_pos_weight": args.use_pos_weight,
        "pos_weight_clip_max": args.pos_weight_clip_max,
        "scheduler": args.scheduler,
        "seq_len": args.seq_len,
        "patch_len": args.patch_len,
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "run_dir": str(run_dir),
        "output": str(output_path),
        "last_output": str(last_output_path),
    }
    _append_results_summary(Path("results_summary.csv"), results_row)


if __name__ == "__main__":
    main()
