from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path

import wfdb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export ShareGPT-style JSONL from matched_diagnoses.csv.")
    parser.add_argument(
        "--input",
        default=None,
        help="Path to matched_diagnoses.csv (default: <repo>/mimic3wdb-matched/matched_diagnoses.csv)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL path (default: stdout)",
    )
    return parser.parse_args()


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(text, fmt)
            except ValueError:
                continue
    return None


def _format_number(value: float | int | None) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return "null"
        if value.is_integer():
            return str(int(value))
        return format(value, ".15g")
    return str(value)


def _normalize_record_path(record_path: str | None, base_dir: Path) -> str:
    if not record_path:
        return ""
    text = str(record_path).strip().replace("\\", "/")
    text = text[:-4] if text.lower().endswith(".hea") else text
    text = text[:-4] if text.lower().endswith(".dat") else text

    p = Path(text)
    if p.is_absolute() or re.match(r"^[a-zA-Z]:", text):
        parts = [part for part in p.parts if part not in (p.anchor, "")]
        lower_parts = [part.lower() for part in parts]
        if "mimic3wdb-matched" in lower_parts:
            idx = lower_parts.index("mimic3wdb-matched")
            rel_parts = parts[idx + 1 :]
            return Path(*rel_parts).as_posix()
        try:
            rel = p.relative_to(base_dir)
            return rel.as_posix()
        except ValueError:
            if len(parts) >= 3:
                return Path(*parts[-3:]).as_posix()
            return Path(p.name).as_posix()

    return p.as_posix().lstrip("./")


def _infer_record_type(record_path: str) -> str:
    base = os.path.basename(record_path)
    if base.endswith("n"):
        return "numeric"
    if re.search(r"p\d{6}-\d{4}-\d{2}-\d{2}-\d{2}-\d{2}", base):
        return "waveform"
    return "segment"


def _read_header(
    cache: dict[str, wfdb.Header | None],
    base_dir: Path,
    record_path: str,
) -> wfdb.Header | None:
    if record_path in cache:
        return cache[record_path]
    try:
        hdr = wfdb.rdheader(str(base_dir / record_path))
    except Exception:
        hdr = None
    cache[record_path] = hdr
    return hdr


def _get_channels(
    cache: dict[str, wfdb.Header | None],
    base_dir: Path,
    record_path: str,
    hdr: wfdb.Header | None,
) -> list[str]:
    if hdr and hdr.sig_name:
        return list(hdr.sig_name)

    if not hdr or not getattr(hdr, "seg_name", None):
        return []

    parent_dir = Path(record_path).parent
    seg_names = list(hdr.seg_name)
    seg_lens = list(hdr.seg_len) if getattr(hdr, "seg_len", None) is not None else []

    for name, seg_len in zip(seg_names, seg_lens):
        if name == "~":
            continue
        if int(seg_len) == 0:
            layout_path = (parent_dir / name).as_posix()
            layout_hdr = _read_header(cache, base_dir, layout_path)
            if layout_hdr and layout_hdr.sig_name:
                return list(layout_hdr.sig_name)

    for name in seg_names:
        if name == "~":
            continue
        seg_path = (parent_dir / name).as_posix()
        seg_hdr = _read_header(cache, base_dir, seg_path)
        if seg_hdr and seg_hdr.sig_name:
            return list(seg_hdr.sig_name)

    return []


def _build_human_value(
    record: str,
    record_type: str,
    channels: list[str],
    fs: float | None,
    t0: str | None,
    duration_s: float | None,
    sampfrom: int | None,
    sampto: int | None,
) -> str:
    channels_text = json.dumps(channels, ensure_ascii=False)
    t0_text = t0.strip() if t0 and str(t0).strip().lower() != "nan" else "null"
    duration_text = _format_number(duration_s)
    fs_text = _format_number(fs)
    sampfrom_text = _format_number(sampfrom)
    sampto_text = _format_number(sampto)

    return (
        "Given the ICU time-series segment, output the patient's diagnosis list.\n"
        "<ts>\n"
        f"record: {record}\n"
        f"record_type: {record_type}\n"
        f"channels: {channels_text}\n"
        f"fs: {fs_text}\n"
        f"t0: {t0_text}\n"
        f"duration_s: {duration_text}\n"
        f"sampfrom: {sampfrom_text}\n"
        f"sampto: {sampto_text}\n"
        "</ts>"
    )


def _build_gpt_value(icd9_codes: str | None, diag_titles: str | None) -> str:
    icd_text = (icd9_codes or "").strip()
    diag_text = (diag_titles or "").strip()
    return f"ICD9: {icd_text}. Diagnoses: {diag_text}"


def _open_output(path: str | None, stack: ExitStack):
    if not path:
        sys.stdout.reconfigure(encoding="utf-8", newline="\n")
        return sys.stdout
    return stack.enter_context(open(path, "w", encoding="utf-8", newline="\n"))


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    default_input = repo_root / "mimic3wdb-matched" / "matched_diagnoses.csv"
    input_path = Path(args.input) if args.input else default_input
    base_dir = input_path.resolve().parent

    header_cache: dict[str, wfdb.Header | None] = {}

    with open(input_path, "r", encoding="utf-8-sig", newline="") as csv_f, ExitStack() as stack:
        out_f = _open_output(args.output, stack)
        reader = csv.DictReader(csv_f)
        for idx, row in enumerate(reader):
            raw_record_path = row.get("record_path")
            record_path = _normalize_record_path(raw_record_path, base_dir)
            record_id = record_path or f"row_{idx}"

            record_type = (row.get("record_type") or "").strip() or _infer_record_type(record_path)

            hdr = _read_header(header_cache, base_dir, record_path) if record_path else None
            channels = _get_channels(header_cache, base_dir, record_path, hdr) if record_path else []
            fs = float(getattr(hdr, "fs", None)) if hdr and getattr(hdr, "fs", None) is not None else None
            sig_len = int(hdr.sig_len) if hdr and getattr(hdr, "sig_len", None) is not None else None

            start_text = row.get("start_time")
            end_text = row.get("end_time")
            start_dt = _parse_datetime(start_text)
            end_dt = _parse_datetime(end_text)

            duration_s = None
            if start_dt and end_dt:
                duration_s = (end_dt - start_dt).total_seconds()
            elif fs is not None and sig_len is not None:
                duration_s = (sig_len - 1) / fs

            sampfrom = 0 if sig_len is not None else None
            sampto = sig_len - 1 if sig_len is not None else None

            human_value = _build_human_value(
                record=record_path,
                record_type=record_type,
                channels=channels,
                fs=fs,
                t0=start_text,
                duration_s=duration_s,
                sampfrom=sampfrom,
                sampto=sampto,
            )

            gpt_value = _build_gpt_value(row.get("icd9_codes"), row.get("diag_titles"))

            obj = {
                "id": record_id,
                "conversations": [
                    {"from": "system", "value": "You are a medical assistant."},
                    {"from": "human", "value": human_value},
                    {"from": "gpt", "value": gpt_value},
                ],
            }
            out_f.write(json.dumps(obj, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
