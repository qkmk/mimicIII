from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ICD9 label vocabulary from metadata JSONL.")
    parser.add_argument("--input", required=True, help="Metadata JSONL path.")
    parser.add_argument("--output", required=True, help="Output vocab JSON path.")
    return parser.parse_args()


def _iter_metadata(path: Path):
    if path.suffix.lower() == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                text = line.strip()
                if not text:
                    continue
                try:
                    yield json.loads(text)
                except json.JSONDecodeError as exc:
                    print(f"Warning: invalid JSONL line {line_num}: {exc}", file=sys.stderr)
        return

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        yield from data
    else:
        yield data


def _coerce_codes(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        codes = [str(item).strip().upper() for item in value if str(item).strip()]
        return codes
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


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    code2idx: dict[str, int] = {}
    idx2code: list[str] = []

    for item in _iter_metadata(input_path):
        codes = _coerce_codes(item.get("icd9") if isinstance(item, dict) else None)
        for code in codes:
            if code in code2idx:
                continue
            code2idx[code] = len(idx2code)
            idx2code.append(code)

    vocab = {"code2idx": code2idx, "idx2code": idx2code}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=True, indent=2)


if __name__ == "__main__":
    main()
