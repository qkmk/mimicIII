from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse ShareGPT-style JSON/JSONL into metadata JSONL.")
    parser.add_argument("--input", required=True, help="Path to ShareGPT JSON or JSONL.")
    parser.add_argument("--output", required=True, help="Output metadata JSONL path.")
    return parser.parse_args()


def _read_json_objects(path: Path):
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
    elif isinstance(data, dict) and isinstance(data.get("data"), list):
        yield from data["data"]
    else:
        yield data


def _find_conversation_value(conversations, role: str) -> str:
    role_lower = role.lower()
    for turn in conversations or []:
        if str(turn.get("from", "")).lower() == role_lower:
            return str(turn.get("value", "") or "")
    for turn in conversations or []:
        if str(turn.get("role", "")).lower() == role_lower:
            return str(turn.get("value", "") or "")
    return ""


def _extract_ts_fields(text: str) -> dict[str, str]:
    """Extract key/value fields from the <ts>...</ts> block."""
    if not text:
        return {}
    match = re.search(r"<ts>\s*(.*?)\s*</ts>", text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return {}
    block = match.group(1)
    fields: dict[str, str] = {}
    for line in block.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip().lower()
        value = value.strip()
        if key:
            fields[key] = value
    return fields


def _parse_channels(value: str | None) -> list[str]:
    if not value:
        return []
    text = str(value).strip()
    if not text or text.lower() in {"null", "none", "nan"}:
        return []
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [str(item) for item in data if str(item).strip()]
        if isinstance(data, str):
            text = data
    except Exception:
        pass
    text = text.strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    parts = [part.strip().strip("'\"") for part in text.split(",")]
    return [part for part in parts if part]


def _parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"null", "none", "nan"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _parse_int(value: str | None) -> int | None:
    num = _parse_float(value)
    if num is None:
        return None
    try:
        return int(num)
    except (TypeError, ValueError):
        return None


def _extract_icd9(text: str) -> list[str]:
    """Parse ICD9 codes from the assistant response."""
    if not text:
        return []
    match = re.search(r"ICD9[^:]*:\s*(.*)", text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return []
    rest = match.group(1).strip()
    rest = re.split(r"\bDiagnoses?\b\s*:", rest, flags=re.IGNORECASE)[0]
    rest = rest.strip().strip(".")
    codes = re.findall(r"[A-Z]?\d{3,5}(?:\.\d+)?", rest, flags=re.IGNORECASE)
    if not codes:
        codes = re.split(r"[;,\s]+", rest)
    seen: set[str] = set()
    output: list[str] = []
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

    with open(output_path, "w", encoding="utf-8") as out_f:
        for idx, obj in enumerate(_read_json_objects(input_path)):
            record_id = obj.get("id") or f"row_{idx}"
            conversations = obj.get("conversations") if isinstance(obj, dict) else []
            human_value = _find_conversation_value(conversations, "human")
            gpt_value = _find_conversation_value(conversations, "gpt")

            fields = _extract_ts_fields(human_value)
            record = fields.get("record", "")
            channels = _parse_channels(fields.get("channels"))
            fs = _parse_float(fields.get("fs"))
            sampfrom = _parse_int(fields.get("sampfrom"))
            sampto = _parse_int(fields.get("sampto"))
            icd9 = _extract_icd9(gpt_value)

            if not record:
                print(f"Warning: missing record for {record_id}", file=sys.stderr)

            meta = {
                "id": record_id,
                "record": record,
                "channels": channels,
                "fs": fs,
                "sampfrom": sampfrom,
                "sampto": sampto,
                "icd9": icd9,
            }
            out_f.write(json.dumps(meta, ensure_ascii=True) + "\n")


if __name__ == "__main__":
    main()
