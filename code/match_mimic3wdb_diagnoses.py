import os
import re
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import wfdb


LOCAL_WDB_ROOT = r"e:\mimicIII\mimic3wdb-matched"
CLINICAL_ROOT = r"e:\mimicIII\mimic-iii-clinical-database-1.4"
OUT_CSV = os.path.join(LOCAL_WDB_ROOT, "matched_diagnoses.csv")

MASTER_RECORDS_FILE = os.path.join(LOCAL_WDB_ROOT, "RECORDS")


def _parse_subject_id(record_path: str) -> int | None:
    # record_path like p00/p000020/p000020-2183-04-28-17-47
    parts = record_path.replace("\\", "/").split("/")
    if len(parts) < 2:
        return None
    pid = parts[1]
    if not pid.startswith("p"):
        return None
    try:
        return int(pid[1:])
    except ValueError:
        return None


def _parse_datetime_from_name(record_name: str) -> datetime | None:
    # record_name like p000020-2183-04-28-17-47 or p000020-2183-04-28-17-47n
    base = os.path.basename(record_name)
    base = base[:-1] if base.endswith("n") else base
    m = re.search(r"p\d{6}-(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})(?:-(\d{2}))?", base)
    if not m:
        return None
    yyyy, mm, dd, hh, mi, ss = m.group(1, 2, 3, 4, 5, 6)
    ss = ss or "00"
    return datetime(int(yyyy), int(mm), int(dd), int(hh), int(mi), int(ss))


def _infer_record_type(record_path: str) -> str:
    base = os.path.basename(record_path)
    if base.endswith("n"):
        return "numeric"
    if _parse_datetime_from_name(base):
        return "waveform"
    return "segment"


def _build_segment_map(folder: str, records: list[str]) -> dict[str, list[dict[str, object]]]:
    seg_map: dict[str, list[dict[str, object]]] = {}
    for rec in records:
        local_record = os.path.join(LOCAL_WDB_ROOT, folder, rec)
        try:
            hdr = wfdb.rdheader(local_record)
        except Exception:
            continue

        if not hdr or getattr(hdr, "n_seg", 1) <= 1:
            continue

        if hdr.base_date and hdr.base_time:
            base_dt = datetime.combine(hdr.base_date, hdr.base_time)
        else:
            base_dt = _parse_datetime_from_name(rec)

        if base_dt is None or not hdr.fs:
            continue

        offset = 0
        for seg_name, seg_len in zip(hdr.seg_name, hdr.seg_len):
            seg_len = int(seg_len)
            if seg_name == "~":
                offset += seg_len
                continue
            seg_start = base_dt + timedelta(seconds=offset / hdr.fs)
            if seg_len > 0:
                seg_end = seg_start + timedelta(seconds=(seg_len - 1) / hdr.fs)
            else:
                seg_end = seg_start
            seg_map.setdefault(seg_name, []).append(
                {
                    "start_time": seg_start,
                    "end_time": seg_end,
                    "parent_record": rec,
                }
            )
            offset += seg_len
    return seg_map


def _get_record_times(
    record_path: str, segment_map: dict[str, list[dict[str, object]]] | None
) -> tuple[datetime | None, datetime | None]:
    # Prefer header timestamps; fall back to record name or segment mapping.
    try:
        local_record = os.path.join(LOCAL_WDB_ROOT, record_path)
        hdr = wfdb.rdheader(local_record)
    except Exception:
        hdr = None

    start_dt = None
    if hdr and hdr.base_date and hdr.base_time:
        start_dt = datetime.combine(hdr.base_date, hdr.base_time)
    else:
        start_dt = _parse_datetime_from_name(record_path)

    if start_dt is None and segment_map:
        seg_key = os.path.basename(record_path)
        if seg_key in segment_map:
            seg_info = segment_map[seg_key][0]
            return seg_info["start_time"], seg_info["end_time"]

    if start_dt is None or hdr is None or hdr.sig_len is None or hdr.fs is None:
        return start_dt, None

    duration_s = (float(hdr.sig_len) - 1.0) / float(hdr.fs)
    end_dt = start_dt + timedelta(seconds=duration_s)
    return start_dt, end_dt


def _load_records() -> pd.DataFrame:
    rows = []
    if not os.path.exists(MASTER_RECORDS_FILE):
        print(f"Missing master RECORDS at {MASTER_RECORDS_FILE}")
        return pd.DataFrame()

    with open(MASTER_RECORDS_FILE, "r") as master_f:
        for line in master_f:
            folder = line.strip().rstrip("/")
            if not folder:
                continue
            folder_records_path = os.path.join(LOCAL_WDB_ROOT, folder, "RECORDS")
            if not os.path.exists(folder_records_path):
                continue

            with open(folder_records_path, "r") as rec_f:
                records = [r.strip() for r in rec_f if r.strip()]

            segment_map = _build_segment_map(folder, records)

            for rec in records:
                record_path = f"{folder}/{rec}".replace("\\", "/")
                hea_path = os.path.join(LOCAL_WDB_ROOT, record_path + ".hea")
                if not os.path.exists(hea_path):
                    continue
                subject_id = _parse_subject_id(record_path)
                start_dt, end_dt = _get_record_times(record_path, segment_map)
                rows.append(
                    {
                        "record_path": record_path,
                        "record_type": _infer_record_type(record_path),
                        "subject_id": subject_id,
                        "start_time": start_dt,
                        "end_time": end_dt,
                    }
                )
    df = pd.DataFrame(rows)
    df = df.dropna(subset=["record_path", "subject_id"])
    return df


def _load_icustays(subject_ids: set[int]) -> pd.DataFrame:
    icu_path = os.path.join(CLINICAL_ROOT, "ICUSTAYS.csv.gz")
    usecols = ["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "INTIME", "OUTTIME"]
    chunks = []
    for chunk in pd.read_csv(
        icu_path,
        usecols=usecols,
        parse_dates=["INTIME", "OUTTIME"],
        chunksize=200000,
    ):
        if subject_ids:
            chunk = chunk[chunk["SUBJECT_ID"].isin(subject_ids)]
        if not chunk.empty:
            chunks.append(chunk)
    if chunks:
        icu = pd.concat(chunks, ignore_index=True)
    else:
        icu = pd.DataFrame(columns=usecols)
    return icu


def _match_record_to_stay(rec_row: pd.Series, stays: pd.DataFrame) -> pd.Series | None:
    if stays.empty:
        return None

    if pd.isna(rec_row["start_time"]):
        return None

    start = rec_row["start_time"]
    end = rec_row["end_time"]
    if pd.isna(end):
        end = start

    latest_start = stays["INTIME"].where(stays["INTIME"] > start, start)
    earliest_end = stays["OUTTIME"].where(stays["OUTTIME"] < end, end)
    overlap = (earliest_end - latest_start).dt.total_seconds()
    overlap = overlap.clip(lower=0)

    if overlap.max() > 0:
        idx = overlap.idxmax()
        return stays.loc[idx]

    # No overlap: choose the closest stay by time gap.
    gap_before = (stays["INTIME"] - end).dt.total_seconds()
    gap_after = (start - stays["OUTTIME"]).dt.total_seconds()
    gap = np.where(end < stays["INTIME"], gap_before, np.where(start > stays["OUTTIME"], gap_after, 0))
    idx = int(np.argmin(gap))
    return stays.iloc[idx]


def _build_matches(records_df: pd.DataFrame, icu_df: pd.DataFrame) -> pd.DataFrame:
    matches = []
    icu_by_subject = {sid: df for sid, df in icu_df.groupby("SUBJECT_ID")}
    for _, row in records_df.iterrows():
        sid = int(row["subject_id"])
        stays = icu_by_subject.get(sid, pd.DataFrame())
        stay = _match_record_to_stay(row, stays)
        if stay is None:
            matches.append({**row, "hadm_id": None, "icustay_id": None})
        else:
            matches.append(
                {
                    **row,
                    "hadm_id": int(stay["HADM_ID"]),
                    "icustay_id": int(stay["ICUSTAY_ID"]),
                }
            )
    return pd.DataFrame(matches)


def _load_diag_descriptions(hadm_ids: set[int]) -> pd.DataFrame:
    diag_path = os.path.join(CLINICAL_ROOT, "DIAGNOSES_ICD.csv.gz")
    d_icd_path = os.path.join(CLINICAL_ROOT, "D_ICD_DIAGNOSES.csv.gz")

    d_icd = pd.read_csv(d_icd_path, usecols=["ICD9_CODE", "LONG_TITLE"])
    d_icd["ICD9_CODE"] = d_icd["ICD9_CODE"].astype(str)

    chunks = []
    for chunk in pd.read_csv(diag_path, usecols=["SUBJECT_ID", "HADM_ID", "ICD9_CODE"], chunksize=200000):
        if hadm_ids:
            chunk = chunk[chunk["HADM_ID"].isin(hadm_ids)]
        if not chunk.empty:
            chunks.append(chunk)

    if chunks:
        diag = pd.concat(chunks, ignore_index=True)
    else:
        diag = pd.DataFrame(columns=["SUBJECT_ID", "HADM_ID", "ICD9_CODE"])

    diag["ICD9_CODE"] = diag["ICD9_CODE"].astype(str)
    diag = diag.merge(d_icd, on="ICD9_CODE", how="left")
    return diag


def main() -> None:
    records = _load_records()
    if records.empty:
        print("No records found locally. Check the master RECORDS file and folder RECORDS files.")
        return

    subject_ids = set(records["subject_id"].astype(int).tolist())
    icu = _load_icustays(subject_ids)
    matches = _build_matches(records, icu)

    hadm_ids = set(matches["hadm_id"].dropna().astype(int).tolist())
    diag = _load_diag_descriptions(hadm_ids)

    # Aggregate diagnoses per HADM_ID for a compact view.
    diag_grouped = (
        diag.groupby("HADM_ID")
        .agg(
            icd9_codes=("ICD9_CODE", lambda x: ";".join(sorted(set(x)))),
            diag_titles=("LONG_TITLE", lambda x: "; ".join(sorted(set(x.dropna())))),
        )
        .reset_index()
    )

    out = matches.merge(diag_grouped, left_on="hadm_id", right_on="HADM_ID", how="left")
    out = out.drop(columns=["HADM_ID"], errors="ignore")
    out = out.sort_values(["subject_id", "record_path"]).reset_index(drop=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"Matched records: {len(out)}")
    print(f"Output: {OUT_CSV}")


if __name__ == "__main__":
    main()
