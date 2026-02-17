# src/segmentation.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import wfdb

from .preprocessing import PreprocessConfig, preprocess_fhr, preprocess_uc
from .utils import window_qc_post


@dataclass(frozen=True)
class SegmentationConfig:
    fs: float = 4.0
    window_min: int = 10
    step_min: int = 1

    # Window QC thresholds (post-preprocess)
    min_valid_pct_post: float = 70.0
    max_gap_sec_post: float = 120.0

    drop_unknown_labels: bool = True


def load_ctg_record(record_id: str, data_dir: Path):
    rec_path = str((data_dir / str(record_id)).resolve())
    p_signal, fields = wfdb.rdsamp(rec_path)

    sig_names = [s.upper() for s in fields.get("sig_name", [])]

    def find_idx(keys):
        for i, name in enumerate(sig_names):
            for k in keys:
                if k in name:
                    return i
        return None

    fhr_idx = find_idx(["FHR", "FETAL"])
    uc_idx = find_idx(["UC", "TOCO", "UA", "UTERINE"])
    fs = float(fields.get("fs", 4.0))
    return p_signal, fields, fhr_idx, uc_idx, fs


def build_segmentation_manifest(
    dataset_summary_csv: Path,
    data_dir: Path,
    out_path: Path,
    seg_cfg: SegmentationConfig = SegmentationConfig(),
    pre_cfg: PreprocessConfig = PreprocessConfig(),
) -> pd.DataFrame:
    """
    Create overlapping window manifest CSV.
    Preprocesses each record once, then windows post-preprocess signals.
    """
    dfA = pd.read_csv(dataset_summary_csv)
    label_map = dict(zip(dfA["record_id"].astype(str), dfA["outcome_label"]))

    w = int(seg_cfg.window_min * 60 * seg_cfg.fs)
    step = int(seg_cfg.step_min * 60 * seg_cfg.fs)

    rows: List[Dict] = []
    failures: List[Dict] = []

    for rid in dfA["record_id"].astype(str).tolist():
        outcome_label = label_map.get(rid, np.nan)
        if seg_cfg.drop_unknown_labels:
            if pd.isna(outcome_label) or str(outcome_label).strip() == "":
                continue

        try:
            sig, fields, fhr_idx, uc_idx, fs = load_ctg_record(rid, data_dir)
            if fhr_idx is None or uc_idx is None:
                failures.append({"record_id": rid, "error": "Missing FHR/UC channel"})
                continue

            fhr_raw = sig[:, fhr_idx]
            uc_raw = sig[:, uc_idx]

            # Use actual fs from file but keep your config windows in samples
            # If fs differs (unlikely), you can adapt window calc; CTU-CHB is 4 Hz.
            fs = float(fs)

            fhr_clean, _ = preprocess_fhr(fhr_raw, fs=fs, cfg=pre_cfg)
            uc_clean, _ = preprocess_uc(uc_raw, fs=fs, cfg=pre_cfg)

            n = len(fhr_clean)
            window_idx = 0

            for start in range(0, n - w + 1, step):
                end = start + w
                fhr_w = fhr_clean[start:end]

                qc = window_qc_post(fhr_w, fs=fs)
                keep = (qc["valid_pct_post"] >= seg_cfg.min_valid_pct_post) and (qc["max_gap_sec_post"] <= seg_cfg.max_gap_sec_post)

                rows.append({
                    "record_id": str(rid),
                    "window_idx": int(window_idx),
                    "start_sample": int(start),
                    "end_sample": int(end),
                    "start_min": float(start / fs / 60.0),
                    "end_min": float(end / fs / 60.0),
                    "fs": float(fs),
                    "window_minutes": float(seg_cfg.window_min),
                    "step_minutes": float(seg_cfg.step_min),
                    "outcome_label": outcome_label,
                    "fhr_remaining_nan_pct_window": float(qc["remaining_nan_pct"]),
                    "fhr_valid_pct_post_window": float(qc["valid_pct_post"]),
                    "fhr_max_missing_gap_sec_post_window": float(qc["max_gap_sec_post"]),
                    "keep_window": bool(keep),
                })
                window_idx += 1

        except Exception as e:
            failures.append({"record_id": rid, "error": str(e)})

    manifest = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out_path, index=False)

    if failures:
        fail_path = out_path.parent / "blockC_failures.csv"
        pd.DataFrame(failures).to_csv(fail_path, index=False)

    return manifest
