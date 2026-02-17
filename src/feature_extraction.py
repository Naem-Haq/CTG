# src/feature_extraction.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import wfdb

from scipy.signal import welch, find_peaks
from sklearn.neighbors import KDTree

from .preprocessing import PreprocessConfig, preprocess_fhr, preprocess_uc
from .utils import (
    nan_robust,
    trimmed_mean,
    downsample_mean,
    runs_of_true,
    fill_nans_for_processing,
)


@dataclass(frozen=True)
class FeatureConfig:
    # Window duration (used for UC sentinel, and rates)
    window_min: int = 10

    # Event detection thresholds
    accel_thresh_bpm: float = 15.0
    accel_min_dur_sec: float = 15.0
    decel_thresh_bpm: float = 15.0
    decel_min_dur_sec: float = 15.0

    # PSD
    psd_nperseg_sec: int = 60

    # Entropy
    perm_order: int = 3
    perm_delay: int = 1
    sampen_m: int = 2
    sampen_r: float = 0.2
    sampen_downsample_to_hz: float = 1.0

    # UC peak detection
    uc_peak_distance_sec: float = 75.0  # <-- your chosen value
    uc_prom_scale: float = 0.3          # prominence = std * scale


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


def baseline_estimate(fhr: np.ndarray) -> float:
    x = nan_robust(fhr)
    if len(x) == 0:
        return np.nan
    return float(np.median(x))


def detect_events_threshold(
    fhr: np.ndarray,
    fs: float,
    baseline: float,
    thresh_bpm: float,
    min_dur_sec: float,
    mode: str = "accel"
) -> Dict[str, float]:
    """
    Threshold crossing events relative to baseline.
    mode:
      - "accel": fhr >= baseline + thresh
      - "decel": fhr <= baseline - thresh
    """
    x = np.asarray(fhr, dtype=float)
    valid = np.isfinite(x)
    if not np.isfinite(baseline) or valid.sum() < 2:
        return {"count": 0, "total_dur_sec": 0.0, "mean_dur_sec": 0.0, "max_dur_sec": 0.0,
                "mean_amp_bpm": 0.0, "max_amp_bpm": 0.0, "area_bpm_sec": 0.0}

    if mode == "accel":
        mask = valid & (x >= baseline + thresh_bpm)
        amp = np.maximum(0.0, x - (baseline + thresh_bpm))
    else:
        mask = valid & (x <= baseline - thresh_bpm)
        amp = np.maximum(0.0, (baseline - thresh_bpm) - x)

    starts, ends = runs_of_true(mask)
    min_len = int(round(min_dur_sec * fs))
    events = [(s, e) for s, e in zip(starts, ends) if (e - s) >= min_len]

    if len(events) == 0:
        return {"count": 0, "total_dur_sec": 0.0, "mean_dur_sec": 0.0, "max_dur_sec": 0.0,
                "mean_amp_bpm": 0.0, "max_amp_bpm": 0.0, "area_bpm_sec": 0.0}

    durations = np.array([(e - s) / fs for s, e in events], dtype=float)
    amps = np.array([np.nanmax(amp[s:e]) for s, e in events], dtype=float)
    area = np.array([np.nansum(amp[s:e]) / fs for s, e in events], dtype=float)

    return {
        "count": int(len(events)),
        "total_dur_sec": float(np.sum(durations)),
        "mean_dur_sec": float(np.mean(durations)),
        "max_dur_sec": float(np.max(durations)),
        "mean_amp_bpm": float(np.mean(amps)),
        "max_amp_bpm": float(np.max(amps)),
        "area_bpm_sec": float(np.sum(area)),
    }


def bandpower_welch(x: np.ndarray, fs: float, fmin: float, fmax: float, nperseg: int) -> float:
    x = nan_robust(x)
    if len(x) < 8:
        return np.nan
    freqs, psd = welch(x, fs=fs, nperseg=min(len(x), nperseg))
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not mask.any():
        return np.nan
    return float(np.trapz(psd[mask], freqs[mask]))


def spectral_entropy(x: np.ndarray, fs: float, nperseg: int) -> float:
    x = nan_robust(x)
    if len(x) < 8:
        return np.nan
    freqs, psd = welch(x, fs=fs, nperseg=min(len(x), nperseg))
    psd = np.maximum(psd, 1e-12)
    p = psd / np.sum(psd)
    h = -np.sum(p * np.log(p))
    return float(h / np.log(len(p)))


def permutation_entropy(x: np.ndarray, order: int = 3, delay: int = 1) -> float:
    x = nan_robust(x)
    n = len(x)
    if n < order * delay + 1:
        return np.nan

    patterns = []
    for i in range(n - delay * (order - 1)):
        w = x[i:(i + delay * order):delay]
        patterns.append(tuple(np.argsort(w)))

    from collections import Counter
    c = Counter(patterns)
    probs = np.array(list(c.values()), dtype=float)
    probs /= probs.sum()

    h = -np.sum(probs * np.log(probs))
    return float(h / np.log(len(probs))) if len(probs) > 1 else 0.0


def sample_entropy_kdtree(x: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    x = nan_robust(x)
    n = len(x)
    if n <= m + 2:
        return np.nan

    sd = np.std(x)
    if not np.isfinite(sd) or sd == 0:
        return 0.0
    tol = r * sd

    def embed(seq, dim):
        return np.column_stack([seq[i:n - dim + 1 + i] for i in range(dim)])

    Xm = embed(x, m)
    Xm1 = embed(x, m + 1)

    tree_m = KDTree(Xm, metric="chebyshev")
    tree_m1 = KDTree(Xm1, metric="chebyshev")

    cnt_m = tree_m.query_radius(Xm, r=tol, count_only=True) - 1
    cnt_m1 = tree_m1.query_radius(Xm1, r=tol, count_only=True) - 1

    Cm = np.sum(cnt_m)
    Cm1 = np.sum(cnt_m1)
    if Cm <= 0 or Cm1 <= 0:
        return np.nan

    return float(-np.log(Cm1 / Cm))


def uc_contraction_features(uc: np.ndarray, fs: float, cfg: FeatureConfig) -> Dict[str, float]:
    """
    UC features with:
      - peak distance = cfg.uc_peak_distance_sec
      - sentinel for mean interpeak when <2 peaks: window_sec
      - extras: peak_rate_per_min, prom_mean, auc
    """
    uc = np.asarray(uc, dtype=float)
    x = uc.copy()

    # fill NaNs for peak detection only
    if np.isnan(x).any():
        x = fill_nans_for_processing(x)

    distance = int(cfg.uc_peak_distance_sec * fs)
    prom = np.std(x) * cfg.uc_prom_scale if np.std(x) > 0 else 1.0
    peaks, props = find_peaks(x, distance=distance, prominence=prom)

    peak_count = int(len(peaks))
    peak_heights = x[peaks] if peak_count > 0 else np.array([], dtype=float)

    window_sec = float(len(uc) / fs) if fs else float(cfg.window_min * 60)

    # sentinel: if <2 peaks, set to window length (meaning: very infrequent / none)
    if peak_count >= 2:
        ipi = np.diff(peaks) / fs
        mean_ipi = float(np.mean(ipi))
    else:
        mean_ipi = float(cfg.window_min * 60)

    prom_arr = props.get("prominences", np.array([], dtype=float))
    prom_mean = float(np.mean(prom_arr)) if len(prom_arr) else 0.0

    peak_rate_per_min = peak_count / (window_sec / 60.0) if window_sec > 0 else np.nan
    uc_auc = float(np.nanmean(uc) * window_sec) if np.isfinite(window_sec) else np.nan

    return {
        "uc_mean": float(np.nanmean(uc)),
        "uc_std": float(np.nanstd(uc)),
        "uc_max": float(np.nanmax(uc)) if np.isfinite(uc).any() else np.nan,
        "uc_range": float(np.nanmax(uc) - np.nanmin(uc)) if np.isfinite(uc).any() else np.nan,

        "uc_peak_count": peak_count,
        "uc_peak_mean": float(np.mean(peak_heights)) if peak_count > 0 else 0.0,
        "uc_peak_max": float(np.max(peak_heights)) if peak_count > 0 else 0.0,
        "uc_mean_interpeak_sec": mean_ipi,

        "uc_peak_rate_per_min": float(peak_rate_per_min),
        "uc_peak_prom_mean": prom_mean,
        "uc_auc": uc_auc,
    }


def extract_window_features(fhr_w: np.ndarray, uc_w: np.ndarray, fs: float, fcfg: FeatureConfig) -> Dict[str, float]:
    feats: Dict[str, float] = {}

    # Quality
    nan_pct = float(100.0 * np.isnan(fhr_w).mean())
    feats["fhr_remaining_nan_pct"] = nan_pct
    feats["fhr_valid_pct_post"] = 100.0 - nan_pct

    fhr_valid = nan_robust(fhr_w)
    if len(fhr_valid) < 10:
        return feats

    # Central tendency
    base = baseline_estimate(fhr_w)
    feats["fhr_baseline_median"] = base
    feats["fhr_mean"] = float(np.mean(fhr_valid))
    feats["fhr_median"] = float(np.median(fhr_valid))
    feats["fhr_trimmed_mean_10"] = trimmed_mean(fhr_valid, trim=0.1)

    # Variability
    feats["fhr_std"] = float(np.std(fhr_valid))
    feats["fhr_mad"] = float(np.median(np.abs(fhr_valid - np.median(fhr_valid))))
    feats["fhr_iqr"] = float(np.percentile(fhr_valid, 75) - np.percentile(fhr_valid, 25))

    diffs = np.diff(fhr_valid)
    feats["fhr_mean_abs_diff"] = float(np.mean(np.abs(diffs))) if len(diffs) else np.nan

    # LTV proxy: std of 1-min means
    fhr_1min = downsample_mean(fhr_w, factor=int(fs * 60))
    fhr_1min = fhr_1min[np.isfinite(fhr_1min)]
    feats["fhr_std_1min_means"] = float(np.std(fhr_1min)) if len(fhr_1min) >= 2 else np.nan

    # Events
    accel = detect_events_threshold(
        fhr_w, fs, base, fcfg.accel_thresh_bpm, fcfg.accel_min_dur_sec, mode="accel"
    )
    decel = detect_events_threshold(
        fhr_w, fs, base, fcfg.decel_thresh_bpm, fcfg.decel_min_dur_sec, mode="decel"
    )

    for k, v in accel.items():
        feats[f"accel_{k}"] = v
    for k, v in decel.items():
        feats[f"decel_{k}"] = v

    # Frequency + entropy
    nperseg = int(fcfg.psd_nperseg_sec * fs)
    feats["fhr_bandpower_0p03_0p15"] = bandpower_welch(fhr_valid, fs, 0.03, 0.15, nperseg=nperseg)
    feats["fhr_bandpower_0p15_0p5"] = bandpower_welch(fhr_valid, fs, 0.15, 0.5, nperseg=nperseg)
    feats["fhr_spectral_entropy"] = spectral_entropy(fhr_valid, fs, nperseg=nperseg)
    feats["fhr_perm_entropy"] = permutation_entropy(fhr_valid, order=fcfg.perm_order, delay=fcfg.perm_delay)

    # Sample entropy (downsample to ~1 Hz for speed)
    ds_factor = int(round(fs / fcfg.sampen_downsample_to_hz))
    ds_factor = max(1, ds_factor)
    fhr_ds = downsample_mean(fhr_w, factor=ds_factor)
    fhr_ds = fhr_ds[np.isfinite(fhr_ds)]
    feats["fhr_sample_entropy"] = (
        sample_entropy_kdtree(fhr_ds, m=fcfg.sampen_m, r=fcfg.sampen_r) if len(fhr_ds) >= 30 else np.nan
    )

    # UC
    feats.update(uc_contraction_features(uc_w, fs, cfg=fcfg))
    return feats


def build_feature_matrix(
    manifest_csv: Path,
    dataset_summary_csv: Path,
    data_dir: Path,
    out_csv: Path,
    pre_cfg: PreprocessConfig = PreprocessConfig(),
    feat_cfg: FeatureConfig = FeatureConfig(),
) -> pd.DataFrame:
    """
    Build feature matrix from manifest (expects keep_window already filtered or will filter here).
    Loads each record once, preprocesses once, then extracts window features.
    """
    manifest = pd.read_csv(manifest_csv)
    manifest = manifest.loc[manifest["keep_window"] == True].copy()

    dfA = pd.read_csv(dataset_summary_csv)
    label_map = dict(zip(dfA["record_id"].astype(str), dfA["outcome_label"]))

    rows: List[Dict] = []
    failures: List[Dict] = []

    for rid, g in manifest.groupby("record_id", sort=False):
        rid = str(rid)
        try:
            sig, fields, fhr_idx, uc_idx, fs = load_ctg_record(rid, data_dir)
            if fhr_idx is None or uc_idx is None:
                failures.append({"record_id": rid, "error": "Missing FHR/UC channel"})
                continue

            fs = float(fs)
            fhr_raw = sig[:, fhr_idx]
            uc_raw = sig[:, uc_idx]

            fhr_clean, _ = preprocess_fhr(fhr_raw, fs=fs, cfg=pre_cfg)
            uc_clean, _ = preprocess_uc(uc_raw, fs=fs, cfg=pre_cfg)

            y_label = label_map.get(rid, g["outcome_label"].iloc[0])

            for _, row in g.iterrows():
                start = int(row["start_sample"])
                end = int(row["end_sample"])
                fhr_w = fhr_clean[start:end]
                uc_w = uc_clean[start:end]

                feats = extract_window_features(fhr_w, uc_w, fs, fcfg=feat_cfg)

                meta = {
                    "record_id": rid,
                    "window_idx": int(row["window_idx"]),
                    "start_sample": start,
                    "end_sample": end,
                    "start_min": float(row["start_min"]),
                    "end_min": float(row["end_min"]),
                    "fs": float(fs),
                    "outcome_label": y_label,
                }
                rows.append({**meta, **feats})

        except Exception as e:
            failures.append({"record_id": rid, "error": str(e)})

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    if failures:
        fail_path = out_csv.parent / "blockD_failures.csv"
        pd.DataFrame(failures).to_csv(fail_path, index=False)

    return df
