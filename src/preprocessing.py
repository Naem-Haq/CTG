# src/preprocessing.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .utils import interp_nan_with_gap_limit, runs_of_true, fill_nans_for_processing


@dataclass(frozen=True)
class PreprocessConfig:
    fs_default: float = 4.0

    # FHR cleaning
    fhr_min: float = 80.0
    fhr_max: float = 240.0
    fhr_missing_sentinel: float = 0.0

    # UC cleaning
    uc_min: float = 0.0
    uc_max: float = 100.0

    # Interp + smoothing
    max_interp_gap_sec: float = 30.0
    gauss_sigma_sec: float = 1.5

    # Optional z-scoring
    zscore_per_record: bool = False


def preprocess_fhr(fhr: np.ndarray, fs: float, cfg: PreprocessConfig) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    FHR preprocessing:
      1) sentinel/NaN -> NaN
      2) outliers outside [fhr_min, fhr_max] -> NaN
      3) interpolate short gaps (<= max_interp_gap_sec)
      4) Gaussian smoothing
      5) optional z-score normalization
    Returns cleaned fhr and qc dict.
    """
    y = np.asarray(fhr, dtype=float).copy()
    n = len(y)

    # Track original missing sources
    sentinel_mask = (y == cfg.fhr_missing_sentinel)
    nan_input_mask = np.isnan(y)

    # Missing -> NaN
    y[sentinel_mask | nan_input_mask] = np.nan

    # Outliers -> NaN
    outlier_mask = np.isfinite(y) & ((y < cfg.fhr_min) | (y > cfg.fhr_max))
    y[outlier_mask] = np.nan

    # Pre-interp QC
    pre_nan = np.isnan(y)
    valid_pre = ~pre_nan
    starts, ends = runs_of_true(pre_nan)
    pre_max_gap_samples = int(np.max(ends - starts)) if len(starts) else 0
    pre_max_gap_sec = pre_max_gap_samples / fs if fs else np.nan

    # Interpolate short gaps only
    max_fill = max(1, int(round(cfg.max_interp_gap_sec * fs)))
    if np.isfinite(y).sum() >= 2:
        y = interp_nan_with_gap_limit(y, max_gap_samples=max_fill)

    # Gaussian smoothing (do not destroy NaN gaps)
    sigma_samp = max(0.0, cfg.gauss_sigma_sec * fs)
    if sigma_samp > 0:
        if np.isnan(y).any():
            y_fill = fill_nans_for_processing(y)
            y_smooth = gaussian_filter1d(y_fill, sigma=sigma_samp, mode="nearest")
            y_smooth[np.isnan(y)] = np.nan
            y = y_smooth
        else:
            y = gaussian_filter1d(y, sigma=sigma_samp, mode="nearest")

    # Optional z-score
    if cfg.zscore_per_record:
        mu = np.nanmean(y)
        sd = np.nanstd(y)
        if np.isfinite(sd) and sd > 0:
            y = (y - mu) / sd
        else:
            y = y * 0.0

    # Post QC
    post_nan = np.isnan(y)
    starts2, ends2 = runs_of_true(post_nan)
    post_max_gap_samples = int(np.max(ends2 - starts2)) if len(starts2) else 0
    post_max_gap_sec = post_max_gap_samples / fs if fs else np.nan

    qc = {
        "fs": float(fs),
        "n": int(n),

        "fhr_sentinel_zero_pct": float(100.0 * sentinel_mask.mean()) if n else np.nan,
        "fhr_nan_input_pct": float(100.0 * nan_input_mask.mean()) if n else np.nan,

        "fhr_outliers_pct": float(100.0 * outlier_mask.mean()) if n else np.nan,
        "fhr_missing_pct": float(100.0 * pre_nan.mean()) if n else np.nan,
        "fhr_valid_pct": float(100.0 * valid_pre.mean()) if n else np.nan,
        "fhr_max_missing_gap_sec": float(pre_max_gap_sec),

        "fhr_remaining_nan_pct": float(100.0 * post_nan.mean()) if n else np.nan,
        "fhr_valid_pct_post": float(100.0 * (~post_nan).mean()) if n else np.nan,
        "fhr_max_missing_gap_sec_post": float(post_max_gap_sec),

        "interp_max_gap_sec": float(cfg.max_interp_gap_sec),
        "gauss_sigma_sec": float(cfg.gauss_sigma_sec),
        "zscore_per_record": bool(cfg.zscore_per_record),
        "fhr_range_bpm": f"[{cfg.fhr_min}, {cfg.fhr_max}]",
        "interp_filled_pct": float(max(0.0, (pre_nan.mean() - post_nan.mean()) * 100.0)) if n else np.nan,
    }
    return y, qc


def preprocess_uc(uc: np.ndarray, fs: float, cfg: PreprocessConfig) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    UC preprocessing:
      - clamp outliers outside [uc_min, uc_max] -> NaN
      - Gaussian smoothing
      - optional z-score
    """
    y = np.asarray(uc, dtype=float).copy()
    n = len(y)

    outlier_mask = np.isfinite(y) & ((y < cfg.uc_min) | (y > cfg.uc_max))
    y[outlier_mask] = np.nan

    sigma_samp = max(0.0, cfg.gauss_sigma_sec * fs)
    if sigma_samp > 0:
        if np.isnan(y).any():
            y_fill = fill_nans_for_processing(y)
            y_smooth = gaussian_filter1d(y_fill, sigma=sigma_samp, mode="nearest")
            y_smooth[np.isnan(y)] = np.nan
            y = y_smooth
        else:
            y = gaussian_filter1d(y, sigma=sigma_samp, mode="nearest")

    if cfg.zscore_per_record:
        mu = np.nanmean(y)
        sd = np.nanstd(y)
        if np.isfinite(sd) and sd > 0:
            y = (y - mu) / sd
        else:
            y = y * 0.0

    qc = {
        "uc_missing_pct": float(100.0 * np.isnan(y).mean()) if n else np.nan,
        "uc_outliers_pct": float(100.0 * outlier_mask.mean()) if n else np.nan,
    }
    return y, qc
