"""Common utilities and numerical helper functions."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd


def ensure_directory(path: os.PathLike | str) -> Path:
    """Create directory if it does not exist and return it."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_project_root() -> Path:
    """Resolve project root using env override or package location."""
    env_root = os.getenv("CTG_PROJECT_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


def get_data_dir() -> Path:
    """Return canonical CTU-CHB data directory."""
    env_dir = os.getenv("CTG_DATA_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    return get_project_root() / "data" / "ctu-chb-intrapartum-cardiotocography-database-1.0.0"


def get_output_dir() -> Path:
    """Return canonical outputs directory."""
    env_dir = os.getenv("CTG_OUTPUT_DIR")
    if env_dir:
        return ensure_directory(Path(env_dir).expanduser().resolve())
    return ensure_directory(get_project_root() / "outputs")


def get_models_dir() -> Path:
    """Return models subdirectory under outputs."""
    return ensure_directory(get_output_dir() / "models")


def list_hea_files(data_dir: os.PathLike | str | None = None) -> list[str]:
    """List WFDB .hea files from a directory."""
    directory = Path(data_dir) if data_dir is not None else get_data_dir()
    return sorted([f.name for f in directory.glob("*.hea")])


def save_csv(df: pd.DataFrame, filename: str, output_dir: os.PathLike | str | None = None) -> Path:
    """Save dataframe to CSV and return path."""
    out_dir = Path(output_dir) if output_dir is not None else get_output_dir()
    ensure_directory(out_dir)
    path = out_dir / filename
    df.to_csv(path, index=False)
    return path


def load_csv(filename: str, output_dir: os.PathLike | str | None = None) -> pd.DataFrame:
    """Load CSV by filename from output directory."""
    out_dir = Path(output_dir) if output_dir is not None else get_output_dir()
    return pd.read_csv(out_dir / filename)


def save_model(model, model_name: str, models_dir: os.PathLike | str | None = None) -> Path:
    """Save model as pickle and return file path."""
    import pickle

    mdir = Path(models_dir) if models_dir is not None else get_models_dir()
    ensure_directory(mdir)
    path = mdir / f"{model_name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    return path


def load_model(model_name: str, models_dir: os.PathLike | str | None = None):
    """Load model pickle by model name."""
    import pickle

    mdir = Path(models_dir) if models_dir is not None else get_models_dir()
    path = mdir / f"{model_name}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def runs_of_true(mask: Iterable[bool]) -> Tuple[np.ndarray, np.ndarray]:
    """Return start/end indices for contiguous True runs in a boolean mask."""
    b = np.asarray(mask, dtype=bool)
    if b.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    padded = np.r_[False, b, False]
    diff = np.diff(padded.astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return starts.astype(int), ends.astype(int)


def interp_nan_with_gap_limit(x: np.ndarray, max_gap_samples: int) -> np.ndarray:
    """Linearly interpolate NaN gaps up to max_gap_samples."""
    y = np.asarray(x, dtype=float).copy()
    n = y.size
    if n == 0:
        return y

    nan_mask = np.isnan(y)
    finite_idx = np.flatnonzero(~nan_mask)
    if finite_idx.size < 2:
        return y

    full_idx = np.arange(n)
    y_interp = y.copy()
    y_interp[nan_mask] = np.interp(full_idx[nan_mask], full_idx[~nan_mask], y[~nan_mask])

    starts, ends = runs_of_true(nan_mask)
    for s, e in zip(starts, ends):
        if (e - s) <= max_gap_samples:
            y[s:e] = y_interp[s:e]
    return y


def fill_nans_for_processing(x: np.ndarray) -> np.ndarray:
    """Fill NaNs for signal processing with nearest/edge interpolation."""
    y = np.asarray(x, dtype=float).copy()
    nan_mask = np.isnan(y)
    if not nan_mask.any():
        return y

    finite = np.flatnonzero(~nan_mask)
    if finite.size == 0:
        return np.zeros_like(y)
    if finite.size == 1:
        y[nan_mask] = y[finite[0]]
        return y

    idx = np.arange(y.size)
    y[nan_mask] = np.interp(idx[nan_mask], idx[~nan_mask], y[~nan_mask])
    return y


def window_qc_post(fhr_w: np.ndarray, fs: float) -> dict[str, float]:
    """Compute post-preprocess window QC metrics."""
    x = np.asarray(fhr_w, dtype=float)
    if x.size == 0:
        return {"remaining_nan_pct": np.nan, "valid_pct_post": np.nan, "max_gap_sec_post": np.nan}

    nan_mask = np.isnan(x)
    starts, ends = runs_of_true(nan_mask)
    max_gap_samples = int(np.max(ends - starts)) if starts.size else 0
    max_gap_sec = float(max_gap_samples / fs) if fs else np.nan
    rem_nan_pct = float(100.0 * nan_mask.mean())

    return {
        "remaining_nan_pct": rem_nan_pct,
        "valid_pct_post": float(100.0 - rem_nan_pct),
        "max_gap_sec_post": max_gap_sec,
    }


def nan_robust(x: np.ndarray) -> np.ndarray:
    """Return finite values only."""
    arr = np.asarray(x, dtype=float)
    return arr[np.isfinite(arr)]


def trimmed_mean(x: np.ndarray, trim: float = 0.1) -> float:
    """Compute simple symmetric trimmed mean on finite values."""
    arr = np.sort(nan_robust(x))
    if arr.size == 0:
        return np.nan
    k = int(np.floor(arr.size * trim))
    if 2 * k >= arr.size:
        return float(np.mean(arr))
    return float(np.mean(arr[k: arr.size - k]))


def downsample_mean(x: np.ndarray, factor: int) -> np.ndarray:
    """Downsample by block-mean, retaining NaN where block has no finite values."""
    arr = np.asarray(x, dtype=float)
    if factor <= 1:
        return arr.copy()

    n = arr.size
    m = n // factor
    if m == 0:
        return np.array([], dtype=float)

    reshaped = arr[: m * factor].reshape(m, factor)
    with np.errstate(invalid="ignore"):
        return np.nanmean(reshaped, axis=1)
