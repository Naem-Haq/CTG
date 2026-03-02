from __future__ import annotations

import pandas as pd


def classify_contractions(uc_peak_rate_per_min: float) -> str:
    """
    NICE-inspired UC classification using contraction rate per 10 minutes.

    uc_peak_rate_per_min is expected to be contractions/min.
    """
    if pd.isna(uc_peak_rate_per_min):
        return "Unknown"

    rate_10min = float(uc_peak_rate_per_min) * 10.0
    if rate_10min <= 4.0:
        return "Normal"
    return "Suspicious"


def classify_baseline_fhr(fhr_baseline_median: float) -> str:
    """NICE-inspired baseline FHR classification."""
    if pd.isna(fhr_baseline_median):
        return "Unknown"

    v = float(fhr_baseline_median)
    if 110.0 <= v <= 160.0:
        return "Normal"
    if v < 100.0 or v > 160.0:
        return "Pathological"
    if 100.0 <= v < 110.0:
        return "Suspicious"
    return "Unknown"


def classify_variability(fhr_std: float) -> str:
    """NICE-inspired variability proxy classification."""
    if pd.isna(fhr_std):
        return "Unknown"

    v = float(fhr_std)
    if 5.0 <= v <= 25.0:
        return "Normal"
    if v < 5.0 or v > 25.0:
        return "Suspicious"
    return "Unknown"


def classify_decelerations(decel_count: float, decel_max_dur_sec: float, accel_count: float) -> str:
    """NICE-inspired deceleration pattern proxy classification."""
    if pd.isna(decel_count):
        decel_count = 0

    accel_present = pd.notna(accel_count) and float(accel_count) > 0
    n_decel = int(decel_count)

    if n_decel == 0:
        return "Normal"
    if n_decel == 1 and (pd.isna(decel_max_dur_sec) or float(decel_max_dur_sec) < 180.0):
        return "Suspicious" if not accel_present else "Normal"
    if n_decel >= 2 or (pd.notna(decel_max_dur_sec) and float(decel_max_dur_sec) >= 180.0):
        return "Pathological"
    return "Suspicious"


def combine_nice_classifications(cont_class: str, baseline_class: str, var_class: str, decel_class: str) -> str:
    """
    Combine per-component classes into 3-class NICE-style label.

    - Pathological if any pathological, or >=2 suspicious
    - Suspicious if exactly 1 suspicious
    - Normal otherwise
    """
    components = [cont_class, baseline_class, var_class, decel_class]

    if "Unknown" in components:
        if "Pathological" in components:
            return "Pathological"
        if components.count("Suspicious") >= 2:
            return "Suspicious"
        return "Normal"

    pathological_count = components.count("Pathological")
    suspicious_count = components.count("Suspicious")

    if pathological_count > 0:
        return "Pathological"
    if suspicious_count >= 2:
        return "Pathological"
    if suspicious_count == 1:
        return "Suspicious"
    return "Normal"


def assign_3class_label(row: pd.Series) -> str:
    """Apply NICE-inspired classification logic to a feature row."""
    cont = classify_contractions(row.get("uc_peak_rate_per_min"))
    base = classify_baseline_fhr(row.get("fhr_baseline_median"))
    var = classify_variability(row.get("fhr_std"))
    decel = classify_decelerations(
        row.get("decel_count"),
        row.get("decel_max_dur_sec"),
        row.get("accel_count"),
    )
    return combine_nice_classifications(cont, base, var, decel)
