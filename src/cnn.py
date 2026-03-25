from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import json
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wfdb
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from .preprocessing import PreprocessConfig, preprocess_fhr, preprocess_uc
from .utils import fill_nans_for_processing

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:  # pragma: no cover
    raise ImportError("PyTorch is required for CNN training. Install with: pip install torch") from exc


CLASS_NAMES: List[str] = ["Normal", "Suspicious", "Pathological"]
LABEL_TO_IDX: Dict[str, int] = {name: i for i, name in enumerate(CLASS_NAMES)}


@dataclass(frozen=True)
class CNNConfig:
    fs_expected: float = 4.0
    window_min: int = 10
    batch_size: int = 64
    epochs: int = 40
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    dropout: float = 0.3
    patience: int = 8
    random_state: int = 42
    num_workers: int = 2
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    grad_clip_norm: float = 1.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _find_signal_indices(sig_names: Sequence[str]) -> Tuple[int | None, int | None]:
    upper = [name.upper() for name in sig_names]

    def find_idx(keys: Iterable[str]) -> int | None:
        for i, name in enumerate(upper):
            for key in keys:
                if key in name:
                    return i
        return None

    fhr_idx = find_idx(["FHR", "FETAL"])
    uc_idx = find_idx(["UC", "TOCO", "UA", "UTERINE"])
    return fhr_idx, uc_idx


def load_manifest_3class(manifest_csv: Path) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_csv)
    required = ["record_id", "window_idx", "start_sample", "end_sample", "outcome_label", "keep_window"]
    missing = [c for c in required if c not in manifest.columns]
    if missing:
        raise ValueError(f"Missing required columns in manifest: {missing}")

    manifest = manifest.loc[manifest["keep_window"] == True].copy()
    manifest["record_id"] = manifest["record_id"].astype(str)
    manifest["outcome_label"] = manifest["outcome_label"].astype(str).str.strip()
    manifest = manifest.loc[manifest["outcome_label"].isin(CLASS_NAMES)].copy()
    manifest.reset_index(drop=True, inplace=True)

    if manifest.empty:
        raise ValueError("No windows available after keep_window/class filtering.")
    return manifest


def build_grouped_split(manifest: pd.DataFrame, cfg: CNNConfig) -> Tuple[pd.DataFrame, Dict]:
    if not np.isclose(cfg.train_ratio + cfg.val_ratio + cfg.test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    records = (
        manifest[["record_id", "outcome_label"]]
        .drop_duplicates(subset=["record_id"])
        .reset_index(drop=True)
    )

    train_records, temp_records = train_test_split(
        records,
        test_size=(1.0 - cfg.train_ratio),
        random_state=cfg.random_state,
        stratify=records["outcome_label"],
    )

    temp_ratio = cfg.val_ratio + cfg.test_ratio
    val_in_temp = cfg.val_ratio / temp_ratio

    val_records, test_records = train_test_split(
        temp_records,
        test_size=(1.0 - val_in_temp),
        random_state=cfg.random_state,
        stratify=temp_records["outcome_label"],
    )

    split_map: Dict[str, str] = {}
    for rid in train_records["record_id"]:
        split_map[str(rid)] = "train"
    for rid in val_records["record_id"]:
        split_map[str(rid)] = "val"
    for rid in test_records["record_id"]:
        split_map[str(rid)] = "test"

    split_manifest = manifest.copy()
    split_manifest["split"] = split_manifest["record_id"].map(split_map)
    split_manifest = split_manifest.loc[split_manifest["split"].notna()].copy()

    rec_counts = {
        split: int((records["record_id"].isin(ids)).sum())
        for split, ids in {
            "train": train_records["record_id"].tolist(),
            "val": val_records["record_id"].tolist(),
            "test": test_records["record_id"].tolist(),
        }.items()
    }

    summary = {
        "random_state": cfg.random_state,
        "ratios": {"train": cfg.train_ratio, "val": cfg.val_ratio, "test": cfg.test_ratio},
        "records_per_split": rec_counts,
        "windows_per_split": split_manifest["split"].value_counts().to_dict(),
        "class_distribution_windows": (
            split_manifest.groupby(["split", "outcome_label"]).size().unstack(fill_value=0).to_dict()
        ),
        "record_ids": {
            "train": sorted(train_records["record_id"].astype(str).tolist()),
            "val": sorted(val_records["record_id"].astype(str).tolist()),
            "test": sorted(test_records["record_id"].astype(str).tolist()),
        },
    }
    return split_manifest, summary


class RecordCache:
    def __init__(self, data_dir: Path, pre_cfg: PreprocessConfig) -> None:
        self.data_dir = Path(data_dir)
        self.pre_cfg = pre_cfg
        self._cache: Dict[str, Tuple[np.ndarray, np.ndarray, float]] = {}

    def get(self, record_id: str) -> Tuple[np.ndarray, np.ndarray, float]:
        rid = str(record_id)
        if rid in self._cache:
            return self._cache[rid]

        rec_path = str((self.data_dir / rid).resolve())
        p_signal, fields = wfdb.rdsamp(rec_path)
        sig_names = fields.get("sig_name", [])
        fhr_idx, uc_idx = _find_signal_indices(sig_names)
        if fhr_idx is None or uc_idx is None:
            raise ValueError(f"Missing FHR/UC channels for record {rid}")

        fs = float(fields.get("fs", self.pre_cfg.fs_default))
        fhr_raw = p_signal[:, fhr_idx]
        uc_raw = p_signal[:, uc_idx]

        fhr_clean, _ = preprocess_fhr(fhr_raw, fs=fs, cfg=self.pre_cfg)
        uc_clean, _ = preprocess_uc(uc_raw, fs=fs, cfg=self.pre_cfg)

        self._cache[rid] = (fhr_clean.astype(np.float32), uc_clean.astype(np.float32), fs)
        return self._cache[rid]


def _zscore_finite(x: np.ndarray) -> np.ndarray:
    y = np.asarray(x, dtype=np.float32).copy()
    finite = np.isfinite(y)
    if not finite.any():
        return np.zeros_like(y, dtype=np.float32)
    mu = float(np.mean(y[finite]))
    sd = float(np.std(y[finite]))
    if sd > 0:
        y[finite] = (y[finite] - mu) / sd
    else:
        y[finite] = 0.0
    y = fill_nans_for_processing(y).astype(np.float32)
    return y


class CTGWindowDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, cache: RecordCache, cfg: CNNConfig) -> None:
        self.frame = frame.reset_index(drop=True).copy()
        self.cache = cache
        self.expected_len = int(round(cfg.window_min * 60 * cfg.fs_expected))

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        row = self.frame.iloc[idx]
        rid = str(row["record_id"])
        start = int(row["start_sample"])
        end = int(row["end_sample"])
        label = LABEL_TO_IDX[str(row["outcome_label"])]

        fhr, uc, _ = self.cache.get(rid)
        fhr_w = fhr[start:end]
        uc_w = uc[start:end]

        x = np.vstack([_zscore_finite(fhr_w), _zscore_finite(uc_w)])

        if x.shape[1] != self.expected_len:
            if x.shape[1] < self.expected_len:
                pad = self.expected_len - x.shape[1]
                x = np.pad(x, ((0, 0), (0, pad)), mode="edge")
            else:
                x = x[:, : self.expected_len]

        x_t = torch.from_numpy(x.astype(np.float32))
        y_t = torch.tensor(label, dtype=torch.long)
        return x_t, y_t, rid


class CTG1DCNN(nn.Module):
    def __init__(self, n_classes: int = 3, dropout: float = 0.3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.MaxPool1d(2),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.features(x)
        return self.head(z)


def _collect_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    labels = [0, 1, 2]
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(np.mean(y_true == y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "per_class": {
            name: {
                "precision": float(report[name]["precision"]),
                "recall": float(report[name]["recall"]),
                "f1": float(report[name]["f1-score"]),
                "support": int(report[name]["support"]),
            }
            for name in CLASS_NAMES
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
    }


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    grad_clip_norm: float = 1.0,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    is_train = optimizer is not None
    model.train(is_train)

    losses: List[float] = []
    y_true_batches: List[np.ndarray] = []
    y_pred_batches: List[np.ndarray] = []
    prob_batches: List[np.ndarray] = []
    record_ids: List[str] = []

    for xb, yb, rids in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        logits = model(xb)
        loss = criterion(logits, yb)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()

        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        losses.append(float(loss.detach().cpu().item()))
        y_true_batches.append(yb.detach().cpu().numpy())
        y_pred_batches.append(preds.detach().cpu().numpy())
        prob_batches.append(probs.detach().cpu().numpy())
        record_ids.extend([str(r) for r in rids])

    y_true = np.concatenate(y_true_batches) if y_true_batches else np.array([], dtype=int)
    y_pred = np.concatenate(y_pred_batches) if y_pred_batches else np.array([], dtype=int)
    probs = np.concatenate(prob_batches) if prob_batches else np.zeros((0, len(CLASS_NAMES)), dtype=float)
    mean_loss = float(np.mean(losses)) if losses else np.nan
    return mean_loss, y_true, y_pred, probs, record_ids


def _plot_training_curves(history_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history_df["epoch"], history_df["train_loss"], label="train")
    axes[0].plot(history_df["epoch"], history_df["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy")
    axes[0].legend()

    axes[1].plot(history_df["epoch"], history_df["val_macro_f1"], label="val_macro_f1")
    axes[1].set_title("Validation Macro-F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro-F1")
    axes[1].set_ylim(0.0, 1.0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_confusion_matrix(cm: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1, 2], CLASS_NAMES, rotation=20)
    ax.set_yticks([0, 1, 2], CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("CNN 3-class confusion matrix")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _record_level_metrics(probs: np.ndarray, y_true: np.ndarray, record_ids: List[str]) -> Dict:
    rows = pd.DataFrame(
        {
            "record_id": record_ids,
            "y_true": y_true,
            "p0": probs[:, 0],
            "p1": probs[:, 1],
            "p2": probs[:, 2],
        }
    )

    agg = rows.groupby("record_id", as_index=False).agg(
        y_true=("y_true", "first"),
        p0=("p0", "mean"),
        p1=("p1", "mean"),
        p2=("p2", "mean"),
    )
    pmat = agg[["p0", "p1", "p2"]].to_numpy(dtype=float)
    pred = np.argmax(pmat, axis=1)
    return _collect_metrics(agg["y_true"].to_numpy(dtype=int), pred)


def run_cnn_experiment(
    manifest_csv: Path,
    data_dir: Path,
    out_dir: Path,
    cfg: CNNConfig = CNNConfig(),
    pre_cfg: PreprocessConfig = PreprocessConfig(),
    rf_report_path: Path | None = None,
) -> Dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(cfg.random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    manifest = load_manifest_3class(Path(manifest_csv))
    split_manifest, split_summary = build_grouped_split(manifest, cfg)

    split_summary_path = out_dir / "cnn_3class_split_summary.json"
    split_summary_path.write_text(json.dumps(split_summary, indent=2))

    train_df = split_manifest.loc[split_manifest["split"] == "train"].copy()
    val_df = split_manifest.loc[split_manifest["split"] == "val"].copy()
    test_df = split_manifest.loc[split_manifest["split"] == "test"].copy()

    cache = RecordCache(data_dir=Path(data_dir), pre_cfg=pre_cfg)

    train_ds = CTGWindowDataset(train_df, cache=cache, cfg=cfg)
    val_ds = CTGWindowDataset(val_df, cache=cache, cfg=cfg)
    test_ds = CTGWindowDataset(test_df, cache=cache, cfg=cfg)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    class_counts = train_df["outcome_label"].value_counts()
    counts = np.array([float(class_counts.get(name, 0)) for name in CLASS_NAMES], dtype=float)
    weights = counts.sum() / (len(CLASS_NAMES) * np.maximum(counts, 1.0))

    class_weight_t = torch.tensor(weights, dtype=torch.float32, device=device)

    model = CTG1DCNN(n_classes=len(CLASS_NAMES), dropout=cfg.dropout).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weight_t)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    history: List[Dict] = []
    best_epoch = -1
    best_val_f1 = -np.inf
    best_state = None
    patience_counter = 0

    for epoch in range(1, cfg.epochs + 1):
        train_loss, y_train, p_train, _, _ = _run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            grad_clip_norm=cfg.grad_clip_norm,
        )
        val_loss, y_val, p_val, _, _ = _run_epoch(model, val_loader, criterion, device, optimizer=None)
        scheduler.step()

        labels = [0, 1, 2]
        train_macro_f1 = float(f1_score(y_train, p_train, labels=labels, average="macro", zero_division=0))
        val_macro_f1 = float(f1_score(y_val, p_val, labels=labels, average="macro", zero_division=0))

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_macro_f1": train_macro_f1,
                "val_macro_f1": val_macro_f1,
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )

        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= cfg.patience:
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")

    model.load_state_dict(best_state)

    model_path = out_dir / "cnn_3class_model.pt"
    torch.save({"model_state_dict": model.state_dict(), "config": asdict(cfg)}, model_path)

    history_df = pd.DataFrame(history)
    history_path = out_dir / "cnn_3class_history.csv"
    history_df.to_csv(history_path, index=False)

    curves_path = out_dir / "cnn_3class_training_curves.png"
    _plot_training_curves(history_df, curves_path)

    test_loss, y_test, p_test, probs_test, rids_test = _run_epoch(model, test_loader, criterion, device, optimizer=None)
    window_metrics = _collect_metrics(y_test, p_test)
    record_metrics = _record_level_metrics(probs_test, y_test, rids_test)

    cm = np.array(window_metrics["confusion_matrix"], dtype=int)
    cm_path = out_dir / "cnn_3class_confusion_matrix.png"
    _plot_confusion_matrix(cm, cm_path)

    report = {
        "metadata": {
            "notebook": "07-BlockG-CNN",
            "objective": "3-class CTG classification using 1D CNN on raw FHR+UC windows",
            "random_state": cfg.random_state,
            "device": str(device),
            "class_labels": CLASS_NAMES,
            "best_epoch": int(best_epoch),
        },
        "config": asdict(cfg),
        "dataset": {
            "total_windows": int(len(split_manifest)),
            "train_windows": int(len(train_df)),
            "val_windows": int(len(val_df)),
            "test_windows": int(len(test_df)),
            "train_records": int(split_manifest.loc[split_manifest["split"] == "train", "record_id"].nunique()),
            "val_records": int(split_manifest.loc[split_manifest["split"] == "val", "record_id"].nunique()),
            "test_records": int(split_manifest.loc[split_manifest["split"] == "test", "record_id"].nunique()),
        },
        "train_history": {
            "history_csv": str(history_path),
            "best_val_macro_f1": float(best_val_f1),
            "last_epoch": int(history_df["epoch"].max()) if not history_df.empty else 0,
        },
        "window_level": {
            "loss": float(test_loss),
            **window_metrics,
        },
        "record_level": record_metrics,
        "artifacts": {
            "model": str(model_path),
            "split_summary": str(split_summary_path),
            "training_curves": str(curves_path),
            "confusion_matrix": str(cm_path),
        },
    }

    report_path = out_dir / "cnn_3class_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    if rf_report_path is not None and Path(rf_report_path).exists():
        rf = json.loads(Path(rf_report_path).read_text())
        rf_perf = rf.get("performance", {})
        rf_acc = float(rf_perf.get("overall_accuracy", np.nan))
        rf_macro_f1 = float(
            np.nanmean([rf_perf.get("per_class_metrics", {}).get(name, {}).get("f1", np.nan) for name in CLASS_NAMES])
        )
        comparison = {
            "rf": {
                "accuracy": rf_acc,
                "macro_f1": rf_macro_f1,
            },
            "cnn": {
                "accuracy": float(window_metrics["accuracy"]),
                "macro_f1": float(window_metrics["macro_f1"]),
            },
            "delta_cnn_minus_rf": {
                "accuracy": float(window_metrics["accuracy"] - rf_acc) if np.isfinite(rf_acc) else np.nan,
                "macro_f1": float(window_metrics["macro_f1"] - rf_macro_f1) if np.isfinite(rf_macro_f1) else np.nan,
            },
        }
        (out_dir / "cnn_vs_rf_comparison.json").write_text(json.dumps(comparison, indent=2))

    return report
