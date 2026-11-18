#!/usr/bin/env python3
"""Reproducible DART-IIoT experimental implementation.

Implements:
- ARFF/CSV ingestion with flexible UAH label detection
- leakage-resistant temporal splitting and windowing
- robust median/MAD normalization
- depthwise-separable Conv1D + GRU shared encoder
- classification and reconstruction heads
- trainable class prototypes
- energy/reconstruction/prototype evidence fusion
- class-conditional conformal rejection
- evidence-level ablations and deployment profiling

Results are generated
only from the supplied dataset at runtime.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.io import arff
from scipy.optimize import minimize, minimize_scalar
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import OneHotEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ----------------------------- reproducibility ----------------------------- #

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


def normalize_name(name: str) -> str:
    return "".join(ch.lower() for ch in str(name) if ch.isalnum())


def decode_value(v: Any) -> Any:
    if isinstance(v, (bytes, bytearray)):
        return v.decode("utf-8", errors="replace")
    return v


def load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".arff":
        data, _ = arff.loadarff(path)
        df = pd.DataFrame(data)
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].map(decode_value)
        return df
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path, low_memory=False)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported dataset format: {suffix}")


LABEL_ALIASES = {
    "binary": {
        "binaryresult", "binarylabel", "binaryclass", "attack", "isattack",
        "labelbinary", "resultbinary", "normalattack", "binary",
    },
    "category": {
        "categorizedresult", "categorisedresult", "categoryresult",
        "attackcategory", "category", "multiclass", "class", "label",
    },
    "specific": {
        "specificresult", "specificattack", "attacktype", "subclass",
        "specificclass", "detailedlabel", "attacklabel",
    },
}


def detect_label_columns(df: pd.DataFrame, explicit: Optional[str] = None) -> Dict[str, Optional[str]]:
    normalized = {normalize_name(c): c for c in df.columns}
    out: Dict[str, Optional[str]] = {"binary": None, "category": None, "specific": None}
    if explicit:
        if explicit not in df.columns:
            raise ValueError(f"Requested label column '{explicit}' is absent. Columns: {list(df.columns)}")
        out["category"] = explicit
    for role, aliases in LABEL_ALIASES.items():
        if out[role] is not None:
            continue
        for alias in aliases:
            if alias in normalized:
                out[role] = normalized[alias]
                break
    # Conservative fallback: low-cardinality nonnumeric column near the end.
    if out["category"] is None:
        candidates: List[Tuple[int, str]] = []
        for col in df.columns:
            nunique = int(df[col].nunique(dropna=True))
            if 2 <= nunique <= 100 and (not pd.api.types.is_numeric_dtype(df[col])):
                candidates.append((nunique, col))
        if candidates:
            candidates.sort(key=lambda x: (x[0], list(df.columns).index(x[1])))
            out["category"] = candidates[-1][1]
    return out


BENIGN_TOKENS = {
    "normal", "benign", "noevent", "noattack", "0", "false", "none", "background",
    "normaltraffic", "normaloperation", "legitimate",
}


def canonical_label(v: Any) -> str:
    if pd.isna(v):
        return "missing"
    s = str(decode_value(v)).strip()
    return s if s else "missing"


def is_benign_label(label: str) -> bool:
    n = normalize_name(label)
    return n in BENIGN_TOKENS or n.startswith("normal") or n.startswith("benign")


def choose_operational_label(
    df: pd.DataFrame,
    detected: Dict[str, Optional[str]],
    requested: Optional[str] = None,
) -> Tuple[str, pd.Series]:
    if requested:
        col = requested
    else:
        col = detected.get("category") or detected.get("specific") or detected.get("binary")
    if col is None:
        raise ValueError(
            "No label column could be identified. Pass --label-column with the exact column name."
        )
    labels = df[col].map(canonical_label)
    return col, labels


@dataclass
class SplitConfig:
    train_fraction: float = 0.60
    calibration_fraction: float = 0.20
    window: int = 16
    stride: int = 4
    max_rows: Optional[int] = None


@dataclass
class ModelConfig:
    conv_channels: int = 32
    gru_hidden: int = 48
    latent_dim: int = 32
    dropout: float = 0.10
    conv_blocks: int = 2
    kernel_size: int = 3


@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 4
    lambda_cls: float = 1.0
    lambda_rec: float = 0.5
    lambda_proto: float = 0.1
    lambda_sep: float = 0.05
    focal_gamma: float = 2.0


@dataclass
class CalibrationConfig:
    alpha: float = 0.025
    confidence_quantile: float = 0.01
    prototype_quantile: float = 0.99


class MedianMADTransformer:
    """Training-only median/MAD transform for mixed numerical/categorical data."""

    def __init__(self) -> None:
        self.numeric_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self.median: Optional[pd.Series] = None
        self.mad: Optional[pd.Series] = None
        self.encoder: Optional[OneHotEncoder] = None
        self.output_feature_names: List[str] = []

    def fit(self, df: pd.DataFrame) -> "MedianMADTransformer":
        self.numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        self.categorical_cols = [c for c in df.columns if c not in self.numeric_cols]
        if self.numeric_cols:
            numeric = df[self.numeric_cols].apply(pd.to_numeric, errors="coerce")
            self.median = numeric.median(axis=0)
            abs_dev = (numeric - self.median).abs()
            mad = abs_dev.median(axis=0)
            self.mad = mad.replace(0.0, 1.0)
        if self.categorical_cols:
            try:
                self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
            except TypeError:  # compatibility with older sklearn
                self.encoder = OneHotEncoder(handle_unknown="ignore", sparse=False, dtype=np.float32)
            cat = df[self.categorical_cols].fillna("missing").astype(str)
            self.encoder.fit(cat)
        self.output_feature_names = list(self.numeric_cols)
        if self.encoder is not None:
            self.output_feature_names.extend(self.encoder.get_feature_names_out(self.categorical_cols).tolist())
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        parts: List[np.ndarray] = []
        if self.numeric_cols:
            assert self.median is not None and self.mad is not None
            numeric = df[self.numeric_cols].apply(pd.to_numeric, errors="coerce")
            numeric = numeric.fillna(self.median)
            scaled = (numeric - self.median) / (1.4826 * self.mad + 1e-8)
            scaled = scaled.clip(-20.0, 20.0)
            parts.append(scaled.to_numpy(dtype=np.float32))
        if self.categorical_cols:
            assert self.encoder is not None
            cat = df[self.categorical_cols].fillna("missing").astype(str)
            parts.append(self.encoder.transform(cat).astype(np.float32, copy=False))
        if not parts:
            raise ValueError("No usable feature columns remain after excluding labels.")
        return np.concatenate(parts, axis=1)


def feature_columns(df: pd.DataFrame, labels: Dict[str, Optional[str]], requested: str) -> List[str]:
    excluded = {c for c in labels.values() if c is not None}
    excluded.add(requested)
    # Common row identifiers can leak acquisition order/scenario identity.
    for col in df.columns:
        n = normalize_name(col)
        if n in {"id", "index", "rowid", "unnamed0", "timestamp", "time", "datetime", "date"}:
            excluded.add(col)
    return [c for c in df.columns if c not in excluded]


def map_class_labels(labels: Sequence[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    unique = sorted(set(labels), key=lambda s: (not is_benign_label(s), normalize_name(s)))
    benign = [x for x in unique if is_benign_label(x)]
    malicious = [x for x in unique if x not in benign]
    ordered = benign[:1] + malicious
    if not benign:
        warnings.warn(
            "No benign label was recognized. The lexicographically first class is assigned index 0; "
            "use --benign-label to override in production runs."
        )
        ordered = unique
    to_idx = {label: i for i, label in enumerate(ordered)}
    to_name = {i: label for label, i in to_idx.items()}
    return to_idx, to_name


def make_windows(
    X: np.ndarray,
    y: np.ndarray,
    window: int,
    stride: int,
    forbidden_label: Optional[int] = None,
    reject_if_any_forbidden: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    xs: List[np.ndarray] = []
    ys: List[int] = []
    for end in range(window - 1, len(X), stride):
        start = end - window + 1
        yw = y[start : end + 1]
        if forbidden_label is not None and reject_if_any_forbidden and np.any(yw == forbidden_label):
            continue
        xs.append(X[start : end + 1])
        ys.append(int(y[end]))
    if not xs:
        return np.empty((0, window, X.shape[1]), dtype=np.float32), np.empty((0,), dtype=np.int64)
    return np.stack(xs).astype(np.float32), np.asarray(ys, dtype=np.int64)


@dataclass
class PreparedFold:
    X_train: np.ndarray
    y_train: np.ndarray
    X_cal: np.ndarray
    y_cal: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    unknown_test: np.ndarray
    class_to_idx: Dict[str, int]
    idx_to_class: Dict[int, str]
    held_out_attack: Optional[str]
    feature_names: List[str]
    metadata: Dict[str, Any]


def prepare_fold(
    df: pd.DataFrame,
    label_column: Optional[str],
    split_cfg: SplitConfig,
    held_out_attack: Optional[str] = None,
    benign_override: Optional[str] = None,
) -> PreparedFold:
    if split_cfg.max_rows is not None and len(df) > split_cfg.max_rows:
        # Uniformly sample the chronology instead of taking only the beginning.
        idx = np.linspace(0, len(df) - 1, split_cfg.max_rows, dtype=int)
        df = df.iloc[idx].reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    detected = detect_label_columns(df, explicit=label_column)
    actual_label_col, labels = choose_operational_label(df, detected, requested=label_column)
    if benign_override:
        labels = labels.map(lambda x: "Benign" if normalize_name(x) == normalize_name(benign_override) else x)

    classes = sorted(labels.unique().tolist())
    if held_out_attack is not None and held_out_attack not in classes:
        normalized_map = {normalize_name(c): c for c in classes}
        key = normalize_name(held_out_attack)
        if key not in normalized_map:
            raise ValueError(f"Held-out attack '{held_out_attack}' not found. Available classes: {classes}")
        held_out_attack = normalized_map[key]

    known_labels = [x for x in labels.tolist() if x != held_out_attack]
    class_to_idx, idx_to_class = map_class_labels(known_labels)
    unknown_idx = len(class_to_idx)
    y_all = np.asarray([class_to_idx.get(x, unknown_idx) for x in labels], dtype=np.int64)

    features = feature_columns(df, detected, actual_label_col)
    raw_X = df[features].copy()
    n = len(df)
    train_end = int(n * split_cfg.train_fraction)
    cal_end = int(n * (split_cfg.train_fraction + split_cfg.calibration_fraction))
    if not (split_cfg.window < train_end < cal_end < n):
        raise ValueError("Dataset is too small for requested temporal split and window length.")

    train_fit_mask = np.arange(n) < train_end
    if held_out_attack is not None:
        train_fit_mask &= labels.to_numpy() != held_out_attack
    transformer = MedianMADTransformer().fit(raw_X.loc[train_fit_mask])
    X_all = transformer.transform(raw_X)

    # Build windows separately so no window crosses a split boundary.
    Xtr, ytr = make_windows(
        X_all[:train_end], y_all[:train_end], split_cfg.window, split_cfg.stride,
        forbidden_label=unknown_idx if held_out_attack else None,
        reject_if_any_forbidden=True,
    )
    Xcal, ycal = make_windows(
        X_all[train_end:cal_end], y_all[train_end:cal_end], split_cfg.window,
        split_cfg.stride, forbidden_label=unknown_idx if held_out_attack else None,
        reject_if_any_forbidden=True,
    )
    Xte_late, yte_late = make_windows(
        X_all[cal_end:], y_all[cal_end:], split_cfg.window, split_cfg.stride
    )

    # For leave-one-attack-family-out, all held-out records are eligible as unknown
    # test evidence, while known test traffic remains strictly future traffic.
    if held_out_attack is not None:
        held_positions = np.flatnonzero(labels.to_numpy() == held_out_attack)
        unknown_windows: List[np.ndarray] = []
        for end in held_positions:
            start = end - split_cfg.window + 1
            if start >= 0:
                unknown_windows.append(X_all[start : end + 1])
        if unknown_windows:
            X_unknown = np.stack(unknown_windows).astype(np.float32)
            y_unknown = np.full(len(X_unknown), unknown_idx, dtype=np.int64)
            # Deduplication is intentionally not attempted: each attack endpoint is an event.
            Xte = np.concatenate([Xte_late[yte_late != unknown_idx], X_unknown], axis=0)
            yte = np.concatenate([yte_late[yte_late != unknown_idx], y_unknown], axis=0)
        else:
            Xte, yte = Xte_late, yte_late
    else:
        Xte, yte = Xte_late, yte_late

    # Deterministic test shuffle prevents metrics being order-sensitive while preserving
    # the temporal construction of each individual window.
    rng = np.random.default_rng(12345)
    perm = rng.permutation(len(Xte))
    Xte, yte = Xte[perm], yte[perm]
    unknown_test = yte == unknown_idx

    meta = {
        "rows": int(n),
        "label_column": actual_label_col,
        "detected_label_columns": detected,
        "raw_feature_columns": len(features),
        "transformed_features": int(X_all.shape[1]),
        "train_windows": int(len(Xtr)),
        "calibration_windows": int(len(Xcal)),
        "test_windows": int(len(Xte)),
        "unknown_test_windows": int(unknown_test.sum()),
        "classes": classes,
        "known_classes": list(class_to_idx.keys()),
        "held_out_attack": held_out_attack,
        "split": asdict(split_cfg),
    }
    if len(Xtr) == 0 or len(Xcal) == 0 or len(Xte) == 0:
        raise ValueError(f"Empty window split generated: {meta}")
    return PreparedFold(
        Xtr, ytr, Xcal, ycal, Xte, yte, unknown_test,
        class_to_idx, idx_to_class, held_out_attack,
        transformer.output_feature_names, meta,
    )


# --------------------------------- model ---------------------------------- #

class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        pad = kernel_size - 1
        self.pad = pad
        self.depthwise = nn.Conv1d(in_ch, in_ch, kernel_size, groups=in_ch, padding=0)
        self.pointwise = nn.Conv1d(in_ch, out_ch, 1)
        self.norm = nn.BatchNorm1d(out_ch)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.pad, 0))  # causal left padding
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        x = F.gelu(x)
        return self.dropout(x)


class DARTIIoT(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, cfg: ModelConfig) -> None:
        super().__init__()
        blocks: List[nn.Module] = []
        ch = input_dim
        for _ in range(cfg.conv_blocks):
            blocks.append(DepthwiseSeparableBlock(ch, cfg.conv_channels, cfg.kernel_size, cfg.dropout))
            ch = cfg.conv_channels
        self.conv = nn.Sequential(*blocks)
        self.encoder_gru = nn.GRU(cfg.conv_channels, cfg.gru_hidden, batch_first=True)
        self.latent = nn.Linear(cfg.gru_hidden, cfg.latent_dim)
        self.classifier = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.latent_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.latent_dim, num_classes),
        )
        self.decoder_gru = nn.GRU(cfg.latent_dim, cfg.gru_hidden, batch_first=True)
        self.decoder_out = nn.Linear(cfg.gru_hidden, input_dim)
        self.prototypes = nn.Parameter(torch.randn(num_classes, cfg.latent_dim) * 0.05)
        self.num_classes = num_classes
        self.cfg = cfg

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: [B, T, F]
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        _, h_last = self.encoder_gru(h)
        z = self.latent(h_last[-1])
        logits = self.classifier(z)
        repeated = z.unsqueeze(1).expand(-1, x.shape[1], -1)
        dec, _ = self.decoder_gru(repeated)
        recon = self.decoder_out(dec)
        return z, logits, recon


def effective_number_weights(y: np.ndarray, num_classes: int, beta: float = 0.999) -> torch.Tensor:
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = (1.0 - beta) / (1.0 - np.power(beta, counts))
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def focal_loss(logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor, gamma: float) -> torch.Tensor:
    logp = F.log_softmax(logits, dim=1)
    p = logp.exp()
    idx = torch.arange(len(targets), device=targets.device)
    pt = p[idx, targets]
    logpt = logp[idx, targets]
    wt = weights.to(targets.device)[targets]
    return (-wt * (1.0 - pt).pow(gamma) * logpt).mean()


def prototype_losses(z: torch.Tensor, y: torch.Tensor, prototypes: torch.Tensor, margin: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    own = prototypes[y]
    compact = ((z - own) ** 2).sum(dim=1).mean()
    dist = torch.cdist(z, prototypes, p=2).pow(2)
    mask = F.one_hot(y, num_classes=prototypes.shape[0]).bool()
    other = dist.masked_fill(mask, float("inf")).min(dim=1).values
    sep = F.relu(margin - other).pow(2).mean()
    return compact, sep


@dataclass
class TrainHistory:
    train_loss: List[float]
    val_loss: List[float]
    best_epoch: int
    seconds: float


def train_model(
    fold: PreparedFold,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    seed: int,
    device: torch.device,
) -> Tuple[DARTIIoT, TrainHistory]:
    set_seed(seed)
    model = DARTIIoT(fold.X_train.shape[2], len(fold.class_to_idx), model_cfg).to(device)
    weights = effective_number_weights(fold.y_train, len(fold.class_to_idx))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay
    )
    train_ds = TensorDataset(torch.from_numpy(fold.X_train), torch.from_numpy(fold.y_train))
    val_ds = TensorDataset(torch.from_numpy(fold.X_cal), torch.from_numpy(fold.y_cal))
    train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=train_cfg.batch_size, shuffle=False, num_workers=0)

    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_val = float("inf")
    best_epoch = -1
    bad = 0
    train_losses: List[float] = []
    val_losses: List[float] = []
    started = time.perf_counter()

    for epoch in range(train_cfg.epochs):
        model.train()
        total, count = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            z, logits, recon = model(xb)
            l_cls = focal_loss(logits, yb, weights, train_cfg.focal_gamma)
            l_rec = F.huber_loss(recon, xb, delta=1.0)
            l_proto, l_sep = prototype_losses(z, yb, model.prototypes)
            loss = (
                train_cfg.lambda_cls * l_cls
                + train_cfg.lambda_rec * l_rec
                + train_cfg.lambda_proto * l_proto
                + train_cfg.lambda_sep * l_sep
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach()) * len(xb)
            count += len(xb)
        train_losses.append(total / max(count, 1))

        model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                z, logits, recon = model(xb)
                l_cls = focal_loss(logits, yb, weights, train_cfg.focal_gamma)
                l_rec = F.huber_loss(recon, xb, delta=1.0)
                l_proto, l_sep = prototype_losses(z, yb, model.prototypes)
                loss = (
                    train_cfg.lambda_cls * l_cls
                    + train_cfg.lambda_rec * l_rec
                    + train_cfg.lambda_proto * l_proto
                    + train_cfg.lambda_sep * l_sep
                )
                total += float(loss) * len(xb)
                count += len(xb)
        val_loss = total / max(count, 1)
        val_losses.append(val_loss)
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= train_cfg.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    elapsed = time.perf_counter() - started
    return model, TrainHistory(train_losses, val_losses, best_epoch, elapsed)


@dataclass
class RawOutputs:
    z: np.ndarray
    logits: np.ndarray
    recon_score: np.ndarray
    proto_nearest: np.ndarray
    proto_predicted: np.ndarray
    prediction: np.ndarray


def infer_raw(model: DARTIIoT, X: np.ndarray, batch_size: int, device: torch.device) -> RawOutputs:
    ds = TensorDataset(torch.from_numpy(X))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    zs, logits_all, recs, nearest, predicted_dist, preds = [], [], [], [], [], []
    model.eval()
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            z, logits, recon = model(xb)
            pred = logits.argmax(dim=1)
            dist = torch.cdist(z, model.prototypes, p=2).pow(2)
            rec = F.huber_loss(recon, xb, delta=1.0, reduction="none").mean(dim=(1, 2))
            zs.append(z.cpu().numpy())
            logits_all.append(logits.cpu().numpy())
            recs.append(rec.cpu().numpy())
            nearest.append(dist.min(dim=1).values.cpu().numpy())
            predicted_dist.append(dist.gather(1, pred[:, None]).squeeze(1).cpu().numpy())
            preds.append(pred.cpu().numpy())
    return RawOutputs(
        np.concatenate(zs), np.concatenate(logits_all), np.concatenate(recs),
        np.concatenate(nearest), np.concatenate(predicted_dist), np.concatenate(preds)
    )


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    def nll(log_t: float) -> float:
        t = float(np.exp(log_t))
        x = torch.tensor(logits / t, dtype=torch.float64)
        yy = torch.tensor(y, dtype=torch.long)
        return float(F.cross_entropy(x, yy))
    result = minimize_scalar(nll, bounds=(math.log(0.05), math.log(10.0)), method="bounded")
    return float(np.exp(result.x)) if result.success else 1.0


def softmax_np(logits: np.ndarray, temperature: float) -> np.ndarray:
    x = logits / max(temperature, 1e-6)
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def energy_np(logits: np.ndarray, temperature: float) -> np.ndarray:
    x = logits / max(temperature, 1e-6)
    m = x.max(axis=1)
    return -temperature * (m + np.log(np.exp(x - m[:, None]).sum(axis=1)))


@dataclass
class RobustScoreScaler:
    median: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, scores: np.ndarray) -> "RobustScoreScaler":
        med = np.median(scores, axis=0)
        mad = np.median(np.abs(scores - med), axis=0)
        scale = 1.4826 * np.where(mad < 1e-8, 1.0, mad)
        return cls(med.astype(np.float64), scale.astype(np.float64))

    def transform(self, scores: np.ndarray) -> np.ndarray:
        return (scores - self.median) / (self.scale + 1e-8)


@dataclass
class FusionModel:
    intercept: float
    weights: np.ndarray
    feature_mask: Tuple[bool, bool, bool]

    def score(self, standardized: np.ndarray) -> np.ndarray:
        mask = np.asarray(self.feature_mask, dtype=bool)
        linear = self.intercept + standardized[:, mask] @ self.weights
        return 1.0 / (1.0 + np.exp(-np.clip(linear, -40, 40)))


def fit_nonnegative_fusion(
    standardized: np.ndarray,
    y_cal: np.ndarray,
    mask: Tuple[bool, bool, bool] = (True, True, True),
) -> FusionModel:
    """Fit fusion using episodic attack-class pseudo-unknown labels.

    Every non-benign known attack class is treated once as pseudo-unknown, while
    all remaining calibration examples are treated as known. This does not use
    the final held-out attack family.
    """
    classes = sorted(int(k) for k in np.unique(y_cal) if int(k) != 0)
    X_eps, y_eps = [], []
    if not classes:
        # Fallback: use robust upper-tail synthetic pseudo-unknowns.
        aggregate = standardized.mean(axis=1)
        cutoff = np.quantile(aggregate, 0.90)
        y = (aggregate >= cutoff).astype(np.float64)
        X_eps = [standardized]
        y_eps = [y]
    else:
        rng = np.random.default_rng(2026)
        for k in classes:
            pos = np.flatnonzero(y_cal == k)
            neg = np.flatnonzero(y_cal != k)
            if len(pos) == 0 or len(neg) == 0:
                continue
            take = min(len(neg), max(len(pos), 64))
            neg_sel = rng.choice(neg, size=take, replace=len(neg) < take)
            idx = np.concatenate([pos, neg_sel])
            X_eps.append(standardized[idx])
            y_eps.append(np.concatenate([np.ones(len(pos)), np.zeros(len(neg_sel))]))
    X = np.concatenate(X_eps, axis=0)
    y = np.concatenate(y_eps, axis=0)
    active = np.asarray(mask, dtype=bool)
    Xa = X[:, active]

    def objective(theta: np.ndarray) -> float:
        b = theta[0]
        w = theta[1:]
        p = 1.0 / (1.0 + np.exp(-np.clip(b + Xa @ w, -40, 40)))
        eps = 1e-9
        loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
        return float(loss + 1e-3 * np.dot(w, w))

    init = np.r_[0.0, np.ones(Xa.shape[1]) / max(Xa.shape[1], 1)]
    bounds = [(None, None)] + [(0.0, None)] * Xa.shape[1]
    result = minimize(objective, init, method="L-BFGS-B", bounds=bounds)
    theta = result.x if result.success else init
    return FusionModel(float(theta[0]), np.asarray(theta[1:], dtype=np.float64), mask)


@dataclass
class Calibrator:
    temperature: float
    score_scaler: RobustScoreScaler
    fusion: FusionModel
    class_scores: Dict[int, np.ndarray]
    confidence_threshold: float
    prototype_thresholds: Dict[int, float]
    alpha: float
    global_score_threshold: float

    def p_values(self, scores: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        q = np.empty(len(scores), dtype=np.float64)
        global_values = np.concatenate(list(self.class_scores.values()))
        for i, (s, k) in enumerate(zip(scores, candidates)):
            values = self.class_scores.get(int(k), global_values)
            q[i] = (1.0 + float(np.sum(values >= s))) / (len(values) + 1.0)
        return q


def make_score_matrix(raw: RawOutputs, temperature: float) -> np.ndarray:
    return np.column_stack([
        raw.recon_score,
        energy_np(raw.logits, temperature),
        raw.proto_nearest,
    ]).astype(np.float64)


def fit_calibrator(
    raw_cal: RawOutputs,
    y_cal: np.ndarray,
    cfg: CalibrationConfig,
    mask: Tuple[bool, bool, bool] = (True, True, True),
) -> Calibrator:
    temperature = fit_temperature(raw_cal.logits, y_cal)
    score_matrix = make_score_matrix(raw_cal, temperature)
    scaler = RobustScoreScaler.fit(score_matrix)
    standardized = scaler.transform(score_matrix)
    fusion = fit_nonnegative_fusion(standardized, y_cal, mask)
    fused = fusion.score(standardized)
    probs = softmax_np(raw_cal.logits, temperature)
    cand = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    class_scores: Dict[int, np.ndarray] = {}
    proto_thresholds: Dict[int, float] = {}
    for k in np.unique(y_cal):
        vals = fused[y_cal == k]
        if len(vals):
            class_scores[int(k)] = np.sort(vals)
        d = raw_cal.proto_predicted[(y_cal == k) & (cand == k)]
        if len(d) == 0:
            d = raw_cal.proto_predicted[y_cal == k]
        if len(d):
            proto_thresholds[int(k)] = float(np.quantile(d, cfg.prototype_quantile))
    correct_conf = conf[cand == y_cal]
    confidence_threshold = float(np.quantile(correct_conf, cfg.confidence_quantile)) if len(correct_conf) else 0.5
    global_threshold = float(np.quantile(fused, 1.0 - cfg.alpha))
    return Calibrator(
        temperature, scaler, fusion, class_scores, confidence_threshold,
        proto_thresholds, cfg.alpha, global_threshold
    )


def evaluate_variant(
    raw: RawOutputs,
    y_true: np.ndarray,
    unknown_mask: np.ndarray,
    calibrator: Calibrator,
    variant: str,
    raw_cal: RawOutputs,
    y_cal: np.ndarray,
) -> Dict[str, Any]:
    num_known = raw.logits.shape[1]
    unknown_label = num_known
    probs = softmax_np(raw.logits, calibrator.temperature)
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    cal_probs = softmax_np(raw_cal.logits, calibrator.temperature)

    feature_masks = {
        "complete": (True, True, True),
        "without_reconstruction": (False, True, True),
        "without_energy": (True, False, True),
        "without_prototype": (True, True, False),
    }
    score_matrix = make_score_matrix(raw, calibrator.temperature)
    cal_score_matrix = make_score_matrix(raw_cal, calibrator.temperature)

    if variant == "classification_only":
        unknown_score = 1.0 - conf
        cal_unknown_score = 1.0 - cal_probs.max(axis=1)
        threshold = float(np.quantile(cal_unknown_score, 1.0 - calibrator.alpha))
        reject = unknown_score > threshold
    elif variant in {"reconstruction_only", "energy_only", "prototype_only"}:
        index = {"reconstruction_only": 0, "energy_only": 1, "prototype_only": 2}[variant]
        one_scaler = RobustScoreScaler.fit(cal_score_matrix[:, [index]])
        unknown_score = one_scaler.transform(score_matrix[:, [index]])[:, 0]
        cal_unknown_score = one_scaler.transform(cal_score_matrix[:, [index]])[:, 0]
        threshold = float(np.quantile(cal_unknown_score, 1.0 - calibrator.alpha))
        reject = unknown_score > threshold
    else:
        mask = feature_masks.get(variant, (True, True, True))
        if mask == calibrator.fusion.feature_mask:
            local = calibrator
        else:
            local = fit_calibrator(raw_cal, y_cal, CalibrationConfig(
                alpha=calibrator.alpha,
                confidence_quantile=0.01,
                prototype_quantile=0.99,
            ), mask=mask)
        standardized = local.score_scaler.transform(score_matrix)
        unknown_score = local.fusion.score(standardized)
        if variant == "without_conformal":
            reject = unknown_score > local.global_score_threshold
        else:
            q = local.p_values(unknown_score, pred)
            proto_thr = np.asarray([
                local.prototype_thresholds.get(int(k), float("inf")) for k in pred
            ])
            accept = (
                (q > local.alpha)
                & (conf >= local.confidence_threshold)
                & (raw.proto_predicted <= proto_thr)
            )
            reject = ~accept

    final_pred = pred.copy()
    final_pred[reject] = unknown_label
    y_open = y_true.copy()
    y_open[unknown_mask] = unknown_label

    known_mask = ~unknown_mask
    known_macro_f1 = float(f1_score(y_true[known_mask], pred[known_mask], average="macro", zero_division=0)) if known_mask.any() else float("nan")
    known_mcc = float(matthews_corrcoef(y_true[known_mask], pred[known_mask])) if known_mask.sum() > 1 else float("nan")
    open_macro_f1 = float(f1_score(y_open, final_pred, average="macro", zero_division=0))
    unknown_binary = unknown_mask.astype(int)
    try:
        auroc = float(roc_auc_score(unknown_binary, unknown_score))
        auprc = float(average_precision_score(unknown_binary, unknown_score))
        fpr, tpr, _ = roc_curve(unknown_binary, unknown_score)
        idx = np.flatnonzero(tpr >= 0.95)
        fpr95 = float(fpr[idx[0]]) if len(idx) else 1.0
    except ValueError:
        auroc = auprc = fpr95 = float("nan")

    return {
        "variant": variant,
        "known_macro_f1": known_macro_f1,
        "known_mcc": known_mcc,
        "unknown_auroc": auroc,
        "unknown_auprc": auprc,
        "fpr_at_95_tpr": fpr95,
        "open_set_macro_f1": open_macro_f1,
        "known_acceptance_rate": float((~reject[known_mask]).mean()) if known_mask.any() else float("nan"),
        "unknown_rejection_rate": float(reject[unknown_mask].mean()) if unknown_mask.any() else float("nan"),
        "confusion_matrix": confusion_matrix(y_open, final_pred, labels=list(range(num_known + 1))).tolist(),
    }


def expected_calibration_error(probs: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            ece += mask.mean() * abs((pred[mask] == y[mask]).mean() - conf[mask].mean())
    return float(ece)


def profile_model(model: DARTIIoT, X: np.ndarray, device: torch.device, repeats: int = 100) -> Dict[str, Any]:
    params = sum(p.numel() for p in model.parameters())
    fp32_bytes = params * 4
    int8_bytes = params
    sample = torch.from_numpy(X[:1]).to(device)
    model.eval()
    with torch.no_grad():
        for _ in range(10):
            model(sample)
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            model(sample)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000.0)
    return {
        "parameters": int(params),
        "fp32_size_kib": fp32_bytes / 1024.0,
        "estimated_int8_size_kib": int8_bytes / 1024.0,
        "latency_median_ms_batch1": float(np.median(times)),
        "latency_p95_ms_batch1": float(np.quantile(times, 0.95)),
        "device": str(device),
        "note": "Latency includes PyTorch eager execution only; preprocessing and communication are excluded.",
    }


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, allow_nan=True)
    temp.replace(path)


def run_fold(
    df: pd.DataFrame,
    output_dir: Path,
    label_column: Optional[str],
    held_out_attack: Optional[str],
    split_cfg: SplitConfig,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    cal_cfg: CalibrationConfig,
    seed: int,
    benign_label: Optional[str],
    device: torch.device,
) -> Dict[str, Any]:
    fold = prepare_fold(df, label_column, split_cfg, held_out_attack, benign_label)
    model, history = train_model(fold, model_cfg, train_cfg, seed, device)
    raw_cal = infer_raw(model, fold.X_cal, train_cfg.batch_size, device)
    raw_test = infer_raw(model, fold.X_test, train_cfg.batch_size, device)
    calibrator = fit_calibrator(raw_cal, fold.y_cal, cal_cfg)

    variants = [
        "classification_only", "reconstruction_only", "energy_only", "prototype_only",
        "without_reconstruction", "without_energy", "without_prototype",
        "without_conformal", "complete",
    ]
    ablations = [
        evaluate_variant(raw_test, fold.y_test, fold.unknown_test, calibrator, v, raw_cal, fold.y_cal)
        for v in variants
    ]
    known_test = ~fold.unknown_test
    probs_test = softmax_np(raw_test.logits, calibrator.temperature)
    ece = expected_calibration_error(probs_test[known_test], fold.y_test[known_test]) if known_test.any() else float("nan")
    deployment = profile_model(model, fold.X_test, device)
    result = {
        "status": "completed",
        "seed": seed,
        "fold_metadata": fold.metadata,
        "model_config": asdict(model_cfg),
        "train_config": asdict(train_cfg),
        "calibration_config": asdict(cal_cfg),
        "training": asdict(history),
        "temperature": calibrator.temperature,
        "confidence_threshold": calibrator.confidence_threshold,
        "fusion": {
            "intercept": calibrator.fusion.intercept,
            "weights_active": calibrator.fusion.weights.tolist(),
            "feature_order": ["reconstruction", "energy", "prototype"],
        },
        "known_ece": ece,
        "deployment": deployment,
        "ablations": ablations,
    }
    fold_name = normalize_name(held_out_attack or "closed_set") or "fold"
    fold_dir = output_dir / fold_name / f"seed_{seed}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(fold_dir / "results.json", result)
    torch.save({
        "state_dict": model.state_dict(),
        "input_dim": fold.X_train.shape[2],
        "num_classes": len(fold.class_to_idx),
        "model_config": asdict(model_cfg),
        "class_to_idx": fold.class_to_idx,
    }, fold_dir / "model.pt")
    return result


def aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for result in results:
        held = result["fold_metadata"]["held_out_attack"] or "closed_set"
        seed = result["seed"]
        for ab in result["ablations"]:
            row = {"held_out_attack": held, "seed": seed, **{k: v for k, v in ab.items() if k != "confusion_matrix"}}
            rows.append(row)
    frame = pd.DataFrame(rows)
    numeric = [c for c in frame.columns if c not in {"held_out_attack", "seed", "variant"}]
    summary_rows = []
    for (held, variant), group in frame.groupby(["held_out_attack", "variant"], dropna=False):
        row: Dict[str, Any] = {"held_out_attack": held, "variant": variant, "n": len(group)}
        for col in numeric:
            vals = pd.to_numeric(group[col], errors="coerce")
            row[f"{col}_mean"] = float(vals.mean())
            row[f"{col}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        summary_rows.append(row)
    return {"per_run": rows, "summary": summary_rows}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run DART-IIoT experiments on UAH ICS ARFF/CSV data.")
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--output", type=Path, default=Path("results"))
    p.add_argument("--label-column", default=None)
    p.add_argument("--benign-label", default=None)
    p.add_argument("--held-out-attack", action="append", default=[])
    p.add_argument("--all-attack-folds", action="store_true")
    p.add_argument("--seeds", default="1,7,21,42,84")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--window", type=int, default=16)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--alpha", type=float, default=0.025)
    p.add_argument("--conv-channels", type=int, default=32)
    p.add_argument("--gru-hidden", type=int, default=48)
    p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--threads", type=int, default=1, help="CPU intra-op threads; 1 is reproducible and avoids oversubscription.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(max(1, args.threads))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    if not args.data.exists():
        raise FileNotFoundError(args.data)
    df = load_table(args.data)
    detected = detect_label_columns(df, explicit=args.label_column)
    label_col, labels = choose_operational_label(df, detected, args.label_column)
    classes = sorted(labels.unique().tolist())
    attacks = [c for c in classes if not is_benign_label(c)]
    requested_folds = list(args.held_out_attack)
    if args.all_attack_folds:
        requested_folds = attacks
    if not requested_folds:
        # A closed-set run is useful as a pipeline check but cannot measure unknown detection.
        requested_folds = [None]

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    split_cfg = SplitConfig(window=args.window, stride=args.stride, max_rows=args.max_rows)
    model_cfg = ModelConfig(args.conv_channels, args.gru_hidden, args.latent_dim)
    train_cfg = TrainConfig(
        epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.learning_rate, patience=args.patience,
    )
    cal_cfg = CalibrationConfig(alpha=args.alpha)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset": str(args.data.resolve()),
        "rows": len(df),
        "columns": list(df.columns),
        "detected_labels": detected,
        "operational_label": label_col,
        "classes": classes,
        "attack_folds": requested_folds,
        "seeds": seeds,
        "device": str(device),
    }
    atomic_json(args.output / "manifest.json", manifest)
    results: List[Dict[str, Any]] = []
    for held in requested_folds:
        for seed in seeds:
            print(f"[DART-IIoT] held_out={held!r}, seed={seed}", flush=True)
            results.append(run_fold(
                df, args.output, args.label_column, held, split_cfg, model_cfg,
                train_cfg, cal_cfg, seed, args.benign_label, device,
            ))
    aggregate = aggregate_results(results)
    atomic_json(args.output / "aggregate_results.json", aggregate)
    pd.DataFrame(aggregate["per_run"]).to_csv(args.output / "per_run_metrics.csv", index=False)
    pd.DataFrame(aggregate["summary"]).to_csv(args.output / "summary_metrics.csv", index=False)
    print(f"Completed {len(results)} run(s). Results: {args.output.resolve()}")


if __name__ == "__main__":
    main()
