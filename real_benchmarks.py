#!/usr/bin/env python3
"""Run DART-IIoT and matched baselines on UNSW-NB15 and N-BaIoT.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, TensorDataset

try:
    import rarfile
except ImportError:  # pragma: no cover
    rarfile = None

from dart_iiot import (
    CalibrationConfig,
    MedianMADTransformer,
    ModelConfig,
    PreparedFold,
    SplitConfig,
    TrainConfig,
    aggregate_results,
    atomic_json,
    canonical_label,
    evaluate_variant,
    expected_calibration_error,
    fit_calibrator,
    fit_temperature,
    infer_raw,
    map_class_labels,
    normalize_name,
    profile_model,
    softmax_np,
    train_model,
)

SEEDS_DEFAULT = (1, 7, 21, 42, 84)
NBAIOT_FAMILY_LEVEL = False
PARTITIONS = ("train", "cal", "test")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def uniform_cap(df: pd.DataFrame, n: Optional[int]) -> pd.DataFrame:
    if n is None or len(df) <= n:
        return df.copy()
    idx = np.linspace(0, len(df) - 1, n, dtype=int)
    return df.iloc[idx].copy()


def normalize_attack_label(v: object) -> str:
    s = canonical_label(v).strip().replace("_", " ")
    if normalize_name(s) in {"normal", "benign", "benigntraffic", "0", "null", "missing"}:
        return "Benign"
    return " ".join(part.capitalize() for part in s.split())


def load_unsw(
    train_csv: Path,
    test_csv: Path,
    max_train_per_class: Optional[int],
    max_test_per_class: Optional[int],
) -> pd.DataFrame:
    """Load the standard UNSW-NB15 training/testing CSV files.

    The official training file is divided into an 80/20 train/calibration split
    within each class while preserving source-file order. The official testing
    file remains untouched as the test partition except for optional uniform
    per-class caps used to make CPU experiments tractable.
    """
    if not train_csv.exists() or not test_csv.exists():
        raise FileNotFoundError(f"Missing UNSW files: {train_csv}, {test_csv}")
    tr = pd.read_csv(train_csv, low_memory=False)
    te = pd.read_csv(test_csv, low_memory=False)
    label_col = "attack_cat" if "attack_cat" in tr.columns else "label"
    if label_col not in te.columns:
        raise ValueError(f"UNSW test file lacks label column {label_col!r}")
    tr["attack_cat"] = tr[label_col].map(normalize_attack_label)
    te["attack_cat"] = te[label_col].map(normalize_attack_label)

    parts: List[pd.DataFrame] = []
    for label, g in tr.groupby("attack_cat", sort=False):
        g = uniform_cap(g, max_train_per_class).reset_index(drop=True)
        cut = max(1, min(len(g) - 1, int(0.8 * len(g)))) if len(g) > 1 else len(g)
        a = g.iloc[:cut].copy(); a["_partition"] = "train"
        b = g.iloc[cut:].copy(); b["_partition"] = "cal"
        a["_sequence_id"] = f"unsw-train-{normalize_name(label)}"
        b["_sequence_id"] = f"unsw-cal-{normalize_name(label)}"
        parts.extend([a, b])
    for label, g in te.groupby("attack_cat", sort=False):
        g = uniform_cap(g, max_test_per_class).reset_index(drop=True)
        g["_partition"] = "test"
        g["_sequence_id"] = f"unsw-test-{normalize_name(label)}"
        parts.append(g)
    out = pd.concat(parts, ignore_index=True)
    out["device"] = "UNSW-Cyber-Range"
    return out


def _read_csv_stream(fileobj, cap: Optional[int]) -> pd.DataFrame:
    df = pd.read_csv(fileobj, low_memory=False)
    return uniform_cap(df, cap).reset_index(drop=True)


def _split_sequence(df: pd.DataFrame, sequence_id: str, device: str, label: str) -> List[pd.DataFrame]:
    n = len(df)
    if n < 3:
        return []
    tr_end = max(1, int(0.60 * n))
    cal_end = max(tr_end + 1, int(0.80 * n))
    cal_end = min(cal_end, n - 1)
    cuts = ((0, tr_end, "train"), (tr_end, cal_end, "cal"), (cal_end, n, "test"))
    out = []
    for start, end, part in cuts:
        piece = df.iloc[start:end].copy()
        if piece.empty:
            continue
        piece["attack_cat"] = label
        piece["device"] = device
        piece["_partition"] = part
        piece["_sequence_id"] = f"{sequence_id}-{part}"
        out.append(piece)
    return out


def _rar_members(path: Path) -> Iterable[Tuple[str, object]]:
    if rarfile is None:
        raise RuntimeError("rarfile is required for N-BaIoT archives: pip install rarfile")
    with rarfile.RarFile(path) as rf:
        for info in rf.infolist():
            if info.filename.lower().endswith(".csv"):
                with rf.open(info) as f:
                    yield info.filename, f


def load_nbaiot(root: Path, max_rows_per_file: Optional[int]) -> pd.DataFrame:
    """Load the extracted UCI N-BaIoT directory.

    Supports the original layout (benign_traffic.csv plus Gafgyt/Mirai RAR
    archives) and already-extracted attack CSVs. Rows are uniformly capped per
    source file and split chronologically 60/20/20 within that file.
    """
    if not root.exists():
        raise FileNotFoundError(root)
    all_parts: List[pd.DataFrame] = []
    device_dirs = sorted([p for p in root.rglob("*") if p.is_dir() and (p / "benign_traffic.csv").exists()])
    if not device_dirs:
        raise ValueError(f"No N-BaIoT device folders with benign_traffic.csv found below {root}")
    for d in device_dirs:
        device = d.name
        benign = _read_csv_stream(d / "benign_traffic.csv", max_rows_per_file)
        all_parts.extend(_split_sequence(benign, f"{device}-benign", device, "Benign"))

        archives = sorted(d.glob("*attacks.rar"))
        for archive in archives:
            family = "Mirai" if "mirai" in archive.name.lower() else "Gafgyt"
            for member, fileobj in _rar_members(archive):
                subtype = Path(member).stem.replace("_", " ").replace(".", " ")
                label = f"{family} {subtype.title()}"
                frame = _read_csv_stream(fileobj, max_rows_per_file)
                all_parts.extend(_split_sequence(frame, f"{device}-{normalize_name(label)}", device, label))

        # Also support pre-extracted attack CSVs.
        for csv_path in sorted(d.glob("*.csv")):
            if csv_path.name == "benign_traffic.csv":
                continue
            lower = csv_path.name.lower()
            if not ("mirai" in lower or "gafgyt" in lower):
                continue
            family = "Mirai" if "mirai" in lower else "Gafgyt"
            subtype = csv_path.stem.replace("_", " ").replace(".", " ")
            label = family if NBAIOT_FAMILY_LEVEL else f"{family} {subtype.title()}"
            frame = _read_csv_stream(csv_path, max_rows_per_file)
            all_parts.extend(_split_sequence(frame, f"{device}-{normalize_name(label)}", device, label))
    if not all_parts:
        raise ValueError("N-BaIoT loader produced no records")
    return pd.concat(all_parts, ignore_index=True)


def feature_columns_explicit(df: pd.DataFrame, dataset: str, include_device: bool) -> List[str]:
    excluded = {"attack_cat", "label", "_partition", "_sequence_id", "id"}
    # Endpoint and absolute-time identifiers can memorize scenarios rather than behaviour.
    excluded.update({"srcip", "dstip", "sport", "dsport", "Stime", "Ltime", "stime", "ltime"})
    if not include_device:
        excluded.add("device")
    return [c for c in df.columns if c not in excluded and not str(c).lower().startswith("unnamed")]


def make_group_windows(
    X: np.ndarray,
    y: np.ndarray,
    sequence_ids: np.ndarray,
    window: int,
    stride: int,
    forbidden: Optional[int],
    reject_forbidden: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    xs: List[np.ndarray] = []
    ys: List[int] = []
    # sort=False equivalent: retain first-seen order
    _, first = np.unique(sequence_ids, return_index=True)
    ordered_ids = sequence_ids[np.sort(first)]
    for sid in ordered_ids:
        idx = np.flatnonzero(sequence_ids == sid)
        xg, yg = X[idx], y[idx]
        for end in range(window - 1, len(xg), stride):
            start = end - window + 1
            yw = yg[start:end + 1]
            if forbidden is not None and reject_forbidden and np.any(yw == forbidden):
                continue
            xs.append(xg[start:end + 1])
            ys.append(int(yg[end]))
    if not xs:
        return np.empty((0, window, X.shape[1]), dtype=np.float32), np.empty(0, dtype=np.int64)
    return np.stack(xs).astype(np.float32), np.asarray(ys, dtype=np.int64)


def prepare_explicit_fold(
    df: pd.DataFrame,
    dataset: str,
    held_out_attack: str,
    window: int,
    stride: int,
    include_device: bool,
    feature_cap: Optional[int] = 32,
) -> PreparedFold:
    labels = df["attack_cat"].map(normalize_attack_label)
    available = labels.unique().tolist()
    norm = {normalize_name(x): x for x in available}
    if normalize_name(held_out_attack) not in norm:
        raise ValueError(f"Unknown held-out class {held_out_attack!r}; available: {available}")
    held = norm[normalize_name(held_out_attack)]

    train_known = labels[(df["_partition"] == "train") & (labels != held)].tolist()
    class_to_idx, idx_to_class = map_class_labels(train_known)
    unknown_idx = len(class_to_idx)
    y_all = np.asarray([class_to_idx.get(x, unknown_idx) for x in labels], dtype=np.int64)
    features = feature_columns_explicit(df, dataset, include_device)
    raw = df[features].copy()
    fit_mask = (df["_partition"].to_numpy() == "train") & (labels.to_numpy() != held)
    transformer = MedianMADTransformer().fit(raw.loc[fit_mask])
    X = transformer.transform(raw)
    selected_names = list(transformer.output_feature_names)
    if feature_cap is not None and X.shape[1] > feature_cap:
        variances = np.var(X[fit_mask], axis=0)
        selected = np.argsort(variances)[-feature_cap:]
        selected = np.sort(selected)
        X = X[:, selected]
        selected_names = [selected_names[i] for i in selected]

    arrays = {}
    for part in PARTITIONS:
        mask = df["_partition"].to_numpy() == part
        arrays[part] = make_group_windows(
            X[mask], y_all[mask], df.loc[mask, "_sequence_id"].to_numpy(),
            window, stride, unknown_idx, part != "test",
        )
    Xtr, ytr = arrays["train"]; Xcal, ycal = arrays["cal"]; Xte, yte = arrays["test"]
    if min(len(Xtr), len(Xcal), len(Xte)) == 0:
        raise ValueError(f"Empty split for {held}: train={len(Xtr)}, cal={len(Xcal)}, test={len(Xte)}")
    unknown = yte == unknown_idx
    rng = np.random.default_rng(12345)
    perm = rng.permutation(len(Xte)); Xte = Xte[perm]; yte = yte[perm]; unknown = unknown[perm]
    meta = {
        "dataset": dataset,
        "rows": int(len(df)),
        "held_out_attack": held,
        "known_classes": list(class_to_idx.keys()),
        "raw_feature_columns": len(features),
        "transformed_features": int(X.shape[1]),
        "feature_selection": "training-only variance ranking",
        "train_windows": int(len(Xtr)),
        "calibration_windows": int(len(Xcal)),
        "test_windows": int(len(Xte)),
        "unknown_test_windows": int(unknown.sum()),
        "window": window,
        "stride": stride,
        "include_device_feature": include_device,
    }
    return PreparedFold(Xtr, ytr, Xcal, ycal, Xte, yte, unknown,
                        class_to_idx, idx_to_class, held,
                        selected_names, meta)


class MLPClassifier(nn.Module):
    def __init__(self, t: int, f: int, k: int):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(t*f, 96), nn.GELU(), nn.Dropout(0.1),
                                 nn.Linear(96, 48), nn.GELU(), nn.Linear(48, k))
    def forward(self, x): return self.net(x)


class GRUClassifier(nn.Module):
    def __init__(self, f: int, k: int):
        super().__init__(); self.gru = nn.GRU(f, 48, batch_first=True); self.fc = nn.Linear(48, k)
    def forward(self, x):
        _, h = self.gru(x); return self.fc(h[-1])


class CNNGRUClassifier(nn.Module):
    def __init__(self, f: int, k: int):
        super().__init__(); self.conv = nn.Conv1d(f, 32, 3, padding=2); self.gru = nn.GRU(32, 48, batch_first=True); self.fc = nn.Linear(48, k)
    def forward(self, x):
        h = F.gelu(self.conv(x.transpose(1, 2))[:, :, :x.shape[1]]).transpose(1, 2)
        _, z = self.gru(h); return self.fc(z[-1])


class LSTMAE(nn.Module):
    def __init__(self, f: int):
        super().__init__(); self.enc = nn.LSTM(f, 48, batch_first=True); self.proj = nn.Linear(48, 32); self.dec = nn.LSTM(32, 48, batch_first=True); self.out = nn.Linear(48, f)
    def forward(self, x):
        _, (h, _) = self.enc(x); z = self.proj(h[-1]); rep = z[:, None, :].expand(-1, x.shape[1], -1); d, _ = self.dec(rep); return self.out(d)


def class_weights(y: np.ndarray, k: int) -> torch.Tensor:
    c = np.maximum(np.bincount(y, minlength=k).astype(float), 1.0)
    return torch.tensor(c.sum() / (k*c), dtype=torch.float32)


def train_classifier(model, fold: PreparedFold, seed: int, epochs: int, batch: int) -> nn.Module:
    set_seed(seed); w = class_weights(fold.y_train, len(fold.class_to_idx)); opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    tr = DataLoader(TensorDataset(torch.from_numpy(fold.X_train), torch.from_numpy(fold.y_train)), batch_size=batch, shuffle=True)
    va = DataLoader(TensorDataset(torch.from_numpy(fold.X_cal), torch.from_numpy(fold.y_cal)), batch_size=batch)
    best, bestv, bad = None, float("inf"), 0
    for _ in range(epochs):
        model.train()
        for xb, yb in tr:
            opt.zero_grad(); loss = F.cross_entropy(model(xb), yb, weight=w); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5); opt.step()
        model.eval(); total = count = 0
        with torch.no_grad():
            for xb, yb in va:
                l = F.cross_entropy(model(xb), yb, weight=w); total += float(l)*len(xb); count += len(xb)
        v = total/max(count, 1)
        if v < bestv - 1e-5: bestv = v; best = copy.deepcopy(model.state_dict()); bad = 0
        else:
            bad += 1
            if bad >= 4: break
    if best is not None: model.load_state_dict(best)
    return model


def infer_logits(model: nn.Module, X: np.ndarray, batch: int) -> np.ndarray:
    model.eval(); out = []
    with torch.no_grad():
        for (xb,) in DataLoader(TensorDataset(torch.from_numpy(X)), batch_size=batch): out.append(model(xb).numpy())
    return np.concatenate(out)


def open_metrics(y: np.ndarray, unknown: np.ndarray, pred: np.ndarray, score: np.ndarray, reject: np.ndarray, k: int, ece: float = float("nan")) -> Dict[str, float]:
    known = ~unknown; y_open = y.copy(); y_open[unknown] = k; final = pred.copy(); final[reject] = k
    try:
        auroc = roc_auc_score(unknown.astype(int), score); auprc = average_precision_score(unknown.astype(int), score)
        fpr, tpr, _ = roc_curve(unknown.astype(int), score); ids = np.flatnonzero(tpr >= 0.95); fpr95 = float(fpr[ids[0]]) if len(ids) else 1.0
    except ValueError:
        auroc = auprc = fpr95 = float("nan")
    return {
        "known_macro_f1": f1_score(y[known], pred[known], average="macro", zero_division=0),
        "known_mcc": matthews_corrcoef(y[known], pred[known]) if known.sum() > 1 else float("nan"),
        "unknown_auroc": auroc, "unknown_auprc": auprc, "fpr_at_95_tpr": fpr95,
        "open_set_macro_f1": f1_score(y_open, final, average="macro", zero_division=0),
        "known_acceptance_rate": float((~reject[known]).mean()) if known.any() else float("nan"),
        "unknown_rejection_rate": float(reject[unknown].mean()) if unknown.any() else float("nan"),
        "known_ece": ece,
    }


def classifier_baseline(name: str, model: nn.Module, fold: PreparedFold, seed: int, epochs: int, batch: int, alpha: float) -> Dict[str, float]:
    model = train_classifier(model, fold, seed, epochs, batch)
    lc = infer_logits(model, fold.X_cal, batch); lt = infer_logits(model, fold.X_test, batch)
    temp = fit_temperature(lc, fold.y_cal); pc = softmax_np(lc, temp); pt = softmax_np(lt, temp)
    score_cal = 1-pc.max(1); score = 1-pt.max(1); threshold = float(np.quantile(score_cal, 1-alpha))
    pred = pt.argmax(1); reject = score > threshold; known = ~fold.unknown_test
    ece = expected_calibration_error(pt[known], fold.y_test[known]) if known.any() else float("nan")
    return {"method": name, **open_metrics(fold.y_test, fold.unknown_test, pred, score, reject, len(fold.class_to_idx), ece)}


def isolation_baseline(fold: PreparedFold, seed: int, alpha: float) -> Dict[str, float]:
    benign_tr = fold.X_train[fold.y_train == 0].reshape((fold.y_train == 0).sum(), -1)
    benign_cal = fold.X_cal[fold.y_cal == 0].reshape((fold.y_cal == 0).sum(), -1)
    test = fold.X_test.reshape(len(fold.X_test), -1)
    model = IsolationForest(n_estimators=200, random_state=seed, n_jobs=1).fit(benign_tr)
    sc_cal = -model.score_samples(benign_cal); score = -model.score_samples(test); th = float(np.quantile(sc_cal, 1-alpha))
    pred = np.zeros(len(test), dtype=int); reject = score > th
    return {"method": "Isolation Forest", **open_metrics(fold.y_test, fold.unknown_test, pred, score, reject, len(fold.class_to_idx))}


def run_one_fold(fold: PreparedFold, dataset: str, out: Path, seed: int, epochs: int, batch: int, alpha: float, device: torch.device) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    mc = ModelConfig(); tc = TrainConfig(
        epochs=epochs, batch_size=batch, patience=6,
        lambda_rec=0.05, lambda_proto=0.01, lambda_sep=0.005,
        focal_gamma=0.0
    ); cc = CalibrationConfig(alpha=alpha)
    model, hist = train_model(fold, mc, tc, seed, device)
    raw_cal = infer_raw(model, fold.X_cal, batch, device); raw_test = infer_raw(model, fold.X_test, batch, device); cal = fit_calibrator(raw_cal, fold.y_cal, cc)
    complete = evaluate_variant(raw_test, fold.y_test, fold.unknown_test, cal, "complete", raw_cal, fold.y_cal)
    rows.append({"dataset": dataset, "method": "DART-IIoT", "held_out_attack": fold.held_out_attack, "seed": seed, **{k:v for k,v in complete.items() if k != "confusion_matrix"}, **profile_model(model, fold.X_test, device)})

    t, f = fold.X_train.shape[1:]; k = len(fold.class_to_idx)
    rows.append({"dataset": dataset, "held_out_attack": fold.held_out_attack, "seed": seed, **classifier_baseline("MLP-MSP", MLPClassifier(t,f,k), fold, seed, epochs, batch, alpha)})
    rows.append({"dataset": dataset, "held_out_attack": fold.held_out_attack, "seed": seed, **classifier_baseline("GRU-MSP", GRUClassifier(f,k), fold, seed, epochs, batch, alpha)})
    rows.append({"dataset": dataset, "held_out_attack": fold.held_out_attack, "seed": seed, **classifier_baseline("CNN-GRU-MSP", CNNGRUClassifier(f,k), fold, seed, epochs, batch, alpha)})
    rows.append({"dataset": dataset, "held_out_attack": fold.held_out_attack, "seed": seed, **isolation_baseline(fold, seed, alpha)})
    return rows


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    id_cols = ["dataset", "method"]
    numeric = [c for c in frame.columns if c not in {"dataset", "method", "held_out_attack", "seed", "variant", "device", "note"}]
    rows = []
    for keys, g in frame.groupby(id_cols):
        row = dict(zip(id_cols, keys)); row["n_runs"] = len(g)
        for c in numeric:
            v = pd.to_numeric(g[c], errors="coerce"); row[c+"_mean"] = v.mean(); row[c+"_std"] = v.std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["unsw", "nbaiot"], required=True)
    p.add_argument("--unsw-train", type=Path)
    p.add_argument("--unsw-test", type=Path)
    p.add_argument("--nbaiot-root", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--fold", action="append", default=[])
    p.add_argument("--seeds", default="1,7,21,42,84")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--window", type=int, default=16)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--alpha", type=float, default=0.025)
    p.add_argument("--max-train-per-class", type=int, default=20000)
    p.add_argument("--max-test-per-class", type=int, default=10000)
    p.add_argument("--max-rows-per-nbaiot-file", type=int, default=5000)
    p.add_argument("--include-device-feature", action="store_true")
    p.add_argument("--feature-cap", type=int, default=32)
    p.add_argument("--nbaiot-family-level", action="store_true")
    p.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    p.add_argument("--threads", type=int, default=1)
    return p.parse_args()


def main() -> None:
    global NBAIOT_FAMILY_LEVEL
    args = parse_args(); torch.set_num_threads(args.threads)
    NBAIOT_FAMILY_LEVEL = args.nbaiot_family_level
    if args.dataset == "unsw":
        if args.unsw_train is None or args.unsw_test is None: raise SystemExit("--unsw-train and --unsw-test are required")
        df = load_unsw(args.unsw_train, args.unsw_test, args.max_train_per_class, args.max_test_per_class); dataset_name = "UNSW-NB15"
    else:
        if args.nbaiot_root is None: raise SystemExit("--nbaiot-root is required")
        df = load_nbaiot(args.nbaiot_root, args.max_rows_per_nbaiot_file); dataset_name = "N-BaIoT"
    attacks = sorted(x for x in df.attack_cat.unique() if normalize_name(x) != "benign")
    folds = args.fold or attacks
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    args.output.mkdir(parents=True, exist_ok=True)
    df.groupby(["_partition", "attack_cat"]).size().rename("records").reset_index().to_csv(args.output/"record_distribution.csv", index=False)
    manifest_args = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    atomic_json(args.output/"manifest.json", {"dataset":dataset_name,"rows":len(df),"attacks":attacks,"folds":folds,"seeds":seeds,"args":manifest_args})
    rows = []
    for held in folds:
        fold = prepare_explicit_fold(df, dataset_name, held, args.window, args.stride, args.include_device_feature, args.feature_cap)
        print(f"[{dataset_name}] holdout={held} metadata={fold.metadata}", flush=True)
        for seed in seeds:
            print(f"  seed={seed}", flush=True)
            rows.extend(run_one_fold(fold, dataset_name, args.output, seed, args.epochs, args.batch_size, args.alpha, device))
            pd.DataFrame(rows).to_csv(args.output/"per_run_metrics.csv", index=False)
    frame = pd.DataFrame(rows); summary = summarize(frame); summary.to_csv(args.output/"summary_metrics.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__": main()
