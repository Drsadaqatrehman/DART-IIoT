#!/usr/bin/env python3

from pathlib import Path
import argparse
import numpy as np
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=Path("synthetic_iiot.csv"))
    p.add_argument("--rows", type=int, default=10000)
    p.add_argument("--features", type=int, default=16)
    args = p.parse_args()
    rng = np.random.default_rng(20260720)
    n, d = args.rows, args.features
    t = np.arange(n)
    X = rng.normal(0, 0.35, size=(n, d))
    X += 0.2 * np.sin(t[:, None] / np.linspace(17, 83, d)[None, :])
    labels = np.full(n, "Normal", dtype=object)

    # Place all attack families throughout the chronology so temporal folds remain usable.
    families = ["DoS", "Reconnaissance", "CommandInjection", "ResponseInjection"]
    for family_i, family in enumerate(families):
        for start in range(800 + family_i * 230, n - 120, 1800 + family_i * 170):
            length = 80 + family_i * 10
            sl = slice(start, min(start + length, n))
            labels[sl] = family
            if family == "DoS":
                X[sl, :4] += 3.0
            elif family == "Reconnaissance":
                X[sl, 4:8] += np.linspace(0.5, 2.5, length)[: len(X[sl]), None]
            elif family == "CommandInjection":
                X[sl, 8:12] *= -3.0
                X[sl, 9] += 2.0
            else:
                X[sl, 12:] += rng.laplace(0, 2.0, size=(len(X[sl]), d - 12))

    # Add benign operational drift only in the future test segment.
    drift_start = int(n * 0.82)
    benign_future = labels[drift_start:] == "Normal"
    X[drift_start:, :3][benign_future] += 0.55

    df = pd.DataFrame(X, columns=[f"feature_{i:02d}" for i in range(d)])
    df.insert(0, "timestamp", t)
    df["categorized_result"] = labels
    df["binary_result"] = np.where(labels == "Normal", "Normal", "Attack")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
