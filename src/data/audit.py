from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .common import ensure_dir, write_json

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:
    plt = None


def _reservoir_sample(df: pd.DataFrame, k: int, rng: np.random.Generator) -> pd.DataFrame:
    """Sample up to k rows from df without keeping the full dataset in memory."""
    n = len(df)
    if k <= 0 or n == 0:
        return df.iloc[0:0]
    if n <= k:
        return df
    idx = rng.choice(n, size=k, replace=False)
    return df.iloc[idx]


def audit_bronze_dataset(
    bronze_dir: Path,
    reports_dir: Path,
    *,
    label_col: str = "label",
    timestamp_col: str = "timestamp",
    sample_rows: int = 100_000,
    max_corr_features: int = 40,
    seed: int = 42,
) -> Dict:
    """
    Reads parquet files from data/interim/bronze and produces:
      - reports/data_audit.json
      - figures in reports/figures/
        * label_distribution.png
        * missing_rate_top30.png
        * corr_heatmap.png
        * attack_share_by_file.png                (NEW)
        * missing_rate_by_file_heatmap.png        (NEW)
    """
    bronze_dir = bronze_dir.resolve()
    reports_dir = reports_dir.resolve()
    fig_dir = ensure_dir(reports_dir / "figures")

    files = sorted(bronze_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {bronze_dir}")

    rng = np.random.default_rng(seed)

    total_rows = 0
    label_counts: Dict[str, int] = {}
    missing_counts: Dict[str, int] = {}
    total_counts: Dict[str, int] = {}

    per_file_attack_share: Dict[str, float] = {}
    per_file_missing_rate: Dict[str, Dict[str, float]] = {}
    per_file_label_top: Dict[str, Dict[str, int]] = {}

    sample_df: Optional[pd.DataFrame] = None

    for fp in files:
        df = pd.read_parquet(fp)
        n = len(df)
        total_rows += n

        # stable file identifier
        if "source_file" in df.columns and df["source_file"].notna().any():
            source = str(df["source_file"].dropna().iloc[0])
        else:
            source = fp.name

        # Label distribution (do NOT cast to str; keep missing as missing)
        if label_col in df.columns:
            s = df[label_col]
            vc = s.value_counts(dropna=False)

            for k, v in vc.items():
                key = "(missing)" if pd.isna(k) else str(k)
                label_counts[key] = label_counts.get(key, 0) + int(v)

            top_vc = vc.head(12)
            per_file_label_top[source] = {
                ("(missing)" if pd.isna(k) else str(k)): int(v) for k, v in top_vc.items()
            }

            # Attack share per file (label != BENIGN)
            s_str = s.astype("string")
            is_attack = s_str.notna() & (s_str.str.upper() != "BENIGN")
            per_file_attack_share[source] = float(is_attack.sum() / max(1, n))

        # Missingness
        miss = df.isna().sum()
        per_file_missing_rate[source] = {c: float(v / max(1, n)) for c, v in miss.items()}

        for c, v in miss.items():
            missing_counts[c] = missing_counts.get(c, 0) + int(v)
            total_counts[c] = total_counts.get(c, 0) + int(n)

        # Sampling for heavier stats
        cur = df
        if sample_rows > 0:
            cur = _reservoir_sample(cur, min(sample_rows, len(cur)), rng)
            if sample_df is None:
                sample_df = cur
            else:
                sample_df = pd.concat([sample_df, cur], ignore_index=True)
                sample_df = _reservoir_sample(sample_df, min(sample_rows, len(sample_df)), rng)

    missing_rate = {c: float(missing_counts[c] / max(1, total_counts[c])) for c in total_counts}

    # Numeric summary on the sample
    numeric_summary: Dict[str, Dict[str, Optional[float]]] = {}
    if sample_df is not None:
        num_cols = sample_df.select_dtypes(include=[np.number]).columns.tolist()
        for c in num_cols:
            s = sample_df[c]
            if not s.notna().any():
                numeric_summary[c] = {
                    "mean": None, "std": None, "min": None,
                    "p01": None, "p50": None, "p99": None, "max": None
                }
            else:
                numeric_summary[c] = {
                    "mean": float(np.nanmean(s)),
                    "std": float(np.nanstd(s)),
                    "min": float(np.nanmin(s)),
                    "p01": float(np.nanpercentile(s, 1)),
                    "p50": float(np.nanpercentile(s, 50)),
                    "p99": float(np.nanpercentile(s, 99)),
                    "max": float(np.nanmax(s)),
                }

    report = {
        "bronze_dir": str(bronze_dir),
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "files": [fp.name for fp in files],
        "total_rows": int(total_rows),
        "label_counts": label_counts,
        "missing_rate": missing_rate,
        "numeric_summary_sampled": numeric_summary,
        "sample_rows_used": int(sample_rows),
        "per_file_attack_share": per_file_attack_share,
        "per_file_label_top": per_file_label_top,
    }

    write_json(reports_dir / "data_audit.json", report)

    # Figures
    if plt is not None:
        # label distribution
        if label_counts:
            lc = pd.Series(label_counts).sort_values(ascending=False)
            plt.figure(figsize=(13, 5))
            ax = plt.gca()
            ax.bar(lc.index.astype(str), lc.values)
            ax.set_yscale("log")
            ax.set_ylabel("count (log scale)")
            ax.set_title("Label distribution (bronze)")
            ax.grid(axis="y", alpha=0.25)
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(fig_dir / "label_distribution.png", dpi=220)
            plt.close()

        # missing rate top 30
        if missing_rate:
            mr = pd.Series(missing_rate).sort_values(ascending=False).head(30)
            plt.figure(figsize=(13, 5))
            ax = plt.gca()
            ax.bar(mr.index.astype(str), mr.values)
            ax.set_ylabel("missing rate")
            ax.set_title("Top missing-rate columns (top 30)")
            ax.grid(axis="y", alpha=0.25)
            plt.xticks(rotation=90, ha="center")
            plt.tight_layout()
            plt.savefig(fig_dir / "missing_rate_top30.png", dpi=220)
            plt.close()

        # correlation heatmap (color, sampled)
        if sample_df is not None:
            num = sample_df.select_dtypes(include=[np.number])
            if num.shape[1] > 2:
                var = num.var(numeric_only=True).sort_values(ascending=False)
                cols = var.head(max_corr_features).index.tolist()
                corr = num[cols].corr()

                plt.figure(figsize=(11, 9))
                im = plt.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
                plt.colorbar(im, fraction=0.046, pad=0.04)
                plt.xticks(range(len(cols)), cols, rotation=90, fontsize=7)
                plt.yticks(range(len(cols)), cols, fontsize=7)
                plt.title("Correlation heatmap (sampled, top-variance features)")
                plt.tight_layout()
                plt.savefig(fig_dir / "corr_heatmap.png", dpi=220)
                plt.close()

        # NEW: attack share by file
        if per_file_attack_share:
            s = pd.Series(per_file_attack_share).sort_values(ascending=False)
            plt.figure(figsize=(13, 5))
            ax = plt.gca()
            ax.bar(s.index.astype(str), 100.0 * s.values)
            ax.set_ylabel("attack share (%)")
            ax.set_title("Attack share by source_file (label != BENIGN)")
            ax.grid(axis="y", alpha=0.25)
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(fig_dir / "attack_share_by_file.png", dpi=220)
            plt.close()

        # NEW: missing-rate heatmap by file
        if per_file_missing_rate and missing_rate:
            top_cols = (
                pd.Series(missing_rate)
                .drop(labels=[label_col, timestamp_col], errors="ignore")
                .sort_values(ascending=False)
                .head(20)
                .index.tolist()
            )

            file_names = sorted(per_file_missing_rate.keys())
            mat = np.array([[per_file_missing_rate[f].get(c, 0.0) for c in top_cols] for f in file_names], dtype=float)

            plt.figure(figsize=(12, 5))
            vmax = float(np.nanmax(mat) if mat.size else 1.0)
            im = plt.imshow(mat, cmap="viridis", aspect="auto", vmin=0.0, vmax=vmax if vmax > 0 else 1.0)
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.yticks(range(len(file_names)), file_names, fontsize=7)
            plt.xticks(range(len(top_cols)), top_cols, rotation=90, fontsize=7)
            plt.title("Missing-rate by file (top missing columns)")
            plt.tight_layout()
            plt.savefig(fig_dir / "missing_rate_by_file_heatmap.png", dpi=220)
            plt.close()

    return report
