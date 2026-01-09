from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .common import ensure_dir, write_json, normalize_label

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:
    plt = None


def _reservoir_sample(df: pd.DataFrame, k: int, rng: np.random.Generator) -> pd.DataFrame:
    """
    Reservoir sampling by concatenation and random subsample (approx).
    For large datasets this is adequate for EDA.
    """
    if len(df) <= k:
        return df
    idx = rng.choice(len(df), size=k, replace=False)
    return df.iloc[idx]


def audit_bronze(
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
    - basic figures in reports/figures/
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

    sample_df: Optional[pd.DataFrame] = None

    for fp in files:
        df = pd.read_parquet(fp)
        total_rows += len(df)

        # label distribution
        if label_col in df.columns:
            s = df[label_col].astype("string")
            s = s.map(normalize_label, na_action="ignore")
            # Use a stable token for missing labels to keep JSON serializable
            vc = s.fillna("(missing)").value_counts(dropna=False)
            for k, v in vc.items():
                label_counts[k] = label_counts.get(k, 0) + int(v)

        # missingness
        miss = df.isna().sum()
        for c, v in miss.items():
            missing_counts[c] = missing_counts.get(c, 0) + int(v)
            total_counts[c] = total_counts.get(c, 0) + int(len(df))

        # sampling for heavier stats
        cur = df
        if sample_rows > 0:
            cur = _reservoir_sample(cur, min(sample_rows, len(cur)), rng)
            if sample_df is None:
                sample_df = cur
            else:
                # concat then downsample again
                sample_df = pd.concat([sample_df, cur], ignore_index=True)
                if len(sample_df) > sample_rows:
                    sample_df = _reservoir_sample(sample_df, sample_rows, rng)

    # Missing rates
    missing_rate = {c: missing_counts[c] / max(1, total_counts[c]) for c in total_counts}

    # Numeric summary
    numeric_summary: Dict[str, Dict] = {}
    if sample_df is not None:
        num_cols = sample_df.select_dtypes(include=[np.number]).columns.tolist()
        for c in num_cols:
            s = sample_df[c]
            numeric_summary[c] = {
                "mean": float(np.nanmean(s)) if s.notna().any() else None,
                "std": float(np.nanstd(s)) if s.notna().any() else None,
                "min": float(np.nanmin(s)) if s.notna().any() else None,
                "p01": float(np.nanpercentile(s, 1)) if s.notna().any() else None,
                "p50": float(np.nanpercentile(s, 50)) if s.notna().any() else None,
                "p99": float(np.nanpercentile(s, 99)) if s.notna().any() else None,
                "max": float(np.nanmax(s)) if s.notna().any() else None,
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
    }

    write_json(reports_dir / "data_audit.json", report)

    # Figures (optional)
    if plt is not None:
        try:
            plt.style.use("seaborn-v0_8-whitegrid")
        except Exception:
            pass

        # 1) Label distribution
        if label_counts:
            lc = pd.Series(label_counts).sort_values(ascending=False)

            # If labels are extremely imbalanced, a log scale is more informative.
            fig = plt.figure(figsize=(12, 4))
            ax = fig.add_subplot(1, 1, 1)
            lc.plot(kind="bar", ax=ax)
            ax.set_title("Label distribution (bronze)")
            ax.set_ylabel("count")
            ax.set_yscale("log")
            ax.tick_params(axis="x", labelrotation=45)
            for tick in ax.get_xticklabels():
                tick.set_horizontalalignment("right")
            fig.tight_layout()
            fig.savefig(fig_dir / "label_distribution.png", dpi=220)
            plt.close(fig)

        # 2) Missing-rate (top 30)
        if missing_rate:
            mr = pd.Series(missing_rate).sort_values(ascending=False).head(30)
            fig = plt.figure(figsize=(12, 4))
            ax = fig.add_subplot(1, 1, 1)
            mr.plot(kind="bar", ax=ax)
            ax.set_title("Top missing-rate columns (top 30)")
            ax.set_ylabel("missing rate")
            ax.set_ylim(0, min(1.0, float(mr.max()) * 1.15 if len(mr) else 1.0))
            ax.tick_params(axis="x", labelrotation=45)
            for tick in ax.get_xticklabels():
                tick.set_horizontalalignment("right")
            fig.tight_layout()
            fig.savefig(fig_dir / "missing_rate_top30.png", dpi=220)
            plt.close(fig)

        # 3) Correlation heatmap (sampled)
        if sample_df is not None:
            num = sample_df.select_dtypes(include=[np.number])
            if num.shape[1] > 2:
                var = num.var(numeric_only=True).sort_values(ascending=False)
                cols = var.head(max_corr_features).index.tolist()
                corr = num[cols].corr()

                fig = plt.figure(figsize=(12, 10))
                ax = fig.add_subplot(1, 1, 1)
                im = ax.imshow(corr.values, vmin=-1, vmax=1)
                ax.set_xticks(range(len(cols)))
                ax.set_yticks(range(len(cols)))
                ax.set_xticklabels(cols, rotation=90, fontsize=7)
                ax.set_yticklabels(cols, fontsize=7)
                ax.set_title("Correlation heatmap (sampled, top-variance features)")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                fig.tight_layout()
                fig.savefig(fig_dir / "corr_heatmap.png", dpi=220)
                plt.close(fig)

    return report

