"""
Data quality audit and EDA utilities for CIC-IDS-2017.

This module provides:
- run_audit(): data quality checks and summary statistics
- run_eda(): publication-ready PNG figures (English labels)
- generate_report(): JSON-only report (no HTML)
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .common import ensure_dir, format_number, get_project_root, load_config
from .ingest import load_bronze_data


def run_audit(
        df: Optional[pd.DataFrame] = None,
        config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Run data quality audit.

    Returns:
        A dictionary with audit results.
    """
    if config is None:
        config = load_config()

    if df is None:
        df = load_bronze_data(config)

    print("=" * 60)
    print("DATA QUALITY AUDIT")
    print("=" * 60)

    audit_results: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "basic_stats": {},
        "missing_values": {},
        "infinities": {},
        "duplicates": {},
        "data_types": {},
        "target_distribution": {},
        "issues": [],
    }

    # 1) Basic stats
    print("\nBasic statistics:")
    audit_results["basic_stats"] = {
        "total_rows": int(len(df)),
        "total_columns": int(len(df.columns)),
        "memory_mb": round(float(df.memory_usage(deep=True).sum() / (1024 * 1024)), 2),
    }
    print(f"  Rows: {format_number(len(df))}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Memory: {audit_results['basic_stats']['memory_mb']:.1f} MB")

    # 2) Data types
    print("\nData types:")
    dtype_counts = df.dtypes.astype(str).value_counts().to_dict()
    audit_results["data_types"] = {k: int(v) for k, v in dtype_counts.items()}
    for dtype, count in dtype_counts.items():
        print(f"  {dtype}: {count}")

    # 3) Missing values
    print("\nMissing values (NaN):")
    nan_counts = df.isna().sum()
    nan_cols = nan_counts[nan_counts > 0].sort_values(ascending=False)

    audit_results["missing_values"]["total_cells_with_nan"] = int(nan_counts.sum())
    audit_results["missing_values"]["columns_with_nan"] = int(len(nan_cols))
    audit_results["missing_values"]["details"] = {
        col: {"count": int(count), "percent": round(float(100 * count / len(df)), 4)}
        for col, count in nan_cols.items()
    }

    if len(nan_cols) > 0:
        print(f"  Columns with NaN: {len(nan_cols)}")
        for col, count in nan_cols.head(10).items():
            pct = 100 * count / len(df)
            print(f"  - {col}: {format_number(count)} ({pct:.2f}%)")
        audit_results["issues"].append(f"Found {len(nan_cols)} columns with NaN values")
    else:
        print("  No missing values detected.")

    # 4) Infinities
    print("\nInfinities (Inf):")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_counts: Dict[str, Any] = {}

    for col in numeric_cols:
        pos_inf = int((df[col] == np.inf).sum())
        neg_inf = int((df[col] == -np.inf).sum())
        total_inf = pos_inf + neg_inf
        if total_inf > 0:
            inf_counts[col] = {
                "positive_inf": pos_inf,
                "negative_inf": neg_inf,
                "total": total_inf,
                "percent": round(float(100 * total_inf / len(df)), 4),
            }

    audit_results["infinities"]["columns_with_inf"] = int(len(inf_counts))
    audit_results["infinities"]["details"] = inf_counts

    if inf_counts:
        print(f"  Columns with Inf: {len(inf_counts)}")
        for col, info in inf_counts.items():
            print(f"  - {col}: {format_number(info['total'])} ({info['percent']:.2f}%)")
        audit_results["issues"].append(f"Found {len(inf_counts)} columns with Inf values")
    else:
        print("  No infinities detected.")

    # 5) Duplicates
    print("\nDuplicates:")
    full_dups = int(df.duplicated().sum())
    audit_results["duplicates"]["full_duplicates"] = {
        "count": full_dups,
        "percent": round(float(100 * full_dups / len(df)), 2),
    }
    print(f"  Full duplicates: {format_number(full_dups)} ({100 * full_dups / len(df):.2f}%)")

    analysis_cols = [c for c in df.columns if not c.startswith("_")]
    dups_no_meta = int(df.duplicated(subset=analysis_cols).sum())
    audit_results["duplicates"]["without_meta"] = {
        "count": dups_no_meta,
        "percent": round(float(100 * dups_no_meta / len(df)), 2),
    }
    print(f"  Duplicates (excluding meta columns): {format_number(dups_no_meta)} ({100 * dups_no_meta / len(df):.2f}%)")

    if full_dups > 0:
        audit_results["issues"].append(
            f"Found {full_dups:,} duplicate rows ({100 * full_dups / len(df):.1f}%)"
        )

    # 6) Target distribution
    target_col = config["ingestion"]["target_column"]
    if target_col in df.columns:
        print(f"\nTarget distribution ({target_col}):")
        target_counts = df[target_col].value_counts()
        audit_results["target_distribution"] = {
            str(label): {"count": int(count), "percent": round(float(100 * count / len(df)), 2)}
            for label, count in target_counts.items()
        }

        for label, count in target_counts.items():
            pct = 100 * count / len(df)
            print(f"  {label}: {format_number(int(count))} ({pct:.2f}%)")

    # 7) Column naming issues
    print("\nColumn checks:")
    col_names = df.columns.tolist()
    duplicate_cols = [col for col in set(col_names) if col_names.count(col) > 1]

    if duplicate_cols:
        print(f"  Duplicate column names: {duplicate_cols}")
        audit_results["issues"].append(f"Duplicate column names: {duplicate_cols}")
    else:
        fwd_header_cols = [c for c in col_names if "Fwd Header Length" in c]
        if len(fwd_header_cols) > 1:
            print(f"  Similar column names detected: {fwd_header_cols}")
            audit_results["issues"].append(f"Similar column names found: {fwd_header_cols}")
        else:
            print("  Column names look consistent.")

    # 8) Quick extreme-value scan (first 20 numeric columns)
    print("\nExtreme values (quick scan):")
    extreme_cols = []
    for col in list(numeric_cols)[:20]:
        valid_data = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        if len(valid_data) > 0:
            q01, q99 = valid_data.quantile([0.01, 0.99])
            min_val, max_val = valid_data.min(), valid_data.max()
            if max_val > q99 * 100 or (q01 != 0 and min_val < q01 * 100):
                extreme_cols.append(col)

    if extreme_cols:
        print(f"  Columns with extreme outliers (heuristic): {len(extreme_cols)}")
        audit_results["issues"].append(f"Columns with extreme outliers: {extreme_cols[:5]}...")

    print("\n" + "=" * 60)
    print(f"TOTAL ISSUES: {len(audit_results['issues'])}")
    for issue in audit_results["issues"]:
        print(f"  - {issue}")
    print("=" * 60)

    return audit_results


def run_eda(
        df: Optional[pd.DataFrame] = None,
        config: Optional[Dict[str, Any]] = None,
        output_path: Optional[Path] = None,
) -> Dict[str, Path]:
    """
    Run EDA and export publication-ready PNG figures (English labels only).

    Produces figures:
    01_class_distribution
    02_day_distribution (if _day exists)
    03_class_imbalance_log
    04_correlation_matrix
    05_feature_distributions
    """
    if config is None:
        config = load_config()

    if df is None:
        df = load_bronze_data(config)

    root = get_project_root()
    if output_path is None:
        output_path = root / config["paths"]["reports"]

    figures_dir = output_path / "figures"
    ensure_dir(output_path)
    ensure_dir(figures_dir)

    # Optional: silence kaleido/choreographer noise in console output
    eda_cfg = config.get("eda", {})
    if eda_cfg.get("suppress_kaleido_logs", True):
        for name in ["kaleido", "choreographer"]:
            logging.getLogger(name).setLevel(logging.WARNING)

    created_files: Dict[str, Path] = {}

    sample_size = int(eda_cfg.get("sample_size", 100000))
    if len(df) > sample_size:
        df_sample = df.sample(n=sample_size, random_state=42)
        print(f"Using a sample of {sample_size:,} rows for heavy visualizations.")
    else:
        df_sample = df

    # Figure export settings (publication-friendly)
    fig_width = int(eda_cfg.get("figure_width", 1800))
    fig_height = int(eda_cfg.get("figure_height", 1100))
    fig_scale = int(eda_cfg.get("figure_scale", 4))  # scale=4 gives crisp PNGs
    template = eda_cfg.get("template", "plotly_white")

    target_col = config["ingestion"]["target_column"]

    def save_png(fig: go.Figure, filename: str) -> Path:
        fig.update_layout(
            template=template,
            font=dict(family="Arial", size=16),
            title_x=0.5,
            paper_bgcolor="white",
            plot_bgcolor="white",
            margin=dict(l=90, r=40, t=90, b=90),
        )
        path = figures_dir / f"{filename}.png"
        fig.write_image(str(path), width=fig_width, height=fig_height, scale=fig_scale)
        return path

    # 1) Class distribution (bar only)
    print("Creating figure 01: Class distribution...")
    fig = create_class_distribution_plot(df, target_col)
    created_files["class_distribution"] = save_png(fig, "01_class_distribution")

    # 2) Day distribution
    if "_day" in df.columns:
        print("Creating figure 02: Day distribution...")
        fig = create_day_distribution_plot(df, target_col)
        created_files["day_distribution"] = save_png(fig, "02_day_distribution")

    # 3) Class imbalance (log)
    print("Creating figure 03: Class imbalance (log scale)...")
    fig = create_class_imbalance_plot(df, target_col)
    created_files["class_imbalance"] = save_png(fig, "03_class_imbalance_log")

    # 4) Correlation matrix (sample)
    print("Creating figure 04: Correlation matrix...")
    fig = create_correlation_matrix(df_sample, config)
    created_files["correlation_matrix"] = save_png(fig, "04_correlation_matrix")

    # 5) Feature distributions (sample)
    print("Creating figure 05: Feature distributions...")
    fig = create_feature_distributions(df_sample, config)
    created_files["feature_distributions"] = save_png(fig, "05_feature_distributions")

    print(f"Created {len(created_files)} figures in: {figures_dir}")

    return created_files


def create_class_distribution_plot(df: pd.DataFrame, target_col: str) -> go.Figure:
    """Create a readable class distribution plot (horizontal bars, counts + percentages)."""
    class_counts = df[target_col].value_counts().reset_index()
    class_counts.columns = ["Class", "Count"]
    class_counts["Percentage"] = (class_counts["Count"] / len(df) * 100).round(2)

    # Sort descending for readability
    class_counts = class_counts.sort_values("Count", ascending=True)

    colors = ["#2ecc71" if c == "BENIGN" else "#e74c3c" for c in class_counts["Class"]]
    text = [f"{int(c):,} ({p:.2f}%)" for c, p in zip(class_counts["Count"], class_counts["Percentage"])]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=class_counts["Class"],
            x=class_counts["Count"],
            orientation="h",
            text=text,
            textposition="outside",
            marker_color=colors,
            hovertemplate="Class: %{y}<br>Count: %{x:,}<extra></extra>",
        )
    )

    fig.update_layout(
        title="CIC-IDS-2017: Class Distribution",
        xaxis_title="Number of flows",
        yaxis_title="Class",
        height=max(700, 40 * len(class_counts) + 250),
    )

    fig.update_xaxes(tickformat=",")
    return fig


def create_day_distribution_plot(df: pd.DataFrame, target_col: str) -> go.Figure:
    """Create a readable Benign vs Attack distribution by weekday."""
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    day_class = df.groupby(["_day", target_col]).size().reset_index(name="count")
    day_class["TrafficType"] = day_class[target_col].apply(lambda x: "Attack" if x != "BENIGN" else "Benign")
    day_binary = day_class.groupby(["_day", "TrafficType"])["count"].sum().reset_index()

    # Ensure ordering
    day_binary["_day"] = pd.Categorical(day_binary["_day"], categories=day_order, ordered=True)
    day_binary = day_binary.sort_values("_day")

    # Pivot for stacked bars
    pivot = day_binary.pivot_table(index="_day", columns="TrafficType", values="count", aggfunc="sum").fillna(0)
    pivot = pivot.loc[pivot.index.notna()]

    benign = pivot["Benign"].values if "Benign" in pivot.columns else np.zeros(len(pivot))
    attack = pivot["Attack"].values if "Attack" in pivot.columns else np.zeros(len(pivot))

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=pivot.index.astype(str),
            y=benign,
            name="Benign",
            marker_color="#2ecc71",
            hovertemplate="Day: %{x}<br>Benign: %{y:,}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=pivot.index.astype(str),
            y=attack,
            name="Attack",
            marker_color="#e74c3c",
            hovertemplate="Day: %{x}<br>Attack: %{y:,}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Benign vs Attack by Weekday",
        barmode="stack",
        xaxis_title="Weekday",
        yaxis_title="Number of flows",
        legend_title="Traffic type",
        height=650,
    )
    fig.update_yaxes(tickformat=",")
    return fig


def create_class_imbalance_plot(df: pd.DataFrame, target_col: str) -> go.Figure:
    """Create class imbalance plot with logarithmic x-axis (horizontal bars)."""
    class_counts = df[target_col].value_counts().sort_values(ascending=True)

    colors = ["#2ecc71" if c == "BENIGN" else "#e74c3c" for c in class_counts.index]
    text = [f"{int(v):,}" for v in class_counts.values]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=class_counts.index,
            x=class_counts.values,
            orientation="h",
            text=text,
            textposition="outside",
            marker_color=colors,
            hovertemplate="Class: %{y}<br>Count: %{x:,}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Class Imbalance (Log Scale)",
        xaxis_title="Number of flows (log scale)",
        yaxis_title="Class",
        xaxis_type="log",
        height=max(700, 35 * len(class_counts) + 250),
    )

    return fig


def create_correlation_matrix(df: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
    """Create correlation matrix for numeric features (top by variance)."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if not c.startswith("_")]

    max_features = int(config.get("eda", {}).get("max_features_corr", 30))
    if len(numeric_cols) > max_features:
        variances = df[numeric_cols].var().sort_values(ascending=False)
        numeric_cols = variances.head(max_features).index.tolist()

    df_clean = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    corr_matrix = df_clean.corr()

    fig = go.Figure(
        data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale="RdBu",
            zmin=-1,
            zmax=1,
            zmid=0,
            xgap=1,
            ygap=1,
            hovertemplate="x=%{x}<br>y=%{y}<br>corr=%{z:.3f}<extra></extra>",
            colorbar=dict(title="Correlation"),
        )
    )

    fig.update_layout(
        title="Correlation Matrix (Top Features by Variance)",
        height=1100,
        width=1200,
    )
    fig.update_xaxes(tickangle=45, automargin=True)
    fig.update_yaxes(automargin=True)
    return fig


def create_feature_distributions(df: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
    """Create histograms for numeric feature distributions (clipped to 1st-99th percentiles)."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if not c.startswith("_")]

    top_n = min(16, len(numeric_cols))
    cols_to_plot = numeric_cols[:top_n]

    n_cols = 4
    n_rows = (len(cols_to_plot) + n_cols - 1) // n_cols

    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=cols_to_plot)

    for i, col in enumerate(cols_to_plot):
        row = i // n_cols + 1
        col_idx = i % n_cols + 1

        valid_data = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        if len(valid_data) == 0:
            continue

        q01, q99 = valid_data.quantile([0.01, 0.99])
        clipped = valid_data.clip(q01, q99)

        fig.add_trace(
            go.Histogram(
                x=clipped,
                nbinsx=30,
                showlegend=False,
                hovertemplate=f"{col}<br>value=%{{x}}<br>count=%{{y}}<extra></extra>",
            ),
            row=row,
            col=col_idx,
        )

    fig.update_layout(
        title_text="Feature Distributions (Clipped to 1st–99th Percentiles)",
        height=280 * n_rows + 200,
        showlegend=False,
    )

    return fig


def generate_report(
        audit_results: Dict[str, Any],
        eda_files: Dict[str, Path],
        config: Optional[Dict[str, Any]] = None,
        output_path: Optional[Path] = None
) -> Path:
    """
    Generate a JSON-only report (no HTML).

    Writes:
      - reports/audit_report.json
    """
    if config is None:
        config = load_config()

    root = get_project_root()
    if output_path is None:
        output_path = root / config["paths"]["reports"]

    ensure_dir(output_path)

    report_payload = {
        "audit": audit_results,
        "figures": {name: str(path) for name, path in eda_files.items()},
        "generated_at": datetime.now().isoformat(),
    }

    report_path = output_path / "audit_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2, ensure_ascii=False, default=str)

    print(f"Report saved to: {report_path}")
    return report_path
