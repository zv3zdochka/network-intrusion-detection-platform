"""
Visualization and reporting for simulation results.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class SimulationReporter:
    """
    Generate visualizations and reports from simulation results.
    """

    def __init__(self, output_dir: str = "reports/simulation"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_full_report(
            self,
            report: Dict[str, Any],
            snapshots: List[Dict[str, Any]],
            run_id: Optional[int] = None,
            save_html: bool = False
    ) -> Dict[str, Path]:
        """
        Generate all visualizations for a simulation run.

        Returns:
            Dict mapping visualization name to file path
        """
        suffix = f"_{run_id}" if run_id else ""
        created_files = {}

        # 1. Confusion Matrix
        fig = self.plot_confusion_matrix(report["classification"])
        path = self._save_figure(fig, f"confusion_matrix{suffix}", save_html)
        created_files["confusion_matrix"] = path

        # 2. Metrics Over Time
        if snapshots:
            fig = self.plot_metrics_over_time(snapshots)
            path = self._save_figure(fig, f"metrics_timeline{suffix}", save_html)
            created_files["metrics_timeline"] = path

        # 3. Throughput Over Time
        if snapshots:
            fig = self.plot_throughput_over_time(snapshots)
            path = self._save_figure(fig, f"throughput_timeline{suffix}", save_html)
            created_files["throughput_timeline"] = path

        # 4. Classification Metrics Bar Chart
        fig = self.plot_classification_metrics(report["classification"])
        path = self._save_figure(fig, f"classification_metrics{suffix}", save_html)
        created_files["classification_metrics"] = path

        # 5. Summary Dashboard
        fig = self.plot_summary_dashboard(report, snapshots)
        path = self._save_figure(fig, f"dashboard{suffix}", save_html)
        created_files["dashboard"] = path

        # 6. Latency Distribution (if available)
        if snapshots and any(s.get("latency_p95_ms", 0) > 0 for s in snapshots):
            fig = self.plot_latency_over_time(snapshots)
            path = self._save_figure(fig, f"latency_timeline{suffix}", save_html)
            created_files["latency_timeline"] = path

        # 7. Alerts Over Time
        if snapshots:
            fig = self.plot_alerts_over_time(snapshots)
            path = self._save_figure(fig, f"alerts_timeline{suffix}", save_html)
            created_files["alerts_timeline"] = path

        print(f"Generated {len(created_files)} visualizations in {self.output_dir}")

        return created_files

    def plot_confusion_matrix(
            self,
            classification: Dict[str, Any],
            title: str = "Confusion Matrix"
    ) -> go.Figure:
        """Plot confusion matrix."""
        tp = classification["true_positives"]
        fp = classification["false_positives"]
        tn = classification["true_negatives"]
        fn = classification["false_negatives"]

        # Absolute values
        cm = np.array([[tn, fp], [fn, tp]])

        # Normalized (by row)
        cm_norm = cm.astype(float)
        cm_norm[0] = cm_norm[0] / cm_norm[0].sum() if cm_norm[0].sum() > 0 else 0
        cm_norm[1] = cm_norm[1] / cm_norm[1].sum() if cm_norm[1].sum() > 0 else 0

        # Create text annotations
        text = [[f"{cm[i][j]:,}<br>({cm_norm[i][j] * 100:.2f}%)"
                 for j in range(2)] for i in range(2)]

        fig = go.Figure(data=go.Heatmap(
            z=cm_norm,
            x=["Predicted Benign", "Predicted Attack"],
            y=["Actual Benign", "Actual Attack"],
            text=text,
            texttemplate="%{text}",
            textfont={"size": 16},
            colorscale="Blues",
            showscale=True,
            colorbar=dict(title="Ratio")
        ))

        fig.update_layout(
            title=dict(text=title, x=0.5),
            height=500,
            width=600,
            xaxis_title="Predicted",
            yaxis_title="Actual"
        )

        return fig

    def plot_metrics_over_time(
            self,
            snapshots: List[Dict[str, Any]],
            title: str = "Classification Metrics Over Time"
    ) -> go.Figure:
        """Plot precision, recall, F1 over time."""
        df = pd.DataFrame(snapshots)

        fig = go.Figure()

        metrics = [
            ("precision", "Precision", "#2ecc71"),
            ("recall", "Recall", "#3498db"),
            ("f1", "F1 Score", "#e74c3c"),
            ("accuracy", "Accuracy", "#9b59b6")
        ]

        for col, name, color in metrics:
            if col in df.columns:
                fig.add_trace(go.Scatter(
                    x=df["elapsed_seconds"],
                    y=df[col],
                    mode="lines+markers",
                    name=name,
                    line=dict(color=color, width=2),
                    marker=dict(size=6)
                ))

        fig.update_layout(
            title=dict(text=title, x=0.5),
            xaxis_title="Elapsed Time (seconds)",
            yaxis_title="Score",
            yaxis=dict(range=[0.99, 1.001]),  # Zoom in since values are high
            height=500,
            width=900,
            legend=dict(x=0.02, y=0.02),
            hovermode="x unified"
        )

        return fig

    def plot_throughput_over_time(
            self,
            snapshots: List[Dict[str, Any]],
            title: str = "Throughput Over Time"
    ) -> go.Figure:
        """Plot flows per second over time."""
        df = pd.DataFrame(snapshots)

        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Flows per Second", "Cumulative Flows Processed"),
            vertical_spacing=0.15
        )

        # Throughput
        fig.add_trace(go.Scatter(
            x=df["elapsed_seconds"],
            y=df["flows_per_second"],
            mode="lines+markers",
            name="Flows/sec",
            line=dict(color="#3498db", width=2),
            fill="tozeroy",
            fillcolor="rgba(52, 152, 219, 0.2)"
        ), row=1, col=1)

        # Cumulative
        fig.add_trace(go.Scatter(
            x=df["elapsed_seconds"],
            y=df["flows_processed"],
            mode="lines+markers",
            name="Total Flows",
            line=dict(color="#2ecc71", width=2)
        ), row=2, col=1)

        fig.update_layout(
            title=dict(text=title, x=0.5),
            height=600,
            width=900,
            showlegend=True
        )

        fig.update_xaxes(title_text="Elapsed Time (seconds)", row=2, col=1)
        fig.update_yaxes(title_text="Flows/sec", row=1, col=1)
        fig.update_yaxes(title_text="Total Flows", row=2, col=1)

        return fig

    def plot_classification_metrics(
            self,
            classification: Dict[str, Any],
            title: str = "Classification Metrics"
    ) -> go.Figure:
        """Bar chart of classification metrics."""
        metrics = ["precision", "recall", "f1", "accuracy"]
        values = [classification.get(m, 0) for m in metrics]

        colors = ["#2ecc71", "#3498db", "#e74c3c", "#9b59b6"]

        fig = go.Figure(data=go.Bar(
            x=[m.upper() for m in metrics],
            y=values,
            marker_color=colors,
            text=[f"{v:.4f}" for v in values],
            textposition="outside"
        ))

        fig.update_layout(
            title=dict(text=title, x=0.5),
            yaxis=dict(range=[0.99, 1.005]),
            height=400,
            width=600,
            showlegend=False
        )

        return fig

    def plot_alerts_over_time(
            self,
            snapshots: List[Dict[str, Any]],
            title: str = "Alerts Over Time"
    ) -> go.Figure:
        """Plot alert generation over time."""
        df = pd.DataFrame(snapshots)

        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Alerts per Second", "Cumulative: TP vs FP"),
            vertical_spacing=0.15
        )

        # Alerts per second
        fig.add_trace(go.Scatter(
            x=df["elapsed_seconds"],
            y=df["alerts_per_second"],
            mode="lines+markers",
            name="Alerts/sec",
            line=dict(color="#e74c3c", width=2),
            fill="tozeroy",
            fillcolor="rgba(231, 76, 60, 0.2)"
        ), row=1, col=1)

        # TP vs FP cumulative
        fig.add_trace(go.Scatter(
            x=df["elapsed_seconds"],
            y=df["true_positives"],
            mode="lines",
            name="True Positives",
            line=dict(color="#2ecc71", width=2)
        ), row=2, col=1)

        fig.add_trace(go.Scatter(
            x=df["elapsed_seconds"],
            y=df["false_positives"],
            mode="lines",
            name="False Positives",
            line=dict(color="#e74c3c", width=2)
        ), row=2, col=1)

        fig.update_layout(
            title=dict(text=title, x=0.5),
            height=600,
            width=900,
            showlegend=True
        )

        return fig

    def plot_latency_over_time(
            self,
            snapshots: List[Dict[str, Any]],
            title: str = "Inference Latency Over Time"
    ) -> go.Figure:
        """Plot latency percentiles over time."""
        df = pd.DataFrame(snapshots)

        fig = go.Figure()

        latency_cols = [
            ("latency_p50_ms", "p50", "#2ecc71"),
            ("latency_p95_ms", "p95", "#f39c12"),
            ("latency_p99_ms", "p99", "#e74c3c")
        ]

        for col, name, color in latency_cols:
            if col in df.columns:
                fig.add_trace(go.Scatter(
                    x=df["elapsed_seconds"],
                    y=df[col],
                    mode="lines+markers",
                    name=name,
                    line=dict(color=color, width=2)
                ))

        fig.update_layout(
            title=dict(text=title, x=0.5),
            xaxis_title="Elapsed Time (seconds)",
            yaxis_title="Latency (ms)",
            height=400,
            width=900,
            legend=dict(x=0.02, y=0.98)
        )

        return fig

    def plot_summary_dashboard(
            self,
            report: Dict[str, Any],
            snapshots: List[Dict[str, Any]],
            title: str = "Simulation Summary Dashboard"
    ) -> go.Figure:
        """Create a comprehensive dashboard."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Classification Metrics",
                "Confusion Matrix",
                "Throughput Over Time",
                "F1 Score Stability"
            ),
            specs=[
                [{"type": "bar"}, {"type": "heatmap"}],
                [{"type": "scatter"}, {"type": "scatter"}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )

        classification = report["classification"]

        # 1. Metrics Bar Chart
        metrics = ["precision", "recall", "f1", "accuracy"]
        values = [classification.get(m, 0) for m in metrics]
        colors = ["#2ecc71", "#3498db", "#e74c3c", "#9b59b6"]

        fig.add_trace(go.Bar(
            x=[m.upper() for m in metrics],
            y=values,
            marker_color=colors,
            text=[f"{v:.4f}" for v in values],
            textposition="outside",
            showlegend=False
        ), row=1, col=1)

        # 2. Confusion Matrix
        tp = classification["true_positives"]
        fp = classification["false_positives"]
        tn = classification["true_negatives"]
        fn = classification["false_negatives"]
        cm = np.array([[tn, fp], [fn, tp]])
        cm_norm = cm.astype(float)
        cm_norm[0] = cm_norm[0] / cm_norm[0].sum() if cm_norm[0].sum() > 0 else 0
        cm_norm[1] = cm_norm[1] / cm_norm[1].sum() if cm_norm[1].sum() > 0 else 0

        fig.add_trace(go.Heatmap(
            z=cm_norm,
            x=["Pred Benign", "Pred Attack"],
            y=["Benign", "Attack"],
            colorscale="Blues",
            showscale=False,
            text=[[f"{cm[i][j]:,}" for j in range(2)] for i in range(2)],
            texttemplate="%{text}",
            textfont={"size": 12}
        ), row=1, col=2)

        if snapshots:
            df = pd.DataFrame(snapshots)

            # 3. Throughput
            fig.add_trace(go.Scatter(
                x=df["elapsed_seconds"],
                y=df["flows_per_second"],
                mode="lines",
                name="Flows/sec",
                line=dict(color="#3498db", width=2),
                fill="tozeroy",
                fillcolor="rgba(52, 152, 219, 0.2)",
                showlegend=False
            ), row=2, col=1)

            # 4. F1 Over Time
            fig.add_trace(go.Scatter(
                x=df["elapsed_seconds"],
                y=df["f1"],
                mode="lines+markers",
                name="F1",
                line=dict(color="#e74c3c", width=2),
                showlegend=False
            ), row=2, col=2)

        # Update layout
        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=18)),
            height=700,
            width=1100,
            showlegend=False
        )

        # Update axes
        fig.update_yaxes(range=[0.99, 1.002], row=1, col=1)
        fig.update_xaxes(title_text="Time (s)", row=2, col=1)
        fig.update_xaxes(title_text="Time (s)", row=2, col=2)
        fig.update_yaxes(title_text="Flows/sec", row=2, col=1)
        fig.update_yaxes(title_text="F1 Score", range=[0.998, 1.001], row=2, col=2)

        return fig

    def _save_figure(
            self,
            fig: go.Figure,
            name: str,
            save_html: bool = True
    ) -> Path:
        """Save figure as PNG and optionally HTML."""
        png_path = self.output_dir / f"{name}.png"
        fig.write_image(str(png_path), width=1200, height=800, scale=2)

        if save_html:
            html_path = self.output_dir / f"{name}.html"
            fig.write_html(str(html_path))

        return png_path

    def generate_text_report(
            self,
            report: Dict[str, Any],
            run_id: Optional[int] = None
    ) -> str:
        """Generate text summary report."""
        summary = report["summary"]
        classification = report["classification"]
        latency = report["latency"]

        text = f"""
================================================================================
SIMULATION REPORT {f'(Run #{run_id})' if run_id else ''}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
================================================================================

SUMMARY
-------
Total Flows Processed: {summary['total_flows']:,}
Total Alerts Generated: {summary['total_alerts']:,}
Elapsed Time: {summary['elapsed_seconds']:.2f} seconds
Throughput: {summary['throughput_flows_per_sec']:.0f} flows/sec

CLASSIFICATION METRICS
----------------------
Precision: {classification['precision']:.4f}
Recall:    {classification['recall']:.4f}
F1 Score:  {classification['f1']:.4f}
Accuracy:  {classification['accuracy']:.4f}

CONFUSION MATRIX
----------------
                    Predicted
                    Benign      Attack
Actual Benign      {classification['true_negatives']:>10,}  {classification['false_positives']:>10,}
Actual Attack      {classification['false_negatives']:>10,}  {classification['true_positives']:>10,}

LATENCY (ms)
------------
p50: {latency['p50_ms']:.2f}
p95: {latency['p95_ms']:.2f}
p99: {latency['p99_ms']:.2f}
Mean: {latency['mean_ms']:.2f}
Max: {latency['max_ms']:.2f}

================================================================================
"""
        return text