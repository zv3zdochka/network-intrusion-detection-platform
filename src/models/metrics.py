"""
Metrics and visualization utilities
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def calculate_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        task: str = "binary",
        class_names: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Рассчитать все метрики классификации

    Args:
        y_true: Истинные метки
        y_pred: Предсказанные метки
        y_proba: Вероятности классов
        task: "binary" или "multiclass"
        class_names: Названия классов

    Returns:
        Словарь с метриками
    """
    metrics = {}

    # Базовые метрики
    metrics['accuracy'] = accuracy_score(y_true, y_pred)

    if task == "binary":
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)

        if y_proba is not None:
            # Берём вероятность позитивного класса
            if len(y_proba.shape) > 1:
                y_proba_pos = y_proba[:, 1]
            else:
                y_proba_pos = y_proba

            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba_pos)
                metrics['pr_auc'] = average_precision_score(y_true, y_proba_pos)
            except ValueError:
                metrics['roc_auc'] = 0.0
                metrics['pr_auc'] = 0.0
    else:
        # Multiclass
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        if y_proba is not None:
            try:
                metrics['roc_auc_ovr'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
            except ValueError:
                metrics['roc_auc_ovr'] = 0.0

    # Confusion matrix stats
    cm = confusion_matrix(y_true, y_pred)
    if task == "binary":
        tn, fp, fn, tp = cm.ravel()
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['true_positives'] = int(tp)
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0

    return metrics


def classification_report_to_df(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """Преобразовать classification_report в DataFrame"""
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    df = pd.DataFrame(report).transpose()
    return df


def plot_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Optional[List[str]] = None,
        title: str = "Confusion Matrix",
        normalize: bool = True
) -> go.Figure:
    """Построить матрицу ошибок"""
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_display = cm_normalized
        text_template = '%{text}<br>%{z:.1%}'
    else:
        cm_display = cm
        text_template = '%{z:,}'

    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(cm))]

    # Создаём текстовые аннотации
    text = [[f'{cm[i][j]:,}' for j in range(len(cm[0]))] for i in range(len(cm))]

    fig = go.Figure(data=go.Heatmap(
        z=cm_display,
        x=class_names,
        y=class_names,
        text=text,
        texttemplate='%{text}',
        textfont={"size": 14},
        colorscale='Blues',
        hoverongaps=False
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Predicted',
        yaxis_title='Actual',
        height=500,
        width=600
    )

    return fig


def plot_roc_curve(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        model_name: str = "Model",
        title: str = "ROC Curve"
) -> go.Figure:
    """Построить ROC кривую"""
    if len(y_proba.shape) > 1:
        y_proba = y_proba[:, 1]

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    fig = go.Figure()

    # ROC curve
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'{model_name} (AUC = {auc:.3f})',
        line=dict(width=2)
    ))

    # Diagonal
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random',
        line=dict(dash='dash', color='gray')
    ))

    fig.update_layout(
        title=title,
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        height=500,
        width=600,
        showlegend=True
    )

    return fig


def plot_precision_recall_curve(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        model_name: str = "Model",
        title: str = "Precision-Recall Curve"
) -> go.Figure:
    """Построить Precision-Recall кривую"""
    if len(y_proba.shape) > 1:
        y_proba = y_proba[:, 1]

    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=recall, y=precision,
        mode='lines',
        name=f'{model_name} (AP = {ap:.3f})',
        line=dict(width=2)
    ))

    # Baseline
    baseline = y_true.mean()
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[baseline, baseline],
        mode='lines',
        name='Baseline',
        line=dict(dash='dash', color='gray')
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Recall',
        yaxis_title='Precision',
        height=500,
        width=600,
        showlegend=True
    )

    return fig


def plot_feature_importance(
        importance_df: pd.DataFrame,
        top_n: int = 20,
        title: str = "Feature Importance"
) -> go.Figure:
    """Построить график важности признаков"""
    df = importance_df.head(top_n).sort_values('importance', ascending=True)

    fig = go.Figure(go.Bar(
        x=df['importance'],
        y=df['feature'],
        orientation='h',
        marker_color='steelblue'
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Importance',
        yaxis_title='Feature',
        height=max(400, top_n * 25),
        width=700,
        margin=dict(l=200)
    )

    return fig


def plot_models_comparison(
        results: Dict[str, Dict[str, float]],
        metric: str = "f1",
        title: str = "Model Comparison"
) -> go.Figure:
    """Сравнение моделей по метрике"""
    models = list(results.keys())
    values = [results[m].get(metric, 0) for m in models]

    # Сортируем по убыванию
    sorted_pairs = sorted(zip(models, values), key=lambda x: x[1], reverse=True)
    models, values = zip(*sorted_pairs)

    colors = ['#2ecc71' if v == max(values) else '#3498db' for v in values]

    fig = go.Figure(go.Bar(
        x=list(models),
        y=list(values),
        marker_color=colors,
        text=[f'{v:.3f}' for v in values],
        textposition='outside'
    ))

    fig.update_layout(
        title=f'{title} ({metric.upper()})',
        xaxis_title='Model',
        yaxis_title=metric.upper(),
        height=500,
        width=800
    )

    return fig


def plot_multi_roc_curves(
        results: Dict[str, Tuple[np.ndarray, np.ndarray]],
        title: str = "ROC Curves Comparison"
) -> go.Figure:
    """Построить несколько ROC кривых"""
    fig = go.Figure()

    colors = px.colors.qualitative.Set1

    for i, (model_name, (y_true, y_proba)) in enumerate(results.items()):
        if len(y_proba.shape) > 1:
            y_proba = y_proba[:, 1]

        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)

        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'{model_name} (AUC = {auc:.3f})',
            line=dict(width=2, color=colors[i % len(colors)])
        ))

    # Diagonal
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random',
        line=dict(dash='dash', color='gray')
    ))

    fig.update_layout(
        title=title,
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        height=600,
        width=800,
        showlegend=True
    )

    return fig