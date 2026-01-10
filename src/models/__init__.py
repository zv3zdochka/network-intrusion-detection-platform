"""
Models module for intrusion detection
"""

from .base import BaseModel
from .random_forest import RandomForestModel
from .xgboost_model import XGBoostModel
from .lightgbm_model import LightGBMModel
from .neural_net import NeuralNetModel
from .ensemble import EnsembleModel
from .metrics import (
    calculate_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_feature_importance,
    classification_report_to_df
)

__all__ = [
    "BaseModel",
    "RandomForestModel",
    "XGBoostModel",
    "LightGBMModel",
    "NeuralNetModel",
    "EnsembleModel",
    "calculate_metrics",
    "plot_confusion_matrix",
    "plot_roc_curve",
    "plot_precision_recall_curve",
    "plot_feature_importance",
    "classification_report_to_df"
]
