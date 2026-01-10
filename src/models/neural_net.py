"""
Neural Network model (MLP)
"""

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from sklearn.neural_network import MLPClassifier
from .base import BaseModel


class NeuralNetModel(BaseModel):
    """Neural Network классификатор (sklearn MLP или PyTorch)"""

    def __init__(
            self,
            name: str = "NeuralNet",
            task: str = "binary",
            hidden_layers: Tuple[int, ...] = (128, 64, 32),
            activation: str = "relu",
            learning_rate: float = 0.001,
            batch_size: int = 256,
            max_epochs: int = 100,
            early_stopping: bool = True,
            dropout: float = 0.2,
            use_pytorch: bool = False,
            use_gpu: bool = False,
            random_state: int = 42,
            **kwargs
    ):
        super().__init__(name=name, task=task, random_state=random_state)

        self.hidden_layers = hidden_layers
        self.activation = activation
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.early_stopping = early_stopping
        self.dropout = dropout
        self.use_pytorch = use_pytorch and HAS_TORCH
        self.use_gpu = use_gpu and HAS_TORCH and torch.cuda.is_available()

        self.params.update({
            'hidden_layers': hidden_layers,
            'activation': activation,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'max_epochs': max_epochs,
            'dropout': dropout,
            'use_pytorch': self.use_pytorch
        })

        self.model = self._create_model()
        self.device = torch.device('cuda' if self.use_gpu else 'cpu') if HAS_TORCH else None

    def _create_model(self):
        if self.use_pytorch:
            return None  # Создадим при fit когда узнаем размерность
        else:
            return MLPClassifier(
                hidden_layer_sizes=self.hidden_layers,
                activation=self.activation,
                learning_rate_init=self.learning_rate,
                batch_size=self.batch_size,
                max_iter=self.max_epochs,
                early_stopping=self.early_stopping,
                validation_fraction=0.1 if self.early_stopping else 0.0,
                random_state=self.random_state,
                verbose=False
            )

    def _create_pytorch_model(self, input_dim: int, output_dim: int) -> nn.Module:
        """Создать PyTorch модель"""
        layers = []
        prev_dim = input_dim

        for hidden_dim in self.hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if self.activation == "relu":
                layers.append(nn.ReLU())
            elif self.activation == "tanh":
                layers.append(nn.Tanh())
            layers.append(nn.Dropout(self.dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        if output_dim == 1:
            layers.append(nn.Sigmoid())
        else:
            layers.append(nn.Softmax(dim=1))

        return nn.Sequential(*layers)

    def fit(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            feature_names: Optional[list] = None,
            **kwargs
    ) -> "NeuralNetModel":
        """Обучить модель"""
        self.feature_names = feature_names

        print(f"🧠 Training {self.name}...")
        print(f"   Parameters: layers={self.hidden_layers}, lr={self.learning_rate}")

        start_time = time.time()

        if self.use_pytorch:
            self._fit_pytorch(X_train, y_train, X_val, y_val)
        else:
            self.model.fit(X_train, y_train)

        self.training_time = time.time() - start_time
        self.is_fitted = True

        print(f"   ✅ Training completed in {self.training_time:.1f}s")

        return self

    def _fit_pytorch(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None
    ):
        """Обучение PyTorch модели"""
        input_dim = X_train.shape[1]
        output_dim = 1 if self.task == "binary" else len(np.unique(y_train))

        self.model = self._create_pytorch_model(input_dim, output_dim).to(self.device)

        # Подготовка данных
        X_tensor = torch.FloatTensor(X_train).to(self.device)
        y_tensor = torch.FloatTensor(y_train).to(self.device) if self.task == "binary" else torch.LongTensor(
            y_train).to(self.device)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Loss и optimizer
        if self.task == "binary":
            criterion = nn.BCELoss()
        else:
            criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Training loop
        self.model.train()
        for epoch in range(self.max_epochs):
            epoch_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                if self.task == "binary":
                    outputs = outputs.squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.use_pytorch:
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X).to(self.device)
                outputs = self.model(X_tensor)
                if self.task == "binary":
                    return (outputs.squeeze().cpu().numpy() > 0.5).astype(int)
                else:
                    return outputs.argmax(dim=1).cpu().numpy()
        else:
            return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.use_pytorch:
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X).to(self.device)
                outputs = self.model(X_tensor).cpu().numpy()
                if self.task == "binary":
                    proba = outputs.squeeze()
                    return np.column_stack([1 - proba, proba])
                else:
                    return outputs
        else:
            return self.model.predict_proba(X)