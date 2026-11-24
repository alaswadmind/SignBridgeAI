"""Inference for the coordinate-based SignBridge MLP checkpoint."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parent
BUNDLE = ROOT / "signbridge_model"


class SignMLP(nn.Module):
    """Exact MLP topology used in the supplied training notebook."""

    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SignBridgePredictor:
    def __init__(self, bundle: Path = BUNDLE):
        checkpoint_path = bundle / "sign_mlp_checkpoint.pt"
        scaler_path = bundle / "standard_scaler.pkl"
        encoder_path = bundle / "label_encoder.pkl"
        map_path = bundle / "label_map.json"
        for path in (checkpoint_path, scaler_path, encoder_path, map_path):
            if not path.is_file():
                raise FileNotFoundError(f"Required model asset missing: {path}")

        # The checkpoint is a plain state_dict and metadata; keep deserialization restricted.
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        with scaler_path.open("rb") as f:
            self.scaler = pickle.load(f)
        with encoder_path.open("rb") as f:
            self.encoder = pickle.load(f)
        self.label_map = json.loads(map_path.read_text(encoding="utf-8"))

        self.feature_cols = [str(c) for c in checkpoint["feature_cols"]]
        self.in_features = int(checkpoint["in_features"])
        self.num_classes = int(checkpoint["num_classes"])
        if self.in_features != 89 or len(self.feature_cols) != self.in_features:
            raise ValueError(f"Expected 89 coordinate/angle features; checkpoint declares {self.in_features}")
        if int(self.scaler.n_features_in_) != self.in_features:
            raise ValueError("StandardScaler feature count does not match the MLP")
        if len(self.encoder.classes_) != self.num_classes:
            raise ValueError("LabelEncoder class count does not match the MLP")

        expected_cols = [str(i) for i in range(4096, 4185)]
        if self.feature_cols != expected_cols:
            raise ValueError("Checkpoint feature order differs from the extracted 4096..4184 feature order")

        self.model = SignMLP(self.in_features, self.num_classes)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.model.eval()

    def predict(self, features: list[float] | np.ndarray) -> dict:
        vector = np.asarray(features, dtype=np.float64).reshape(1, -1)
        if vector.shape[1] != self.in_features:
            raise ValueError(f"Expected {self.in_features} features, received {vector.shape[1]}")
        if not np.isfinite(vector).all():
            raise ValueError("Input feature vector contains NaN or infinity")

        scaled = self.scaler.transform(vector).astype(np.float32, copy=False)
        with torch.inference_mode():
            logits = self.model(torch.from_numpy(scaled))
            probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
        top = np.argsort(probabilities)[::-1][:3]
        candidates = []
        for encoded_id in top:
            original_label = self.encoder.inverse_transform([int(encoded_id)])[0]
            original_label = int(original_label)
            info = self.label_map.get(str(original_label), {"name": str(original_label), "arabic": str(original_label)})
            candidates.append({
                "class_id": original_label,
                "name": info["name"],
                "arabic": info["arabic"],
                "confidence": float(probabilities[encoded_id]),
            })
        return {"prediction": candidates[0], "candidates": candidates}
