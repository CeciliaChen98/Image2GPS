"""
Image2GPS: ResNet50 Model for GPS coordinate prediction.
Authors: Cecilia Chen, Ranty Wang, Xun Wang

This model uses:
- ResNet-50 backbone with pretrained ImageNet weights
- Custom regression head with batch normalization and dropout
- Grayscale augmentation during training (best performing configuration)
"""

import torch
import torch.nn as nn
import torchvision.models as models


# Normalization parameters from training data
NORM_PARAMS = {
    "lat_mean": 39.95170300059418,
    "lat_std": 0.0006492592147412374,
    "lon_mean": -75.19154888963162,
    "lon_std": 0.0006311150924374861
}


class RegressionHead(nn.Module):
    """Regression head with dropout and batch normalization."""

    def __init__(self, in_features, hidden_dim=256, dropout=0.5):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, 2)  # Output: [lat, lon]
        )

    def forward(self, x):
        return self.head(x)


class IMG2GPS(nn.Module):
    """
    ResNet50-based model for GPS coordinate prediction.

    Args:
        pretrained: Whether to use pretrained ImageNet weights
        hidden_dim: Hidden dimension for regression head
        dropout: Dropout rate for regularization
        norm_params: GPS normalization parameters
    """

    def __init__(self, pretrained=True, hidden_dim=256, dropout=0.5, norm_params=None):
        super().__init__()

        # Store normalization parameters
        if norm_params is None:
            norm_params = NORM_PARAMS
        self.lat_mean = norm_params['lat_mean']
        self.lat_std = norm_params['lat_std']
        self.lon_mean = norm_params['lon_mean']
        self.lon_std = norm_params['lon_std']

        # Build ResNet50 backbone
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = models.resnet50(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # Regression head
        self.regression_head = RegressionHead(in_features, hidden_dim, dropout)

    def forward(self, x):
        """
        Forward pass that outputs normalized coordinates.

        Args:
            x: Input tensor of images, shape (N, 3, 224, 224)

        Returns:
            Normalized coordinates, shape (N, 2)
        """
        features = self.backbone(x)
        return self.regression_head(features)

    def predict(self, x):
        """
        Make predictions and return denormalized GPS coordinates.

        Args:
            x: Input tensor or list of tensors, shape (N, 3, 224, 224)

        Returns:
            GPS coordinates [lat, lon] in raw degrees, shape (N, 2)
        """
        self.eval()
        with torch.no_grad():
            device = next(self.parameters()).device

            # Handle list input
            if isinstance(x, list):
                x = torch.stack(x)

            if x.device != device:
                x = x.to(device)

            normalized_output = self.forward(x)

            # Denormalize to get actual GPS coordinates
            lat = normalized_output[:, 0] * self.lat_std + self.lat_mean
            lon = normalized_output[:, 1] * self.lon_std + self.lon_mean

            predictions = torch.stack([lat, lon], dim=1)

        return predictions

    def get_num_parameters(self):
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def get_model():
    """
    Factory function to create and return a model instance.
    Called by the evaluation backend.
    """
    model = IMG2GPS(pretrained=False, hidden_dim=256, dropout=0.5)
    return model


# Alias for compatibility
Model = IMG2GPS
