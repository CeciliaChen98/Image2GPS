"""
Image2GPS: Improved Model for GPS coordinate prediction.

Improvements over baseline:
1. ResNet-50 backbone (deeper than ResNet-18)
2. Additional fully connected layers with dropout for regularization
3. Batch normalization in the regression head
4. Support for multiple backbone options (ResNet-18, ResNet-50, EfficientNet)
"""

import torch
import torch.nn as nn
import torchvision.models as models


# Hardcoded normalization parameters from training data
# UPDATE THESE after training with your dataset!
NORM_PARAMS = {
    "lat_mean": 39.95170300059418,
    "lat_std": 0.0006492592147412374,
    "lon_mean": -75.19154888963162,
    "lon_std": 0.0006311150924374861
}


class RegressionHead(nn.Module):
    """
    Improved regression head with dropout and batch normalization.
    """
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
    Improved model for GPS coordinate prediction.

    Supports multiple backbone architectures:
    - resnet18: Lightweight, fast training
    - resnet50: Deeper, better feature extraction (default)
    - resnet101: Even deeper for complex patterns
    - efficientnet_b0: Efficient architecture with good accuracy
    """

    def __init__(self, backbone='resnet50', pretrained=True, hidden_dim=256,
                 dropout=0.5, norm_params=None):
        """
        Initialize the improved IMG2GPS model.

        Args:
            backbone: Backbone architecture ('resnet18', 'resnet50', 'resnet101', 'efficientnet_b0')
            pretrained: Whether to use pretrained weights
            hidden_dim: Hidden dimension for regression head
            dropout: Dropout rate for regularization
            norm_params: GPS normalization parameters
        """
        super().__init__()

        # Store normalization parameters
        if norm_params is None:
            norm_params = NORM_PARAMS
        self.lat_mean = norm_params['lat_mean']
        self.lat_std = norm_params['lat_std']
        self.lon_mean = norm_params['lon_mean']
        self.lon_std = norm_params['lon_std']

        self.backbone_name = backbone

        # Build backbone
        if backbone == 'resnet18':
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet18(weights=weights)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()

        elif backbone == 'resnet50':
            weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()

        elif backbone == 'resnet101':
            weights = models.ResNet101_Weights.IMAGENET1K_V2 if pretrained else None
            self.backbone = models.resnet101(weights=weights)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()

        elif backbone == 'efficientnet_b0':
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.efficientnet_b0(weights=weights)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()

        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # Improved regression head
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
            x: Input tensor of images, shape (N, 3, 224, 224)

        Returns:
            GPS coordinates [lat, lon] in raw degrees, shape (N, 2)
        """
        self.eval()
        with torch.no_grad():
            # Handle device
            device = next(self.parameters()).device
            if x.device != device:
                x = x.to(device)

            # Get normalized predictions
            normalized_output = self.forward(x)

            # Denormalize to get actual GPS coordinates
            lat = normalized_output[:, 0] * self.lat_std + self.lat_mean
            lon = normalized_output[:, 1] * self.lon_std + self.lon_mean

            # Stack back into (N, 2) tensor
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
    # Use ResNet-50 with improved head for submission
    # Change backbone here if needed
    model = IMG2GPS(backbone='resnet50', pretrained=False, hidden_dim=256, dropout=0.5)
    return model


# Alias for compatibility
Model = IMG2GPS
