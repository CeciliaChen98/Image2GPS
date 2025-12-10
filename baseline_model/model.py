"""
Image2GPS: Model module for GPS coordinate prediction.

This module provides the model class required by the evaluation backend.
The model predicts GPS coordinates (latitude, longitude) from input images.

Weights are loaded from model.pt by the backend automatically.
"""

import torch
import torch.nn as nn
import torchvision.models as models


# Hardcoded normalization parameters from training data
# These MUST match the values used during training
# Update these values based on your actual training dataset statistics
NORM_PARAMS = {
    'lat_mean': 39.952478,   # Mean latitude of training data
    'lat_std': 0.001349,     # Std of latitude in training data
    'lon_mean': -75.193367,  # Mean longitude of training data
    'lon_std': 0.002270,     # Std of longitude in training data
}


class IMG2GPS(nn.Module):
    """
    ResNet-18 based model for GPS coordinate prediction.

    The model takes a batch of images and outputs GPS coordinates [lat, lon]
    in raw degrees (denormalized).

    The backend will load weights from model.pt into this model.
    """

    def __init__(self, norm_params=None):
        """
        Initialize the IMG2GPS model.

        Args:
            norm_params: Dictionary containing normalization parameters
                        (lat_mean, lat_std, lon_mean, lon_std).
                        If None, uses default hardcoded values.
        """
        super().__init__()

        # Store normalization parameters
        if norm_params is None:
            norm_params = NORM_PARAMS
        self.lat_mean = norm_params['lat_mean']
        self.lat_std = norm_params['lat_std']
        self.lon_mean = norm_params['lon_mean']
        self.lon_std = norm_params['lon_std']

        # Build ResNet-18 backbone
        self.resnet = models.resnet18(weights=None)

        # Modify the final layer to output 2 values (lat, lon)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, 2)

    def forward(self, x):
        """
        Forward pass that outputs normalized coordinates.

        Args:
            x: Input tensor of images, shape (N, 3, 224, 224)

        Returns:
            Normalized coordinates, shape (N, 2)
        """
        return self.resnet(x)

    def predict(self, x):
        """
        Make predictions and return denormalized GPS coordinates.

        This method is called by the evaluation backend.

        Args:
            x: Input tensor of images, shape (N, 3, 224, 224)

        Returns:
            GPS coordinates [lat, lon] in raw degrees, shape (N, 2)
        """
        self.eval()
        with torch.no_grad():
            # Get normalized predictions
            normalized_output = self.forward(x)

            # Denormalize to get actual GPS coordinates
            lat = normalized_output[:, 0] * self.lat_std + self.lat_mean
            lon = normalized_output[:, 1] * self.lon_std + self.lon_mean

            # Stack back into (N, 2) tensor
            predictions = torch.stack([lat, lon], dim=1)

        return predictions


def get_model():
    """
    Factory function to create and return a model instance.

    This function is called by the evaluation backend.
    The backend will then load model.pt weights into this model.

    Returns:
        An instance of IMG2GPS model
    """
    return IMG2GPS()


# Also provide Model alias for compatibility
Model = IMG2GPS
