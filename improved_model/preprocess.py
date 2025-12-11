"""
Image2GPS: Preprocessing module for GPS coordinate prediction.

This module provides:
1. prepare_data() - Required by evaluation backend
2. Training transforms with augmentation
3. Custom dataset class for training
"""

import os
import pandas as pd
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image


# ============================================================================
# Transforms
# ============================================================================

def get_train_transform(use_grayscale=True, use_blur=True, use_erasing=True):
    """
    Get training transform with configurable data augmentation.

    Augmentations help the model generalize better to:
    - Different lighting conditions
    - Different camera angles
    - Weather variations

    Args:
        use_grayscale: Enable RandomGrayscale (simulates weather/lighting)
        use_blur: Enable GaussianBlur (simulates camera focus variation)
        use_erasing: Enable RandomErasing (simulates occlusions)

    Returns:
        Composed transform pipeline
    """
    transform_list = [
        # Base augmentations (always enabled)
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.1
        ),
    ]

    # Optional augmentations
    if use_grayscale:
        transform_list.append(transforms.RandomGrayscale(p=0.1))  # Simulate different weather

    if use_blur:
        transform_list.append(transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)))  # Camera focus variation

    # ToTensor and Normalize (always required)
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # RandomErasing must come after ToTensor
    if use_erasing:
        transform_list.append(transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)))  # Simulate occlusions

    return transforms.Compose(transform_list)


def get_val_transform():
    """Get validation/inference transform (no augmentation)."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


# Inference transform for prepare_data
INFERENCE_TRANSFORM = get_val_transform()


# ============================================================================
# Dataset Class for Training
# ============================================================================

class GPSImageDataset(Dataset):
    """
    Custom Dataset for GPS coordinate prediction.

    Handles both HuggingFace datasets and local CSV files.
    """

    def __init__(self, data_source, transform=None,
                 lat_mean=None, lat_std=None, lon_mean=None, lon_std=None,
                 is_huggingface=True):
        """
        Initialize the dataset.

        Args:
            data_source: HuggingFace dataset or pandas DataFrame
            transform: Image transforms to apply
            lat_mean, lat_std, lon_mean, lon_std: Normalization parameters
            is_huggingface: Whether data_source is a HuggingFace dataset
        """
        self.transform = transform
        self.is_huggingface = is_huggingface

        if is_huggingface:
            self.data = data_source
            latitudes = np.array(self.data['Latitude'])
            longitudes = np.array(self.data['Longitude'])
        else:
            self.data = data_source
            self.csv_dir = getattr(data_source, 'csv_dir', '')
            latitudes = self.data['latitude'].values
            longitudes = self.data['longitude'].values

        # Compute or use provided normalization parameters
        self.lat_mean = lat_mean if lat_mean is not None else np.mean(latitudes)
        self.lat_std = lat_std if lat_std is not None else np.std(latitudes)
        self.lon_mean = lon_mean if lon_mean is not None else np.mean(longitudes)
        self.lon_std = lon_std if lon_std is not None else np.std(longitudes)

        # Prevent division by zero
        if self.lat_std == 0:
            self.lat_std = 1e-6
        if self.lon_std == 0:
            self.lon_std = 1e-6

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.is_huggingface:
            example = self.data[idx]
            image = example['image']
            latitude = example['Latitude']
            longitude = example['Longitude']
        else:
            row = self.data.iloc[idx]
            image_path = row['image_path']
            if not os.path.isabs(image_path):
                image_path = os.path.join(self.csv_dir, image_path)
            image = Image.open(image_path)
            latitude = row['latitude']
            longitude = row['longitude']

        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Normalize GPS coordinates
        lat_norm = (latitude - self.lat_mean) / self.lat_std
        lon_norm = (longitude - self.lon_mean) / self.lon_std
        gps_coords = torch.tensor([lat_norm, lon_norm], dtype=torch.float32)

        return image, gps_coords

    def get_normalization_params(self):
        """Return normalization parameters as a dictionary."""
        return {
            'lat_mean': float(self.lat_mean),
            'lat_std': float(self.lat_std),
            'lon_mean': float(self.lon_mean),
            'lon_std': float(self.lon_std)
        }


# ============================================================================
# Prepare Data Function (Required by Backend)
# ============================================================================

def find_column(df, possible_names):
    """Find a column in the dataframe from a list of possible names."""
    for name in possible_names:
        if name in df.columns:
            return name
    raise ValueError(f"Could not find column. Tried: {possible_names}. Available: {list(df.columns)}")


def load_and_preprocess_image(image_path, transform):
    """Load an image from path and apply preprocessing transforms."""
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return transform(image)


def prepare_data(csv_path: str):
    """
    Prepare data for model evaluation.

    Called by the evaluation backend with the path to a CSV file.

    Args:
        csv_path: Path to the CSV file containing image paths and coordinates

    Returns:
        X: Tensor of preprocessed images, shape (N, 3, 224, 224)
        y: Numpy array of raw GPS coordinates [lat, lon] in degrees, shape (N, 2)
    """
    # Read CSV file
    df = pd.read_csv(csv_path)

    # Find the correct column names
    image_col = find_column(df, ['image_path', 'filepath', 'image', 'path', 'file_name'])
    lat_col = find_column(df, ['Latitude', 'latitude', 'lat'])
    lon_col = find_column(df, ['Longitude', 'longitude', 'lon'])

    # Get the directory containing the CSV to resolve relative image paths
    csv_dir = os.path.dirname(csv_path)

    # Process images
    images = []
    valid_indices = []

    for idx, row in df.iterrows():
        image_path = row[image_col]

        # Handle relative paths
        if not os.path.isabs(image_path):
            image_path = os.path.join(csv_dir, image_path)

        try:
            img_tensor = load_and_preprocess_image(image_path, INFERENCE_TRANSFORM)
            images.append(img_tensor)
            valid_indices.append(idx)
        except Exception as e:
            print(f"Warning: Could not load image {image_path}: {e}")
            continue

    if len(images) == 0:
        raise ValueError("No valid images found in the dataset")

    # Stack images into a single tensor
    X = torch.stack(images)

    # Extract raw GPS coordinates (not normalized - backend expects raw degrees)
    latitudes = df.iloc[valid_indices][lat_col].values
    longitudes = df.iloc[valid_indices][lon_col].values
    y = np.column_stack([latitudes, longitudes]).astype(np.float32)

    return X, y
