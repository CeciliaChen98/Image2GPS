"""
Image2GPS: Preprocessing module for GPS coordinate prediction.

This module provides the prepare_data function required by the evaluation backend.
It reads a CSV file containing image paths and GPS coordinates, and returns
preprocessed data suitable for model inference.
"""

import os
import pandas as pd
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image


# Image preprocessing transform (same as validation/inference transform)
INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


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

    This function is called by the evaluation backend with the path to a CSV file.
    The CSV contains image paths and GPS coordinates.

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
