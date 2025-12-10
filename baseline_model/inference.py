"""
Image2GPS: Inference script for GPS coordinate prediction from images.

This script loads a trained model and makes predictions on new images.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import argparse
import os


def load_model(checkpoint_path, device):
    """Load a trained model from checkpoint."""
    # Create model architecture
    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)

    # Load checkpoint (weights_only=False needed for numpy arrays in norm_params)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    norm_params = checkpoint['norm_params']

    print(f"Model loaded from: {checkpoint_path}")
    print(f"Trained for {checkpoint['epoch']} epochs, Val RMSE: {checkpoint['val_rmse']:.2f}m")

    return model, norm_params


def predict_single_image(model, image_path, norm_params, device):
    """Make prediction for a single image."""
    # Define transform (same as validation transform)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load and preprocess image
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    image_tensor = transform(image).unsqueeze(0).to(device)

    # Make prediction
    with torch.no_grad():
        output = model(image_tensor)
        pred_normalized = output.cpu().numpy()[0]

    # Denormalize prediction
    lat_pred = pred_normalized[0] * norm_params['lat_std'] + norm_params['lat_mean']
    lon_pred = pred_normalized[1] * norm_params['lon_std'] + norm_params['lon_mean']

    return lat_pred, lon_pred


def predict_batch(model, image_paths, norm_params, device, batch_size=32):
    """Make predictions for a batch of images."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    predictions = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_tensors = []

        for path in batch_paths:
            image = Image.open(path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            batch_tensors.append(transform(image))

        batch = torch.stack(batch_tensors).to(device)

        with torch.no_grad():
            outputs = model(batch)
            preds_normalized = outputs.cpu().numpy()

        # Denormalize predictions
        for pred in preds_normalized:
            lat = pred[0] * norm_params['lat_std'] + norm_params['lat_mean']
            lon = pred[1] * norm_params['lon_std'] + norm_params['lon_mean']
            predictions.append((lat, lon))

    return predictions


def main():
    parser = argparse.ArgumentParser(description='Image2GPS Inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--image', type=str, default=None,
                        help='Path to single image for prediction')
    parser.add_argument('--image_dir', type=str, default=None,
                        help='Directory containing images for batch prediction')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for predictions (CSV format)')

    args = parser.parse_args()

    if args.image is None and args.image_dir is None:
        print("Error: Please provide either --image or --image_dir")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model, norm_params = load_model(args.checkpoint, device)

    if args.image:
        # Single image prediction
        lat, lon = predict_single_image(model, args.image, norm_params, device)
        print(f"\nPrediction for {args.image}:")
        print(f"  Latitude: {lat:.6f}")
        print(f"  Longitude: {lon:.6f}")
        print(f"  Google Maps: https://www.google.com/maps?q={lat},{lon}")

    if args.image_dir:
        # Batch prediction
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        image_paths = []

        for filename in os.listdir(args.image_dir):
            ext = os.path.splitext(filename)[1].lower()
            if ext in image_extensions:
                image_paths.append(os.path.join(args.image_dir, filename))

        if not image_paths:
            print(f"No images found in {args.image_dir}")
            return

        print(f"\nProcessing {len(image_paths)} images...")
        predictions = predict_batch(model, image_paths, norm_params, device)

        # Output results
        if args.output:
            with open(args.output, 'w') as f:
                f.write("filename,latitude,longitude\n")
                for path, (lat, lon) in zip(image_paths, predictions):
                    filename = os.path.basename(path)
                    f.write(f"{filename},{lat:.6f},{lon:.6f}\n")
            print(f"Predictions saved to: {args.output}")
        else:
            print("\nPredictions:")
            for path, (lat, lon) in zip(image_paths, predictions):
                filename = os.path.basename(path)
                print(f"  {filename}: ({lat:.6f}, {lon:.6f})")


if __name__ == '__main__':
    main()
