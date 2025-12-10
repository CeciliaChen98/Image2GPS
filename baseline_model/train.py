"""
Image2GPS: Training script for GPS coordinate prediction from images.
Based on the baseline model from the course project.

This script trains a ResNet-18 model to predict GPS coordinates (latitude, longitude)
from input images using the CoconutYezi/released_img dataset.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim.lr_scheduler import StepLR
from datasets import load_dataset
from geopy.distance import geodesic
from PIL import Image
import numpy as np
import argparse
import os


class GPSImageDataset(Dataset):
    """
    Custom Dataset for GPS coordinate prediction from images.

    Normalizes GPS coordinates using mean and standard deviation for stable training.
    """

    def __init__(self, hf_dataset, transform=None, lat_mean=None, lat_std=None,
                 lon_mean=None, lon_std=None):
        self.hf_dataset = hf_dataset
        self.transform = transform

        # Compute mean and std from the dataset if not provided
        self.latitude_mean = lat_mean if lat_mean is not None else np.mean(np.array(self.hf_dataset['Latitude']))
        self.latitude_std = lat_std if lat_std is not None else np.std(np.array(self.hf_dataset['Latitude']))
        self.longitude_mean = lon_mean if lon_mean is not None else np.mean(np.array(self.hf_dataset['Longitude']))
        self.longitude_std = lon_std if lon_std is not None else np.std(np.array(self.hf_dataset['Longitude']))

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        example = self.hf_dataset[idx]

        # Load and process the image
        image = example['image']
        latitude = example['Latitude']
        longitude = example['Longitude']

        # Convert to RGB if necessary (some images might be grayscale or RGBA)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Normalize GPS coordinates
        latitude_norm = (latitude - self.latitude_mean) / self.latitude_std
        longitude_norm = (longitude - self.longitude_mean) / self.longitude_std
        gps_coords = torch.tensor([latitude_norm, longitude_norm], dtype=torch.float32)

        return image, gps_coords

    def get_normalization_params(self):
        """Return the normalization parameters for later use (as Python floats for torch.save compatibility)."""
        return {
            'lat_mean': float(self.latitude_mean),
            'lat_std': float(self.latitude_std),
            'lon_mean': float(self.longitude_mean),
            'lon_std': float(self.longitude_std)
        }


def create_dataloaders(dataset_name, batch_size=32, val_split=0.2, num_workers=0):
    """
    Create training and validation dataloaders from a HuggingFace dataset.

    Since the dataset only has train split without test labels, we split
    the training data into train and validation sets.
    """
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split="train")
    print(f"Dataset loaded with {len(dataset)} samples")
    print(f"Features: {dataset.features}")

    # Define transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Split the dataset into train and validation
    total_size = len(dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size

    # Create indices for splitting
    indices = list(range(total_size))
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(indices)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # Create subset datasets
    train_subset = dataset.select(train_indices)
    val_subset = dataset.select(val_indices)

    print(f"Training samples: {len(train_subset)}, Validation samples: {len(val_subset)}")

    # Create the training dataset first to get normalization parameters
    train_dataset = GPSImageDataset(hf_dataset=train_subset, transform=train_transform)

    # Get normalization parameters from training set
    norm_params = train_dataset.get_normalization_params()
    print(f"Normalization params - Lat mean: {norm_params['lat_mean']:.6f}, "
          f"Lat std: {norm_params['lat_std']:.6f}")
    print(f"Normalization params - Lon mean: {norm_params['lon_mean']:.6f}, "
          f"Lon std: {norm_params['lon_std']:.6f}")

    # Create validation dataset using training normalization parameters
    val_dataset = GPSImageDataset(
        hf_dataset=val_subset,
        transform=val_transform,
        lat_mean=norm_params['lat_mean'],
        lat_std=norm_params['lat_std'],
        lon_mean=norm_params['lon_mean'],
        lon_std=norm_params['lon_std']
    )

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    return train_dataloader, val_dataloader, norm_params


def create_model(pretrained=True):
    """
    Create a ResNet-18 model modified for GPS coordinate regression.

    The final fully connected layer is replaced to output 2 values
    (latitude and longitude).
    """
    if pretrained:
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    else:
        resnet = models.resnet18(weights=None)

    # Modify the last fully connected layer to output 2 values
    num_features = resnet.fc.in_features
    resnet.fc = nn.Linear(num_features, 2)

    # Enable gradients for all parameters (full fine-tuning)
    for param in resnet.parameters():
        param.requires_grad = True

    return resnet


def calculate_geodesic_rmse(predictions, actuals, norm_params):
    """
    Calculate RMSE in meters using geodesic distance.

    Args:
        predictions: Normalized predictions (N, 2)
        actuals: Normalized actual coordinates (N, 2)
        norm_params: Dictionary with normalization parameters

    Returns:
        rmse: Root Mean Square Error in meters
        distances: List of individual distances in meters
    """
    # Denormalize predictions and actuals
    lat_mean, lat_std = norm_params['lat_mean'], norm_params['lat_std']
    lon_mean, lon_std = norm_params['lon_mean'], norm_params['lon_std']

    preds_denorm = predictions * np.array([lat_std, lon_std]) + np.array([lat_mean, lon_mean])
    actuals_denorm = actuals * np.array([lat_std, lon_std]) + np.array([lat_mean, lon_mean])

    distances = []
    for pred, actual in zip(preds_denorm, actuals_denorm):
        distance = geodesic((actual[0], actual[1]), (pred[0], pred[1])).meters
        distances.append(distance)

    rmse = np.sqrt(np.mean(np.array(distances) ** 2))
    return rmse, distances


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train the model for one epoch."""
    model.train()
    running_loss = 0.0

    for images, gps_coords in dataloader:
        images, gps_coords = images.to(device), gps_coords.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, gps_coords)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)


def validate(model, dataloader, norm_params, device):
    """
    Validate the model and calculate metrics.

    Returns:
        val_rmse: Validation RMSE in meters
        baseline_rmse: Baseline RMSE (predicting mean coordinates) in meters
    """
    model.eval()
    all_preds = []
    all_actuals = []

    with torch.no_grad():
        for images, gps_coords in dataloader:
            images = images.to(device)
            outputs = model(images)

            all_preds.append(outputs.cpu().numpy())
            all_actuals.append(gps_coords.numpy())

    all_preds = np.concatenate(all_preds)
    all_actuals = np.concatenate(all_actuals)

    # Calculate model RMSE
    val_rmse, _ = calculate_geodesic_rmse(all_preds, all_actuals, norm_params)

    # Calculate baseline RMSE (predicting mean, which is 0 in normalized space)
    baseline_preds = np.zeros_like(all_actuals)
    baseline_rmse, _ = calculate_geodesic_rmse(baseline_preds, all_actuals, norm_params)

    return val_rmse, baseline_rmse


def train(args):
    """Main training function."""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create dataloaders
    train_dataloader, val_dataloader, norm_params = create_dataloaders(
        args.dataset,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers
    )

    # Verify data loading
    for images, gps_coords in train_dataloader:
        print(f"Batch shape: images {images.size()}, gps_coords {gps_coords.size()}")
        break

    # Create model
    model = create_model(pretrained=args.pretrained)
    model = model.to(device)
    print(f"Model created: ResNet-18 with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = StepLR(optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma)

    # Training loop
    best_val_rmse = float('inf')

    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)

    for epoch in range(args.num_epochs):
        # Train
        train_loss = train_one_epoch(model, train_dataloader, criterion, optimizer, device)

        # Validate
        val_rmse, baseline_rmse = validate(model, val_dataloader, norm_params, device)

        # Step scheduler
        scheduler.step()

        # Print metrics
        print(f"Epoch [{epoch + 1}/{args.num_epochs}]")
        print(f"  Training Loss: {train_loss:.4f}")
        print(f"  Validation RMSE: {val_rmse:.2f}m, Baseline RMSE: {baseline_rmse:.2f}m")

        # Save best model
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            save_path = os.path.join(args.output_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_rmse': val_rmse,
                'norm_params': norm_params
            }, save_path)
            print(f"  -> New best model saved! RMSE: {val_rmse:.2f}m")

    print("\n" + "="*60)
    print("Training complete!")
    print(f"Best Validation RMSE: {best_val_rmse:.2f}m")
    print("="*60)

    # Save final model
    final_path = os.path.join(args.output_dir, 'final_model.pth')
    torch.save({
        'epoch': args.num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_rmse': val_rmse,
        'norm_params': norm_params
    }, final_path)
    print(f"Final model saved to: {final_path}")

    # Save normalization parameters separately for inference
    norm_path = os.path.join(args.output_dir, 'norm_params.pth')
    torch.save(norm_params, norm_path)
    print(f"Normalization parameters saved to: {norm_path}")

    return model, norm_params


def main():
    parser = argparse.ArgumentParser(description='Train Image2GPS model')

    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='CoconutYezi/released_img',
                        help='HuggingFace dataset name')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation split ratio (default: 0.2)')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training (default: 32)')
    parser.add_argument('--num_epochs', type=int, default=15,
                        help='Number of training epochs (default: 15)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--scheduler_step', type=int, default=5,
                        help='Step size for learning rate scheduler (default: 5)')
    parser.add_argument('--scheduler_gamma', type=float, default=0.1,
                        help='Gamma for learning rate scheduler (default: 0.1)')

    # Model arguments
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained ResNet weights (default: True)')
    parser.add_argument('--no_pretrained', action='store_false', dest='pretrained',
                        help='Do not use pretrained ResNet weights')

    # System arguments
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loading workers (default: 0)')
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                        help='Directory to save model checkpoints')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Print configuration
    print("="*60)
    print("Image2GPS Training Configuration")
    print("="*60)
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print("="*60 + "\n")

    # Train
    train(args)


if __name__ == '__main__':
    main()
