"""
Image2GPS: Training script for the improved model.

Improvements over baseline training:
1. Support for multiple backbone architectures
2. Learning rate warmup
3. Cosine annealing learning rate schedule
4. Early stopping
5. Gradient clipping
6. Mixed precision training (AMP)
7. Better logging and checkpointing
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import argparse
import os
import json
from datetime import datetime
from datasets import load_dataset
from geopy.distance import geodesic

from model import IMG2GPS
from preprocess import GPSImageDataset, get_train_transform, get_val_transform


def calculate_haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the Haversine distance between two GPS coordinates.

    Args:
        lat1, lon1: First coordinate (degrees)
        lat2, lon2: Second coordinate (degrees)

    Returns:
        Distance in meters
    """
    R = 6371000  # Earth's radius in meters

    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = np.sin(delta_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c


def calculate_metrics(predictions, actuals, norm_params):
    """
    Calculate evaluation metrics.

    Args:
        predictions: Normalized predictions (N, 2)
        actuals: Normalized actual coordinates (N, 2)
        norm_params: Normalization parameters

    Returns:
        Dictionary of metrics
    """
    # Denormalize
    preds_denorm = predictions * np.array([norm_params['lat_std'], norm_params['lon_std']]) + \
                   np.array([norm_params['lat_mean'], norm_params['lon_mean']])
    actuals_denorm = actuals * np.array([norm_params['lat_std'], norm_params['lon_std']]) + \
                     np.array([norm_params['lat_mean'], norm_params['lon_mean']])

    # Calculate distances
    distances = []
    for pred, actual in zip(preds_denorm, actuals_denorm):
        # Using geopy for accurate geodesic distance
        dist = geodesic((actual[0], actual[1]), (pred[0], pred[1])).meters
        distances.append(dist)

    distances = np.array(distances)

    return {
        'rmse': np.sqrt(np.mean(distances ** 2)),
        'mae': np.mean(distances),
        'median': np.median(distances),
        'max': np.max(distances),
        'min': np.min(distances),
        'std': np.std(distances)
    }


def create_dataloaders(dataset_name, batch_size=32, val_split=0.2, num_workers=4):
    """Create training and validation dataloaders."""
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split="train")
    print(f"Dataset loaded with {len(dataset)} samples")

    # Split dataset
    total_size = len(dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size

    indices = list(range(total_size))
    np.random.seed(42)
    np.random.shuffle(indices)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_subset = dataset.select(train_indices)
    val_subset = dataset.select(val_indices)

    print(f"Training samples: {len(train_subset)}, Validation samples: {len(val_subset)}")

    # Create datasets
    train_dataset = GPSImageDataset(
        train_subset,
        transform=get_train_transform(),
        is_huggingface=True
    )

    norm_params = train_dataset.get_normalization_params()
    print(f"Normalization params: {norm_params}")

    val_dataset = GPSImageDataset(
        val_subset,
        transform=get_val_transform(),
        lat_mean=norm_params['lat_mean'],
        lat_std=norm_params['lat_std'],
        lon_mean=norm_params['lon_mean'],
        lon_std=norm_params['lon_std'],
        is_huggingface=True
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, norm_params


def get_lr_scheduler(optimizer, num_epochs, warmup_epochs=5, min_lr=1e-6):
    """
    Create learning rate scheduler with warmup and cosine annealing.
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup
            return (epoch + 1) / warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
            return max(min_lr, 0.5 * (1 + np.cos(np.pi * progress)))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler, use_amp=True):
    """Train for one epoch with optional mixed precision."""
    model.train()
    running_loss = 0.0
    num_batches = 0

    for images, gps_coords in dataloader:
        images = images.to(device)
        gps_coords = gps_coords.to(device)

        optimizer.zero_grad()

        if use_amp and device.type == 'cuda':
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, gps_coords)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, gps_coords)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    return running_loss / num_batches


def validate(model, dataloader, norm_params, device):
    """Validate the model."""
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

    metrics = calculate_metrics(all_preds, all_actuals, norm_params)

    # Calculate baseline (predicting mean)
    baseline_preds = np.zeros_like(all_actuals)
    baseline_metrics = calculate_metrics(baseline_preds, all_actuals, norm_params)

    return metrics, baseline_metrics


def train(args):
    """Main training function."""
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"{args.backbone}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Save args
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Create dataloaders
    train_loader, val_loader, norm_params = create_dataloaders(
        args.dataset,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers
    )

    # Save normalization parameters
    with open(os.path.join(output_dir, 'norm_params.json'), 'w') as f:
        json.dump(norm_params, f, indent=2)

    # Create model
    model = IMG2GPS(
        backbone=args.backbone,
        pretrained=args.pretrained,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        norm_params=norm_params
    )
    model = model.to(device)
    print(f"Model: {args.backbone} with {model.get_num_parameters():,} trainable parameters")

    # Loss and optimizer
    criterion = nn.MSELoss()

    # Different learning rates for backbone and head
    backbone_params = list(model.backbone.parameters())
    head_params = list(model.regression_head.parameters())

    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': args.learning_rate * 0.1},  # Lower LR for pretrained backbone
        {'params': head_params, 'lr': args.learning_rate}
    ], weight_decay=args.weight_decay)

    scheduler = get_lr_scheduler(optimizer, args.num_epochs, args.warmup_epochs)

    # Mixed precision scaler
    scaler = GradScaler()

    # Training loop
    best_rmse = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_rmse': [], 'val_mae': [], 'lr': []}

    print("\n" + "=" * 70)
    print("Starting training...")
    print("=" * 70)

    for epoch in range(args.num_epochs):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, args.use_amp
        )

        # Validate
        val_metrics, baseline_metrics = validate(model, val_loader, norm_params, device)

        # Step scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Log
        history['train_loss'].append(train_loss)
        history['val_rmse'].append(val_metrics['rmse'])
        history['val_mae'].append(val_metrics['mae'])
        history['lr'].append(current_lr)

        print(f"\nEpoch [{epoch + 1}/{args.num_epochs}]")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val RMSE: {val_metrics['rmse']:.2f}m | Baseline: {baseline_metrics['rmse']:.2f}m")
        print(f"  Val MAE: {val_metrics['mae']:.2f}m | Median: {val_metrics['median']:.2f}m")
        print(f"  LR: {current_lr:.6f}")

        # Save best model
        if val_metrics['rmse'] < best_rmse:
            best_rmse = val_metrics['rmse']
            patience_counter = 0

            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_rmse': val_metrics['rmse'],
                'norm_params': norm_params,
                'backbone': args.backbone,
                'hidden_dim': args.hidden_dim,
                'dropout': args.dropout
            }
            torch.save(checkpoint, os.path.join(output_dir, 'best_model.pth'))

            # Save state dict only for submission
            torch.save(model.state_dict(), os.path.join(output_dir, 'model.pt'))

            print(f"  -> New best model! RMSE: {val_metrics['rmse']:.2f}m")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break

    # Save training history
    with open(os.path.join(output_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 70)
    print("Training complete!")
    print(f"Best Validation RMSE: {best_rmse:.2f}m")
    print(f"Model saved to: {output_dir}")
    print("=" * 70)

    # Print instructions for submission
    print("\n" + "=" * 70)
    print("SUBMISSION INSTRUCTIONS:")
    print("=" * 70)
    print(f"1. Copy model.pt from {output_dir}/model.pt")
    print(f"2. Update NORM_PARAMS in model.py with values from {output_dir}/norm_params.json:")
    print(f"   {norm_params}")
    print("3. Submit: preprocess.py, model.py, model.pt")
    print("=" * 70)

    return model, norm_params, history


def main():
    parser = argparse.ArgumentParser(description='Train Improved Image2GPS Model')

    # Dataset
    parser.add_argument('--dataset', type=str, default='CoconutYezi/released_img',
                        help='HuggingFace dataset name')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation split ratio')

    # Model
    parser.add_argument('--backbone', type=str, default='resnet50',
                        choices=['resnet18', 'resnet50', 'resnet101', 'efficientnet_b0'],
                        help='Backbone architecture')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained weights')
    parser.add_argument('--no_pretrained', action='store_false', dest='pretrained')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Hidden dimension for regression head')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate')

    # Training
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--use_amp', action='store_true', default=True,
                        help='Use automatic mixed precision')
    parser.add_argument('--no_amp', action='store_false', dest='use_amp')

    # System
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--output_dir', type=str, default='./checkpoints')

    args = parser.parse_args()

    print("=" * 70)
    print("Image2GPS Improved Model Training")
    print("=" * 70)
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print("=" * 70 + "\n")

    train(args)


if __name__ == '__main__':
    main()
