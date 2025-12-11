"""
Training script with milestone checkpoints.
Runs both with and without grayscale in a single call.
Saves checkpoints at epochs 10, 20, 30, 40, 50 and evaluates each on test set.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import os
import json
import sys
from datasets import load_dataset
from geopy.distance import geodesic

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from improved_model.model import IMG2GPS
from improved_model.preprocess import GPSImageDataset, get_train_transform, get_val_transform

MILESTONES = [10, 20, 30, 40, 50]
NUM_EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_WORKERS = 4
OUTPUT_DIR = './performance_test/milestone_comparison'
DATASET = 'rantyw/image2gps'


def calculate_metrics(predictions, actuals, norm_params):
    """Calculate evaluation metrics."""
    preds_denorm = predictions * np.array([norm_params['lat_std'], norm_params['lon_std']]) + \
                   np.array([norm_params['lat_mean'], norm_params['lon_mean']])
    actuals_denorm = actuals * np.array([norm_params['lat_std'], norm_params['lon_std']]) + \
                     np.array([norm_params['lat_mean'], norm_params['lon_mean']])

    distances = []
    for pred, actual in zip(preds_denorm, actuals_denorm):
        dist = geodesic((actual[0], actual[1]), (pred[0], pred[1])).meters
        distances.append(dist)

    distances = np.array(distances)

    return {
        'rmse': np.sqrt(np.mean(distances ** 2)),
        'mae': np.mean(distances),
        'median': np.median(distances),
    }


def create_dataloaders(use_grayscale=False):
    """Create dataloaders."""
    print(f"\nLoading dataset: {DATASET}")
    print(f"Grayscale augmentation: {use_grayscale}")

    dataset_train = load_dataset(DATASET, split="train")
    dataset_val = load_dataset(DATASET, split="validation")
    dataset_test = load_dataset(DATASET, split="test")

    print(f"Train: {len(dataset_train)} | Val: {len(dataset_val)} | Test: {len(dataset_test)}")

    train_dataset = GPSImageDataset(
        dataset_train,
        transform=get_train_transform(use_grayscale=use_grayscale, use_blur=False, use_erasing=False),
        is_huggingface=True
    )

    norm_params = train_dataset.get_normalization_params()

    val_dataset = GPSImageDataset(
        dataset_val, transform=get_val_transform(),
        lat_mean=norm_params['lat_mean'], lat_std=norm_params['lat_std'],
        lon_mean=norm_params['lon_mean'], lon_std=norm_params['lon_std'],
        is_huggingface=True
    )

    test_dataset = GPSImageDataset(
        dataset_test, transform=get_val_transform(),
        lat_mean=norm_params['lat_mean'], lat_std=norm_params['lat_std'],
        lon_mean=norm_params['lon_mean'], lon_std=norm_params['lon_std'],
        is_huggingface=True
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)

    return train_loader, val_loader, test_loader, norm_params


def get_lr_scheduler(optimizer, num_epochs, warmup_epochs=5, min_lr=1e-6):
    """LR scheduler with warmup and cosine annealing."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
            return max(min_lr, 0.5 * (1 + np.cos(np.pi * progress)))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler):
    """Train for one epoch with AMP."""
    model.train()
    running_loss = 0.0
    num_batches = 0

    for images, gps_coords in dataloader:
        images = images.to(device)
        gps_coords = gps_coords.to(device)

        optimizer.zero_grad()

        with autocast():
            outputs = model(images)
            loss = criterion(outputs, gps_coords)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

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

    return calculate_metrics(all_preds, all_actuals, norm_params)


def train_single_config(use_grayscale, device):
    """Train a single configuration and return milestone results."""
    grayscale_str = "with_grayscale" if use_grayscale else "no_grayscale"
    output_dir = os.path.join(OUTPUT_DIR, f"resnet50_{grayscale_str}")
    os.makedirs(output_dir, exist_ok=True)

    # Create dataloaders
    train_loader, val_loader, test_loader, norm_params = create_dataloaders(use_grayscale=use_grayscale)

    # Create model
    model = IMG2GPS(
        backbone='resnet50', pretrained=True,
        hidden_dim=256, dropout=0.5, norm_params=norm_params
    )
    model = model.to(device)
    print(f"Model parameters: {model.get_num_parameters():,}")

    # Optimizer and scheduler
    criterion = nn.MSELoss()
    optimizer = optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': LEARNING_RATE * 0.1},
        {'params': model.regression_head.parameters(), 'lr': LEARNING_RATE}
    ], weight_decay=1e-4)
    scheduler = get_lr_scheduler(optimizer, NUM_EPOCHS, warmup_epochs=5)
    scaler = GradScaler()

    # Training loop
    milestone_checkpoints = {}
    best_rmse_so_far = float('inf')

    print("\n" + "=" * 70)
    print(f"Training ResNet-50 {'WITH' if use_grayscale else 'WITHOUT'} grayscale")
    print(f"Milestones: {MILESTONES}")
    print("=" * 70)

    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_metrics = validate(model, val_loader, norm_params, device)
        scheduler.step()

        if val_metrics['rmse'] < best_rmse_so_far:
            best_rmse_so_far = val_metrics['rmse']

        print(f"Epoch [{epoch + 1:2d}/{NUM_EPOCHS}] "
              f"Loss: {train_loss:.4f} | Val RMSE: {val_metrics['rmse']:.2f}m | "
              f"Best: {best_rmse_so_far:.2f}m")

        # Save milestone checkpoint
        if (epoch + 1) in MILESTONES:
            checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch{epoch + 1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'val_rmse': val_metrics['rmse'],
                'best_val_rmse': best_rmse_so_far,
            }, checkpoint_path)
            milestone_checkpoints[epoch + 1] = checkpoint_path
            print(f"  >> Saved milestone at epoch {epoch + 1}")

    # Evaluate milestones on test set
    print("\n" + "-" * 50)
    print(f"Evaluating milestones for {grayscale_str}...")
    print("-" * 50)

    milestone_results = []
    for milestone_epoch in MILESTONES:
        checkpoint_path = milestone_checkpoints.get(milestone_epoch)
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            test_metrics = validate(model, test_loader, norm_params, device)

            result = {
                'epoch': milestone_epoch,
                'best_val_rmse': checkpoint['best_val_rmse'],
                'test_rmse': test_metrics['rmse'],
                'test_mae': test_metrics['mae'],
                'test_median': test_metrics['median'],
            }
            milestone_results.append(result)
            print(f"Epoch {milestone_epoch:2d}: Best Val: {result['best_val_rmse']:.2f}m | "
                  f"Test RMSE: {result['test_rmse']:.2f}m | MAE: {result['test_mae']:.2f}m")

    # Save results
    with open(os.path.join(output_dir, 'milestone_results.json'), 'w') as f:
        json.dump(milestone_results, f, indent=2)

    return milestone_results


def write_combined_results(results_no_gray, results_with_gray):
    """Write combined results to RESULTS.txt."""
    output_file = os.path.join(OUTPUT_DIR, "RESULTS.txt")

    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("MILESTONE COMPARISON: ResNet-50 with/without Grayscale (50 epochs)\n")
        f.write("=" * 80 + "\n\n")

        f.write("Results at each milestone (best model up to that epoch evaluated on test set)\n\n")

        f.write("-" * 80 + "\n")
        f.write(f"{'Epoch':<10} {'Config':<20} {'Best Val RMSE':<15} {'Test RMSE':<15} {'Test MAE':<15}\n")
        f.write("-" * 80 + "\n")

        for epoch in MILESTONES:
            # No grayscale
            r_no = next((r for r in results_no_gray if r['epoch'] == epoch), None)
            if r_no:
                f.write(f"{epoch:<10} {'no_grayscale':<20} {r_no['best_val_rmse']:.2f}m{'':<8} "
                        f"{r_no['test_rmse']:.2f}m{'':<8} {r_no['test_mae']:.2f}m\n")

            # With grayscale
            r_with = next((r for r in results_with_gray if r['epoch'] == epoch), None)
            if r_with:
                f.write(f"{epoch:<10} {'with_grayscale':<20} {r_with['best_val_rmse']:.2f}m{'':<8} "
                        f"{r_with['test_rmse']:.2f}m{'':<8} {r_with['test_mae']:.2f}m\n")

            f.write("\n")

        f.write("=" * 80 + "\n")

        # Summary
        f.write("\nSUMMARY (Test RMSE at each milestone):\n\n")
        f.write(f"{'Epoch':<10} {'No Grayscale':<20} {'With Grayscale':<20} {'Difference':<15}\n")
        f.write("-" * 65 + "\n")

        for epoch in MILESTONES:
            r_no = next((r for r in results_no_gray if r['epoch'] == epoch), None)
            r_with = next((r for r in results_with_gray if r['epoch'] == epoch), None)
            if r_no and r_with:
                diff = r_with['test_rmse'] - r_no['test_rmse']
                diff_str = f"{diff:+.2f}m"
                f.write(f"{epoch:<10} {r_no['test_rmse']:.2f}m{'':<13} "
                        f"{r_with['test_rmse']:.2f}m{'':<13} {diff_str}\n")

        f.write("=" * 80 + "\n")

    print(f"\nResults written to: {output_file}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n" + "=" * 80)
    print("RUNNING MILESTONE COMPARISON EXPERIMENT")
    print(f"Configurations: no_grayscale, with_grayscale")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Milestones: {MILESTONES}")
    print("=" * 80)

    # Train without grayscale
    print("\n\n>>> CONFIGURATION 1: WITHOUT GRAYSCALE <<<")
    results_no_gray = train_single_config(use_grayscale=False, device=device)

    # Train with grayscale
    print("\n\n>>> CONFIGURATION 2: WITH GRAYSCALE <<<")
    results_with_gray = train_single_config(use_grayscale=True, device=device)

    # Write combined results
    write_combined_results(results_no_gray, results_with_gray)

    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETE!")
    print("=" * 80)


if __name__ == '__main__':
    main()
