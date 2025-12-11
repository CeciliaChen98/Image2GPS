"""
Quick script to evaluate saved ablation study models on test set.
"""

import os
import sys
import torch
import numpy as np
from geopy.distance import geodesic

# Add improved_model to path
sys.path.insert(0, 'improved_model')

from datasets import load_dataset
from improved_model.preprocess import get_val_transform, GPSImageDataset
from improved_model.model import IMG2GPS

def calculate_geodesic_rmse(predictions, actuals, norm_params):
    """Calculate RMSE in meters using geodesic distance."""
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


def evaluate_model(checkpoint_path, device):
    """Evaluate a single model checkpoint on test set."""
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    norm_params = checkpoint['norm_params']

    # Create model
    model = IMG2GPS(backbone='resnet50', hidden_dim=256, dropout=0.5)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Load test data
    dataset_test = load_dataset('rantyw/image2gps', split='test')
    val_transform = get_val_transform()

    test_dataset = GPSImageDataset(
        data_source=dataset_test,
        transform=val_transform,
        lat_mean=norm_params['lat_mean'],
        lat_std=norm_params['lat_std'],
        lon_mean=norm_params['lon_mean'],
        lon_std=norm_params['lon_std'],
        is_huggingface=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=0
    )

    # Evaluate
    all_preds = []
    all_actuals = []

    with torch.no_grad():
        for images, gps_coords in test_loader:
            images = images.to(device)
            outputs = model(images)
            all_preds.append(outputs.cpu().numpy())
            all_actuals.append(gps_coords.numpy())

    all_preds = np.concatenate(all_preds)
    all_actuals = np.concatenate(all_actuals)

    test_rmse, distances = calculate_geodesic_rmse(all_preds, all_actuals, norm_params)

    return test_rmse, np.mean(distances), np.median(distances)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model checkpoints and their configurations
    checkpoints = [
        ("resnet50_20251210_235157", "base_only", "gray=No, blur=No, erase=No"),
        ("resnet50_20251210_235801", "base+grayscale", "gray=Yes, blur=No, erase=No"),
        ("resnet50_20251211_000401", "base+blur", "gray=No, blur=Yes, erase=No"),
        ("resnet50_20251211_001020", "base+erasing", "gray=No, blur=No, erase=Yes"),
        ("resnet50_20251211_001620", "base+gray+blur", "gray=Yes, blur=Yes, erase=No"),
        ("resnet50_20251211_002222", "full_augmentation", "gray=Yes, blur=Yes, erase=Yes"),
    ]

    print("\n" + "="*70)
    print("ABLATION STUDY - TEST SET EVALUATION")
    print("="*70)

    results = []

    for checkpoint_name, config_name, aug_config in checkpoints:
        checkpoint_path = f"improved_model/checkpoints/{checkpoint_name}/best_model.pth"

        if not os.path.exists(checkpoint_path):
            print(f"\n[SKIP] {config_name}: Checkpoint not found")
            continue

        print(f"\nEvaluating: {config_name} ({aug_config})")

        try:
            test_rmse, test_mae, test_median = evaluate_model(checkpoint_path, device)
            results.append({
                "name": config_name,
                "config": aug_config,
                "test_rmse": test_rmse,
                "test_mae": test_mae,
                "test_median": test_median
            })
            print(f"  Test RMSE: {test_rmse:.2f}m | MAE: {test_mae:.2f}m | Median: {test_median:.2f}m")
        except Exception as e:
            print(f"  [ERROR] {e}")

    # Print summary table
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(f"{'Configuration':<20} {'Test RMSE':>12} {'Test MAE':>12} {'Median':>12}")
    print("-"*70)

    base_rmse = None
    for r in results:
        if r["name"] == "base_only":
            base_rmse = r["test_rmse"]

        diff = ""
        if base_rmse and r["name"] != "base_only":
            pct = ((r["test_rmse"] - base_rmse) / base_rmse) * 100
            diff = f" ({pct:+.1f}%)"

        print(f"{r['name']:<20} {r['test_rmse']:>10.2f}m{diff:>12} {r['test_mae']:>10.2f}m {r['test_median']:>10.2f}m")

    print("="*70)


if __name__ == "__main__":
    main()
