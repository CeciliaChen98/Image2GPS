"""
Compare trained IMG2GPS model with Picarta API (10 samples, with region hint only).
"""

import os
import sys
import json
import tempfile
import time
import numpy as np
from PIL import Image
from geopy.distance import geodesic
from datasets import load_dataset
import torch

# Add final_model to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'final_model'))
from model import IMG2GPS
from preprocess import get_val_transform

# Picarta API
from picarta import Picarta


def calculate_metrics(predictions, actuals):
    """Calculate distance-based metrics."""
    distances = []
    for pred, actual in zip(predictions, actuals):
        if pred[0] is not None and pred[1] is not None:
            dist = geodesic((actual[0], actual[1]), (pred[0], pred[1])).meters
            distances.append(dist)
        else:
            distances.append(None)

    valid_distances = [d for d in distances if d is not None]

    if not valid_distances:
        return None

    valid_distances = np.array(valid_distances)

    return {
        'rmse': float(np.sqrt(np.mean(valid_distances ** 2))),
        'mae': float(np.mean(valid_distances)),
        'median': float(np.median(valid_distances)),
        'min': float(np.min(valid_distances)),
        'max': float(np.max(valid_distances)),
        'valid_count': len(valid_distances),
        'total_count': len(distances),
        'distances': valid_distances.tolist()
    }


def load_trained_model(checkpoint_path, device):
    """Load the trained IMG2GPS model."""
    model = IMG2GPS(pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Compare IMG2GPS model with Picarta API')
    parser.add_argument('--api-token', type=str, required=True, help='Picarta API token')
    args = parser.parse_args()

    # Configuration
    CHECKPOINT_PATH = "final_model/checkpoints/resnet50_20251211_144503/best_model.pth"
    DATASET_NAME = "rantyw/image2gps"
    NUM_SAMPLES = 10  # Limited API calls

    api_token = args.api_token
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load test dataset
    print(f"\nLoading test dataset: {DATASET_NAME}")
    dataset_test = load_dataset(DATASET_NAME, split="test")
    print(f"Total test samples: {len(dataset_test)}")

    # Sample subset for comparison (fixed seed for reproducibility)
    indices = np.random.RandomState(42).choice(len(dataset_test), NUM_SAMPLES, replace=False)

    test_samples = []
    for idx in indices:
        sample = dataset_test[int(idx)]
        image = sample['image']
        if image.mode != 'RGB':
            image = image.convert('RGB')
        lat = sample['Latitude']
        lon = sample['Longitude']
        test_samples.append((image, lat, lon))

    actual_coords = [(lat, lon) for _, lat, lon in test_samples]
    print(f"Selected {len(test_samples)} samples for comparison")

    # Load trained model
    print(f"\nLoading trained model from: {CHECKPOINT_PATH}")
    model = load_trained_model(CHECKPOINT_PATH, device)
    transform = get_val_transform()

    # Evaluate trained model
    print("\n" + "="*60)
    print("Evaluating TRAINED MODEL (IMG2GPS ResNet-50)")
    print("="*60)

    trained_predictions = []
    trained_distances = []

    for i, (image, actual_lat, actual_lon) in enumerate(test_samples):
        img_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model.predict(img_tensor)
            pred_lat = pred[0, 0].item()
            pred_lon = pred[0, 1].item()

        trained_predictions.append((pred_lat, pred_lon))
        dist = geodesic((actual_lat, actual_lon), (pred_lat, pred_lon)).meters
        trained_distances.append(dist)
        print(f"  Sample {i+1}: Pred ({pred_lat:.6f}, {pred_lon:.6f}), Actual ({actual_lat:.6f}, {actual_lon:.6f}), Error: {dist:.2f}m")

    trained_metrics = calculate_metrics(trained_predictions, actual_coords)

    print(f"\nTrained Model Summary:")
    print(f"  RMSE:   {trained_metrics['rmse']:.2f} meters")
    print(f"  MAE:    {trained_metrics['mae']:.2f} meters")
    print(f"  Median: {trained_metrics['median']:.2f} meters")

    # Evaluate Picarta API (with region hint)
    print("\n" + "="*60)
    print("Evaluating PICARTA API (With Philadelphia Region Hint)")
    print("="*60)

    localizer = Picarta(api_token)
    picarta_predictions = []
    picarta_distances = []

    for i, (image, actual_lat, actual_lon) in enumerate(test_samples):
        # Save image to temporary file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            temp_path = f.name
            image.save(temp_path, 'JPEG')

        try:
            result = localizer.localize(
                img_path=temp_path,
                top_k=1,
                country_code="US",
                center_latitude=39.95,
                center_longitude=-75.19,
                radius=10  # 10km radius around Philadelphia
            )

            # Debug first result
            if i == 0:
                print(f"  API Response type: {type(result)}")
                print(f"  API Response: {result}")
                if isinstance(result, dict):
                    print(f"  Keys: {result.keys()}")
                    print(f"  ai_lat in result: {'ai_lat' in result}")

            # Extract prediction - Picarta returns JSON string, need to parse it
            pred_lat, pred_lon = None, None

            # Handle string response (JSON)
            if isinstance(result, str):
                try:
                    result = json.loads(result)
                except:
                    pass

            if isinstance(result, dict):
                if 'ai_lat' in result:
                    pred_lat = float(result['ai_lat'])
                    pred_lon = float(result['ai_lon'])
                elif 'lat' in result:
                    pred_lat = float(result['lat'])
                    pred_lon = float(result['lon'])

            if pred_lat is not None:
                picarta_predictions.append((pred_lat, pred_lon))
                dist = geodesic((actual_lat, actual_lon), (pred_lat, pred_lon)).meters
                picarta_distances.append(dist)
                print(f"  Sample {i+1}: Pred ({pred_lat:.6f}, {pred_lon:.6f}), Actual ({actual_lat:.6f}, {actual_lon:.6f}), Error: {dist:.2f}m")
            else:
                picarta_predictions.append((None, None))
                print(f"  Sample {i+1}: No valid prediction from API")

            time.sleep(0.5)

        except Exception as e:
            print(f"  Sample {i+1}: Error - {e}")
            picarta_predictions.append((None, None))
        finally:
            os.unlink(temp_path)

    picarta_metrics = calculate_metrics(picarta_predictions, actual_coords)

    if picarta_metrics:
        print(f"\nPicarta API Summary:")
        print(f"  RMSE:   {picarta_metrics['rmse']:.2f} meters")
        print(f"  MAE:    {picarta_metrics['mae']:.2f} meters")
        print(f"  Median: {picarta_metrics['median']:.2f} meters")
        print(f"  Valid:  {picarta_metrics['valid_count']}/{picarta_metrics['total_count']} samples")

    # Summary comparison
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"\n{'Metric':<12} {'Trained Model':>15} {'Picarta API':>15}")
    print("-" * 45)
    print(f"{'RMSE (m)':<12} {trained_metrics['rmse']:>15.2f} {picarta_metrics['rmse'] if picarta_metrics else 'N/A':>15}")
    print(f"{'MAE (m)':<12} {trained_metrics['mae']:>15.2f} {picarta_metrics['mae'] if picarta_metrics else 'N/A':>15}")
    print(f"{'Median (m)':<12} {trained_metrics['median']:>15.2f} {picarta_metrics['median'] if picarta_metrics else 'N/A':>15}")

    if picarta_metrics:
        print("\n" + "="*60)
        print("ANALYSIS")
        print("="*60)
        if trained_metrics['rmse'] < picarta_metrics['rmse']:
            improvement = ((picarta_metrics['rmse'] - trained_metrics['rmse']) / picarta_metrics['rmse']) * 100
            print(f"\nTrained model outperforms Picarta by {improvement:.1f}%")
            print(f"  Trained RMSE: {trained_metrics['rmse']:.2f}m vs Picarta: {picarta_metrics['rmse']:.2f}m")
        else:
            improvement = ((trained_metrics['rmse'] - picarta_metrics['rmse']) / trained_metrics['rmse']) * 100
            print(f"\nPicarta outperforms trained model by {improvement:.1f}%")
            print(f"  Picarta RMSE: {picarta_metrics['rmse']:.2f}m vs Trained: {trained_metrics['rmse']:.2f}m")

    # Save results
    results = {
        'num_samples': NUM_SAMPLES,
        'trained_model': {
            'metrics': trained_metrics,
            'predictions': trained_predictions,
            'distances': trained_distances
        },
        'picarta_api': {
            'metrics': picarta_metrics,
            'predictions': [(p[0], p[1]) if p[0] else None for p in picarta_predictions],
            'distances': picarta_distances
        },
        'actual_coords': actual_coords
    }

    with open('comparison_results_10.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: comparison_results_10.json")


if __name__ == "__main__":
    main()
