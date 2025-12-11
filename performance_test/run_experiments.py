"""
Image2GPS: Comprehensive experiment runner for comparing baseline and improved models.

This script:
1. Runs baseline model training (with and without pretrained weights)
2. Runs improved model with different configurations:
   - Different backbones (ResNet-18, ResNet-50)
   - Best augmentation setting (grayscale only based on ablation study)
3. Compares all results and generates a summary report
"""

import os
import sys
import json
import subprocess
import argparse
from datetime import datetime


def run_command(cmd, cwd=None):
    """Run a command and return output."""
    print(f"\n{'='*60}")
    print(f"Running: {cmd}")
    print(f"{'='*60}")

    result = subprocess.run(
        cmd,
        shell=True,
        cwd=cwd,
        capture_output=True,
        text=True
    )

    print(result.stdout)
    if result.stderr:
        # Filter out common warnings
        stderr_lines = result.stderr.split('\n')
        important_errors = [l for l in stderr_lines if 'Error' in l or 'Exception' in l]
        if important_errors:
            print("STDERR:", '\n'.join(important_errors))

    return result.returncode == 0, result.stdout, result.stderr


def parse_training_output(stdout):
    """Parse training output to extract metrics."""
    metrics = {
        'best_val_rmse': None,
        'test_rmse': None,
        'test_baseline_rmse': None,
        'final_train_loss': None
    }

    lines = stdout.split('\n')
    for line in lines:
        if 'Best Validation RMSE:' in line:
            try:
                metrics['best_val_rmse'] = float(line.split(':')[1].strip().replace('m', ''))
            except:
                pass
        elif 'Test RMSE:' in line and 'Baseline' not in line:
            try:
                metrics['test_rmse'] = float(line.split(':')[1].strip().replace('m', ''))
            except:
                pass
        elif 'Test Baseline RMSE:' in line:
            try:
                metrics['test_baseline_rmse'] = float(line.split(':')[1].strip().replace('m', ''))
            except:
                pass

    return metrics


def test_preprocessing():
    """Test preprocessing functions for both models."""
    print("\n" + "="*60)
    print("TESTING PREPROCESSING FUNCTIONS")
    print("="*60)

    results = {"baseline": False, "improved": False}

    # Test baseline preprocess
    print("\n[1/2] Testing baseline_model/preprocess.py...")
    try:
        sys.path.insert(0, 'baseline_model')
        from baseline_model.preprocess import INFERENCE_TRANSFORM, load_and_preprocess_image

        from PIL import Image
        import torch

        dummy_img = Image.new('RGB', (640, 480), color='red')
        dummy_img.save('test_image.jpg')

        tensor = load_and_preprocess_image('test_image.jpg', INFERENCE_TRANSFORM)

        assert tensor.shape == torch.Size([3, 224, 224]), f"Expected (3, 224, 224), got {tensor.shape}"
        print(f"  [OK] Output shape correct: {tensor.shape}")
        print(f"  [OK] Tensor min: {tensor.min():.4f}, max: {tensor.max():.4f}")

        results["baseline"] = True
        print("  [OK] Baseline preprocessing test PASSED")

        os.remove('test_image.jpg')

    except Exception as e:
        print(f"  [FAIL] Baseline preprocessing test FAILED: {e}")

    # Test improved preprocess
    print("\n[2/2] Testing improved_model/preprocess.py...")
    try:
        from improved_model.preprocess import get_train_transform, get_val_transform
        from PIL import Image
        import torch

        dummy_img = Image.new('RGB', (640, 480), color='blue')
        dummy_img.save('test_image.jpg')

        train_tf = get_train_transform()
        train_tensor = train_tf(dummy_img)
        assert train_tensor.shape == torch.Size([3, 224, 224]), f"Train: Expected (3, 224, 224), got {train_tensor.shape}"
        print(f"  [OK] Train transform output shape: {train_tensor.shape}")

        val_tf = get_val_transform()
        val_tensor = val_tf(dummy_img)
        assert val_tensor.shape == torch.Size([3, 224, 224]), f"Val: Expected (3, 224, 224), got {val_tensor.shape}"
        print(f"  [OK] Val transform output shape: {val_tensor.shape}")

        results["improved"] = True
        print("  [OK] Improved preprocessing test PASSED")

        os.remove('test_image.jpg')

    except Exception as e:
        print(f"  [FAIL] Improved preprocessing test FAILED: {e}")

    return results


def test_data_loading(dataset_name="rantyw/image2gps"):
    """Test that the HuggingFace dataset loads correctly."""
    print("\n" + "="*60)
    print("TESTING DATA LOADING")
    print("="*60)

    try:
        from datasets import load_dataset

        print(f"\nLoading dataset: {dataset_name}")

        for split in ["train", "validation", "test"]:
            print(f"\n  Loading '{split}' split...")
            ds = load_dataset(dataset_name, split=split)
            print(f"  [OK] {split}: {len(ds)} samples")
            print(f"    Features: {list(ds.features.keys())}")

            sample = ds[0]
            print(f"    Sample image size: {sample['image'].size}")
            print(f"    Sample coords: ({sample['Latitude']:.6f}, {sample['Longitude']:.6f})")

        print("\n  [OK] Data loading test PASSED")
        return True

    except Exception as e:
        print(f"\n  [FAIL] Data loading test FAILED: {e}")
        return False


def run_baseline_training(epochs=5, batch_size=32, pretrained=False):
    """Run baseline model training."""
    pretrained_str = "pretrained" if pretrained else "scratch"
    print("\n" + "="*60)
    print(f"EXPERIMENT: Baseline Model (ResNet-18, {pretrained_str})")
    print("="*60)

    cmd = f"python train.py --num_epochs {epochs} --batch_size {batch_size}"
    if pretrained:
        cmd += " --pretrained"

    success, stdout, stderr = run_command(cmd, cwd="baseline_model")
    metrics = parse_training_output(stdout)

    return success, metrics


def run_improved_training(backbone="resnet50", epochs=30, batch_size=32,
                          use_grayscale=True, use_blur=False, use_erasing=False):
    """Run improved model training with configurable settings."""
    aug_str = f"gray={use_grayscale}, blur={use_blur}, erase={use_erasing}"
    print("\n" + "="*60)
    print(f"EXPERIMENT: Improved Model ({backbone})")
    print(f"  Augmentations: {aug_str}")
    print("="*60)

    cmd = f"python train.py --backbone {backbone} --num_epochs {epochs} --batch_size {batch_size}"

    # Add augmentation flags
    if not use_grayscale:
        cmd += " --no_grayscale"
    if not use_blur:
        cmd += " --no_blur"
    if not use_erasing:
        cmd += " --no_erasing"

    success, stdout, stderr = run_command(cmd, cwd="improved_model")
    metrics = parse_training_output(stdout)

    return success, metrics


def run_comparison_experiments(args):
    """Run full comparison between baseline and improved models."""

    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "baseline_epochs": args.baseline_epochs,
            "improved_epochs": args.improved_epochs,
            "batch_size": args.batch_size
        },
        "preprocessing_tests": {},
        "data_loading": False,
        "experiments": []
    }

    # 1. Test preprocessing
    if args.test_preprocess:
        results["preprocessing_tests"] = test_preprocessing()

    # 2. Test data loading
    if args.test_data:
        results["data_loading"] = test_data_loading(args.dataset)

    # 3. Baseline experiments
    if args.run_baseline:
        # Baseline without pretrained weights (matches notebook)
        print("\n" + "="*70)
        print("BASELINE MODEL EXPERIMENTS")
        print("="*70)

        success, metrics = run_baseline_training(
            epochs=args.baseline_epochs,
            batch_size=args.batch_size,
            pretrained=False
        )
        results["experiments"].append({
            "name": "baseline_scratch",
            "model": "ResNet-18",
            "pretrained": False,
            "epochs": args.baseline_epochs,
            "success": success,
            "metrics": metrics
        })

    if args.run_baseline_pretrained:
        # Baseline with pretrained weights
        success, metrics = run_baseline_training(
            epochs=args.baseline_epochs,
            batch_size=args.batch_size,
            pretrained=True
        )
        results["experiments"].append({
            "name": "baseline_pretrained",
            "model": "ResNet-18",
            "pretrained": True,
            "epochs": args.baseline_epochs,
            "success": success,
            "metrics": metrics
        })

    # 4. Improved model experiments
    if args.run_improved:
        print("\n" + "="*70)
        print("IMPROVED MODEL EXPERIMENTS")
        print("="*70)

        # Test different backbones with best augmentation (grayscale only)
        for backbone in args.backbones:
            success, metrics = run_improved_training(
                backbone=backbone,
                epochs=args.improved_epochs,
                batch_size=args.batch_size,
                use_grayscale=True,  # Best from ablation
                use_blur=False,
                use_erasing=False
            )
            results["experiments"].append({
                "name": f"improved_{backbone}_grayscale",
                "model": backbone,
                "pretrained": True,
                "augmentation": "grayscale_only",
                "epochs": args.improved_epochs,
                "success": success,
                "metrics": metrics
            })

    # 5. Additional augmentation experiments
    if args.run_aug_comparison:
        print("\n" + "="*70)
        print("AUGMENTATION COMPARISON EXPERIMENTS")
        print("="*70)

        aug_configs = [
            {"name": "no_aug", "gray": False, "blur": False, "erase": False},
            {"name": "grayscale", "gray": True, "blur": False, "erase": False},
            {"name": "full_aug", "gray": True, "blur": True, "erase": True},
        ]

        for config in aug_configs:
            success, metrics = run_improved_training(
                backbone=args.aug_backbone,
                epochs=args.improved_epochs,
                batch_size=args.batch_size,
                use_grayscale=config["gray"],
                use_blur=config["blur"],
                use_erasing=config["erase"]
            )
            results["experiments"].append({
                "name": f"improved_{args.aug_backbone}_{config['name']}",
                "model": args.aug_backbone,
                "pretrained": True,
                "augmentation": config["name"],
                "epochs": args.improved_epochs,
                "success": success,
                "metrics": metrics
            })

    # Save results
    results_path = f"experiment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print_summary(results, results_path)

    return results


def print_summary(results, results_path):
    """Print a formatted summary of all experiments."""
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    print(f"Results saved to: {results_path}")
    print(f"Timestamp: {results['timestamp']}")
    print(f"\nPreprocessing tests: {results['preprocessing_tests']}")
    print(f"Data loading: {'PASSED' if results['data_loading'] else 'SKIPPED/FAILED'}")

    print("\n" + "-"*70)
    print(f"{'Experiment':<35} {'Status':<10} {'Val RMSE':<12} {'Test RMSE':<12}")
    print("-"*70)

    for exp in results["experiments"]:
        status = "[OK]" if exp["success"] else "[FAIL]"
        val_rmse = f"{exp['metrics']['best_val_rmse']:.2f}m" if exp['metrics'].get('best_val_rmse') else "N/A"
        test_rmse = f"{exp['metrics']['test_rmse']:.2f}m" if exp['metrics'].get('test_rmse') else "N/A"

        print(f"{exp['name']:<35} {status:<10} {val_rmse:<12} {test_rmse:<12}")

    print("="*70)

    # Find best model
    valid_experiments = [e for e in results["experiments"] if e['metrics'].get('test_rmse')]
    if valid_experiments:
        best = min(valid_experiments, key=lambda x: x['metrics']['test_rmse'])
        print(f"\nBest Model: {best['name']}")
        print(f"  Test RMSE: {best['metrics']['test_rmse']:.2f}m")
        if best['metrics'].get('best_val_rmse'):
            print(f"  Val RMSE: {best['metrics']['best_val_rmse']:.2f}m")


def main():
    parser = argparse.ArgumentParser(description='Run Image2GPS comparison experiments')

    # Test options
    parser.add_argument('--test-preprocess', action='store_true', default=True,
                        help='Test preprocessing functions')
    parser.add_argument('--test-data', action='store_true', default=True,
                        help='Test data loading from HuggingFace')
    parser.add_argument('--dataset', type=str, default='rantyw/image2gps',
                        help='HuggingFace dataset name')

    # Experiment options
    parser.add_argument('--run-baseline', action='store_true',
                        help='Run baseline model (no pretrained)')
    parser.add_argument('--run-baseline-pretrained', action='store_true',
                        help='Run baseline model (with pretrained)')
    parser.add_argument('--run-improved', action='store_true',
                        help='Run improved model with different backbones')
    parser.add_argument('--run-aug-comparison', action='store_true',
                        help='Run augmentation comparison experiments')
    parser.add_argument('--run-all', action='store_true',
                        help='Run all experiments')

    # Hyperparameters
    parser.add_argument('--baseline-epochs', type=int, default=5,
                        help='Epochs for baseline training (default: 5)')
    parser.add_argument('--improved-epochs', type=int, default=15,
                        help='Epochs for improved training (default: 15)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training (default: 32)')
    parser.add_argument('--backbones', nargs='+',
                        default=['resnet18', 'resnet50'],
                        help='Backbones to test for improved model')
    parser.add_argument('--aug-backbone', type=str, default='resnet50',
                        help='Backbone for augmentation comparison')

    # Quick test mode
    parser.add_argument('--quick', action='store_true',
                        help='Quick test mode (2 epochs)')

    args = parser.parse_args()

    # Handle --run-all
    if args.run_all:
        args.run_baseline = True
        args.run_baseline_pretrained = True
        args.run_improved = True
        args.run_aug_comparison = True

    # Handle --quick mode
    if args.quick:
        args.baseline_epochs = 2
        args.improved_epochs = 2
        args.batch_size = 16
        args.backbones = ['resnet18']
        args.aug_backbone = 'resnet18'

    # Run experiments
    run_comparison_experiments(args)


if __name__ == '__main__':
    main()
