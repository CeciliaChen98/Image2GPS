"""
Image2GPS: Automated experiment runner and testing script.

This script:
1. Tests preprocessing functions
2. Runs baseline and improved model training
3. Compares results and saves metrics for report
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
        print("STDERR:", result.stderr)

    return result.returncode == 0, result.stdout, result.stderr


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

        # Check transform output size
        from PIL import Image
        import torch

        # Create a dummy image
        dummy_img = Image.new('RGB', (640, 480), color='red')
        dummy_img.save('test_image.jpg')

        # Test transform
        tensor = load_and_preprocess_image('test_image.jpg', INFERENCE_TRANSFORM)

        assert tensor.shape == torch.Size([3, 224, 224]), f"Expected (3, 224, 224), got {tensor.shape}"
        print(f"  [OK] Output shape correct: {tensor.shape}")

        # Check normalization range
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

        # Create a dummy image
        dummy_img = Image.new('RGB', (640, 480), color='blue')
        dummy_img.save('test_image.jpg')

        # Test train transform
        train_tf = get_train_transform()
        train_tensor = train_tf(dummy_img)
        assert train_tensor.shape == torch.Size([3, 224, 224]), f"Train: Expected (3, 224, 224), got {train_tensor.shape}"
        print(f"  [OK] Train transform output shape: {train_tensor.shape}")

        # Test val transform
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

        # Test each split
        for split in ["train", "validation", "test"]:
            print(f"\n  Loading '{split}' split...")
            ds = load_dataset(dataset_name, split=split)
            print(f"  [OK] {split}: {len(ds)} samples")
            print(f"    Features: {list(ds.features.keys())}")

            # Check first sample
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
    print("\n" + "="*60)
    print("RUNNING BASELINE MODEL TRAINING")
    print("="*60)

    cmd = f"python train.py --num_epochs {epochs} --batch_size {batch_size}"
    if pretrained:
        cmd += " --pretrained"

    success, stdout, stderr = run_command(cmd, cwd="baseline_model")

    return success, stdout


def run_improved_training(backbone="resnet50", epochs=30, batch_size=32,
                          use_grayscale=True, use_blur=True, use_erasing=True):
    """Run improved model training with configurable augmentations."""
    aug_str = f"gray={use_grayscale}, blur={use_blur}, erase={use_erasing}"
    print("\n" + "="*60)
    print(f"RUNNING IMPROVED MODEL TRAINING")
    print(f"  Backbone: {backbone}")
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

    return success, stdout


def run_ablation_study(backbone="resnet50", epochs=30, batch_size=32):
    """
    Run augmentation ablation study.

    Tests different combinations of augmentations to analyze their impact.
    """
    print("\n" + "="*60)
    print("RUNNING AUGMENTATION ABLATION STUDY")
    print("="*60)

    # Define ablation configurations
    ablation_configs = [
        {"name": "base_only", "grayscale": False, "blur": False, "erasing": False},
        {"name": "base+grayscale", "grayscale": True, "blur": False, "erasing": False},
        {"name": "base+blur", "grayscale": False, "blur": True, "erasing": False},
        {"name": "base+erasing", "grayscale": False, "blur": False, "erasing": True},
        {"name": "base+gray+blur", "grayscale": True, "blur": True, "erasing": False},
        {"name": "full_augmentation", "grayscale": True, "blur": True, "erasing": True},
    ]

    results = []

    for config in ablation_configs:
        print(f"\n--- Ablation: {config['name']} ---")
        success, stdout = run_improved_training(
            backbone=backbone,
            epochs=epochs,
            batch_size=batch_size,
            use_grayscale=config["grayscale"],
            use_blur=config["blur"],
            use_erasing=config["erasing"]
        )

        results.append({
            "name": config["name"],
            "grayscale": config["grayscale"],
            "blur": config["blur"],
            "erasing": config["erasing"],
            "success": success
        })

    return results


def run_experiment_suite(args):
    """Run a suite of experiments for the report."""

    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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

    # 3. Run training experiments
    if args.run_baseline:
        print("\n" + "="*60)
        print("EXPERIMENT: Baseline Model (ResNet-18, no pretrained)")
        print("="*60)
        success, output = run_baseline_training(
            epochs=args.baseline_epochs,
            batch_size=args.batch_size,
            pretrained=False
        )
        results["experiments"].append({
            "name": "baseline_scratch",
            "model": "ResNet-18",
            "pretrained": False,
            "epochs": args.baseline_epochs,
            "success": success
        })

    if args.run_baseline_pretrained:
        print("\n" + "="*60)
        print("EXPERIMENT: Baseline Model (ResNet-18, pretrained)")
        print("="*60)
        success, output = run_baseline_training(
            epochs=args.baseline_epochs,
            batch_size=args.batch_size,
            pretrained=True
        )
        results["experiments"].append({
            "name": "baseline_pretrained",
            "model": "ResNet-18",
            "pretrained": True,
            "epochs": args.baseline_epochs,
            "success": success
        })

    if args.run_improved:
        for backbone in args.backbones:
            print("\n" + "="*60)
            print(f"EXPERIMENT: Improved Model ({backbone})")
            print("="*60)
            success, output = run_improved_training(
                backbone=backbone,
                epochs=args.improved_epochs,
                batch_size=args.batch_size
            )
            results["experiments"].append({
                "name": f"improved_{backbone}",
                "model": backbone,
                "pretrained": True,
                "epochs": args.improved_epochs,
                "success": success
            })

    # 4. Run ablation study
    if args.run_ablation:
        print("\n" + "="*60)
        print("RUNNING AUGMENTATION ABLATION STUDY")
        print("="*60)
        ablation_results = run_ablation_study(
            backbone=args.ablation_backbone,
            epochs=args.ablation_epochs,
            batch_size=args.batch_size
        )
        results["ablation_study"] = ablation_results

        # Print ablation summary
        print("\n--- Ablation Study Summary ---")
        for res in ablation_results:
            status = "[OK]" if res["success"] else "[FAIL]"
            print(f"  {status} {res['name']}: gray={res['grayscale']}, blur={res['blur']}, erase={res['erasing']}")

    # Save results summary
    results_path = f"experiment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    print(f"Results saved to: {results_path}")
    print(f"Preprocessing tests: {results['preprocessing_tests']}")
    print(f"Data loading: {'PASSED' if results['data_loading'] else 'SKIPPED/FAILED'}")
    print(f"Experiments run: {len(results['experiments'])}")

    for exp in results["experiments"]:
        status = "[OK]" if exp["success"] else "[FAIL]"
        print(f"  - {exp['name']}: {status}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Run Image2GPS experiments')

    # Test options
    parser.add_argument('--test-preprocess', action='store_true', default=True,
                        help='Test preprocessing functions')
    parser.add_argument('--test-data', action='store_true', default=True,
                        help='Test data loading from HuggingFace')
    parser.add_argument('--dataset', type=str, default='rantyw/image2gps',
                        help='HuggingFace dataset name')

    # Training options
    parser.add_argument('--run-baseline', action='store_true',
                        help='Run baseline model training (no pretrained)')
    parser.add_argument('--run-baseline-pretrained', action='store_true',
                        help='Run baseline model training (with pretrained)')
    parser.add_argument('--run-improved', action='store_true',
                        help='Run improved model training')
    parser.add_argument('--run-ablation', action='store_true',
                        help='Run augmentation ablation study')
    parser.add_argument('--run-all', action='store_true',
                        help='Run all experiments (excluding ablation)')

    # Hyperparameters
    parser.add_argument('--baseline-epochs', type=int, default=5,
                        help='Epochs for baseline training')
    parser.add_argument('--improved-epochs', type=int, default=30,
                        help='Epochs for improved training')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--backbones', nargs='+',
                        default=['resnet18', 'resnet50'],
                        help='Backbones to test for improved model')

    # Ablation study options
    parser.add_argument('--ablation-backbone', type=str, default='resnet50',
                        help='Backbone to use for ablation study')
    parser.add_argument('--ablation-epochs', type=int, default=30,
                        help='Epochs for each ablation experiment')

    # Quick test mode
    parser.add_argument('--quick', action='store_true',
                        help='Quick test mode (2 epochs, small batch)')

    args = parser.parse_args()

    # Handle --run-all
    if args.run_all:
        args.run_baseline = True
        args.run_baseline_pretrained = True
        args.run_improved = True

    # Handle --quick mode
    if args.quick:
        args.baseline_epochs = 2
        args.improved_epochs = 2
        args.ablation_epochs = 2
        args.batch_size = 16
        args.backbones = ['resnet18']
        args.ablation_backbone = 'resnet18'

    # Run experiments
    run_experiment_suite(args)


if __name__ == '__main__':
    main()
