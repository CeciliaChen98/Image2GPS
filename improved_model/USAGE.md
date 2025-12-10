# Image2GPS Improved Model - Usage Guide

## Overview

This improved model features:
- Multiple backbone architectures (ResNet-18, ResNet-50, ResNet-101, EfficientNet-B0)
- Enhanced regression head with batch normalization and dropout
- Learning rate warmup and cosine annealing schedule
- Early stopping and gradient clipping
- Mixed precision training (AMP) for faster training

## Setup

Install dependencies:
```bash
pip install torch torchvision pandas numpy pillow datasets geopy
```

## Training

Train the model on the CoconutYezi/released_img dataset:

```bash
python train.py
```

### Training Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `CoconutYezi/released_img` | HuggingFace dataset name |
| `--backbone` | `resnet50` | Backbone architecture (`resnet18`, `resnet50`, `resnet101`, `efficientnet_b0`) |
| `--pretrained` | `True` | Use ImageNet pretrained weights |
| `--hidden_dim` | `256` | Hidden dimension for regression head |
| `--dropout` | `0.5` | Dropout rate for regularization |
| `--batch_size` | `32` | Batch size |
| `--num_epochs` | `50` | Maximum number of training epochs |
| `--learning_rate` | `0.001` | Base learning rate |
| `--weight_decay` | `0.0001` | Weight decay for AdamW optimizer |
| `--warmup_epochs` | `5` | Number of warmup epochs |
| `--patience` | `10` | Early stopping patience |
| `--use_amp` | `True` | Use automatic mixed precision |
| `--val_split` | `0.2` | Validation split ratio |
| `--num_workers` | `4` | Number of data loader workers |
| `--output_dir` | `./checkpoints` | Directory to save models |

### Examples

Basic training with default settings (ResNet-50):
```bash
python train.py
```

Train with ResNet-18 (faster, lighter):
```bash
python train.py --backbone resnet18 --num_epochs 30
```

Train with EfficientNet-B0:
```bash
python train.py --backbone efficientnet_b0 --learning_rate 0.0005
```

Train with custom settings:
```bash
python train.py --backbone resnet50 --batch_size 64 --num_epochs 100 --patience 15 --output_dir ./my_models
```

Disable mixed precision (if GPU doesn't support it):
```bash
python train.py --no_amp
```

### Output

Training creates a timestamped directory in the output folder containing:
- `best_model.pth` - Full checkpoint with best validation RMSE
- `model.pt` - Model state dict for submission
- `norm_params.json` - GPS normalization parameters
- `config.json` - Training configuration
- `history.json` - Training metrics history

## Submission

After training, follow these steps to prepare for submission:

1. Copy `model.pt` from the output directory to `improved_model/`

2. Update `NORM_PARAMS` in `model.py` with values from `norm_params.json`:
   ```python
   NORM_PARAMS = {
       'lat_mean': <value from norm_params.json>,
       'lat_std': <value from norm_params.json>,
       'lon_mean': <value from norm_params.json>,
       'lon_std': <value from norm_params.json>,
   }
   ```

3. Submit the following files:
   - `preprocess.py`
   - `model.py`
   - `model.pt`

## Model Architecture

### Backbone Options

| Backbone | Parameters | Notes |
|----------|------------|-------|
| `resnet18` | ~11M | Lightweight, fast training |
| `resnet50` | ~25M | Good balance (default) |
| `resnet101` | ~44M | Deeper, better for complex patterns |
| `efficientnet_b0` | ~5M | Efficient with good accuracy |

### Regression Head

The improved regression head includes:
- Two fully connected layers with decreasing dimensions
- Batch normalization after each linear layer
- ReLU activation
- Dropout for regularization

## Data Augmentation

Training uses the following augmentations:
- Random resized crop (scale 0.8-1.0)
- Random horizontal flip
- Random rotation (±15 degrees)
- Color jitter (brightness, contrast, saturation, hue)
- Random grayscale
- Gaussian blur
- Random erasing

## Training Features

- **Learning Rate Warmup**: Gradual LR increase during first few epochs
- **Cosine Annealing**: Smooth LR decay after warmup
- **Differential Learning Rates**: Lower LR for pretrained backbone (0.1x)
- **Gradient Clipping**: Max norm of 1.0 to prevent exploding gradients
- **Early Stopping**: Stop training when validation RMSE stops improving
- **Mixed Precision**: FP16 training for faster computation (CUDA only)
