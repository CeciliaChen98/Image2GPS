# Image2GPS Usage Guide

## Setup

Install dependencies:
```bash
pip install -r requirements.txt
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
| `--batch_size` | 32 | Batch size |
| `--num_epochs` | 15 | Number of training epochs |
| `--learning_rate` | 0.001 | Learning rate |
| `--val_split` | 0.2 | Validation split ratio |
| `--output_dir` | `./checkpoints` | Directory to save models |

### Example

```bash
python train.py --num_epochs 20 --batch_size 64 --output_dir ./my_models
```

### Output

Training saves two files in the output directory:
- `best_model.pth` - Model with lowest validation RMSE
- `final_model.pth` - Model from the last epoch

## Inference

### Predict on a single image

```bash
python inference.py --checkpoint ./checkpoints/best_model.pth --image path/to/image.jpg
```

Output:
```
Prediction for path/to/image.jpg:
  Latitude: 39.952345
  Longitude: -75.193456
  Google Maps: https://www.google.com/maps?q=39.952345,-75.193456
```

### Predict on a folder of images

```bash
python inference.py --checkpoint ./checkpoints/best_model.pth --image_dir ./my_images
```

### Save predictions to CSV

```bash
python inference.py --checkpoint ./checkpoints/best_model.pth --image_dir ./my_images --output predictions.csv
```

Output CSV format:
```
filename,latitude,longitude
image1.jpg,39.952345,-75.193456
image2.jpg,39.951234,-75.194567
```
