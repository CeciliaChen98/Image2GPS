# Image2GPS Final Project Guide

## CIS 4190/5190: Applied Machine Learning - Fall 2025

**Team Size:** 3 members  
**Deadline:** December 15, 2025  
**Project Track:** Option A - Image2GPS

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Timeline & Task Division](#2-timeline--task-division)
3. [Phase 1: Data Collection](#3-phase-1-data-collection)
4. [Phase 2: Data Processing & Dataset Creation](#4-phase-2-data-processing--dataset-creation)
5. [Phase 3: Model Development](#5-phase-3-model-development)
6. [Phase 4: Submission Files](#6-phase-4-submission-files)
7. [Phase 5: Evaluation & Leaderboard](#7-phase-5-evaluation--leaderboard)
8. [Phase 6: Report Writing](#8-phase-6-report-writing)
9. [Technical Specifications](#9-technical-specifications)
10. [Resources & Links](#10-resources--links)

---

## 1. Project Overview

### Goal
Build a computer vision model that predicts GPS coordinates (latitude, longitude) from images taken on Penn's campus.

### Test Region
The testing area is bounded by the rectangular region from **33rd and Walnut** to **34th and Spruce St** on Penn's campus.

### Evaluation Metric
**Average Haversine Distance (meters)** - Lower is better.

The Haversine formula calculates the great-circle distance between two points on Earth:
```
d(a, b) = 2R × arcsin(√(sin²(Δφ/2) + cos(φ₁)cos(φ₂)sin²(Δλ/2)))
```
Where R = 6,371,000 meters.

### Deliverables
- [ ] Collected dataset (uploaded to Hugging Face)
- [ ] Trained model files (`model.py`, `preprocess.py`, `model.pt`)
- [ ] 5-page project report
- [ ] At least one leaderboard submission

---

## 2. Timeline & Task Division

### Suggested Timeline

| Week | Dates | Phase | Tasks |
|------|-------|-------|-------|
| 1 | Nov 13-19 | Setup | Team formation, read docs, set up environment |
| 2 | Nov 20-26 | Data Collection | Collect images on campus |
| 3 | Nov 27-Dec 3 | Data Processing | Clean data, create HF dataset, run baseline |
| 4 | Dec 4-10 | Model Development | Train & improve models, submit to leaderboard |
| 5 | Dec 11-15 | Finalization | Final submissions, write report |

### Suggested Task Division (3 Members)

**Member A: Data Lead**
- Coordinate data collection sessions
- Ensure GPS/EXIF data is properly captured
- Handle data cleaning and CSV creation
- Upload dataset to Hugging Face

**Member B: Model Lead**
- Set up training environment (Colab/local GPU)
- Implement and train baseline model
- Experiment with model improvements
- Handle model weight saving

**Member C: Integration & Submission Lead**
- Create submission files (`model.py`, `preprocess.py`)
- Test local evaluation scripts
- Submit to leaderboard
- Lead report writing

---

## 3. Phase 1: Data Collection

### 3.1 Setup Your Phone for GPS Capture

**iPhone:**
1. Go to Settings → Privacy → Location Services
2. Enable Location Services
3. Find Camera app → Set to "While Using the App"

**Android:**
1. Open Camera app → Settings
2. Enable "Location tags" or "Save location"
3. Ensure GPS is turned on in phone settings

### 3.2 Data Collection Protocol

**IMPORTANT:** Follow these guidelines strictly for consistent data:

1. **Location**: Only collect along walkways within the test region (33rd & Walnut to 34th & Spruce)

2. **At Each Location**:
   - Stand at a fixed point
   - Rotate and take **8 photos** from different angles (every ~45°)
   - Keep phone upright (portrait orientation)
   - Do NOT zoom in or out

3. **Coverage Strategy**:
   - Divide the campus region among team members
   - Aim for comprehensive coverage of all walkways
   - Take photos in different weather/lighting conditions for robustness

4. **Quality Checks**:
   - Verify EXIF data exists in sample photos before mass collection
   - Avoid photos pointing at the sky
   - Ensure images are clear and not blurry

### 3.3 Verify EXIF Data

Before collecting many images, verify your phone captures GPS:

```python
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

def get_gps_from_image(image_path):
    image = Image.open(image_path)
    exif_data = image._getexif()
    
    if exif_data is None:
        return None
    
    gps_info = {}
    for tag, value in exif_data.items():
        tag_name = TAGS.get(tag, tag)
        if tag_name == "GPSInfo":
            for gps_tag in value:
                gps_tag_name = GPSTAGS.get(gps_tag, gps_tag)
                gps_info[gps_tag_name] = value[gps_tag]
    
    return gps_info

# Test with a sample image
gps = get_gps_from_image("test_image.jpg")
print(gps)  # Should show GPSLatitude, GPSLongitude, etc.
```

---

## 4. Phase 2: Data Processing & Dataset Creation

### 4.1 Extract GPS from Images

Create a script to extract GPS coordinates and build your CSV:

```python
import os
import pandas as pd
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

def get_decimal_coords(gps_info):
    """Convert GPS coordinates to decimal degrees."""
    def convert_to_degrees(value):
        d, m, s = value
        return d + (m / 60.0) + (s / 3600.0)
    
    lat = convert_to_degrees(gps_info['GPSLatitude'])
    lon = convert_to_degrees(gps_info['GPSLongitude'])
    
    if gps_info.get('GPSLatitudeRef') == 'S':
        lat = -lat
    if gps_info.get('GPSLongitudeRef') == 'W':
        lon = -lon
    
    return lat, lon

def process_images(image_folder, output_csv):
    data = []
    
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.heic')):
            filepath = os.path.join(image_folder, filename)
            try:
                image = Image.open(filepath)
                exif = image._getexif()
                
                if exif:
                    for tag, value in exif.items():
                        if TAGS.get(tag) == "GPSInfo":
                            gps_info = {GPSTAGS.get(t, t): v for t, v in value.items()}
                            lat, lon = get_decimal_coords(gps_info)
                            data.append({
                                'image_path': filename,
                                'Latitude': lat,
                                'Longitude': lon
                            })
                            break
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(data)} entries to {output_csv}")
    return df

# Usage
df = process_images("./raw_images/", "metadata.csv")
```

### 4.2 Data Cleaning Checklist

- [ ] Remove images without valid GPS data
- [ ] Remove duplicate/very similar images
- [ ] Verify all coordinates are within the test region bounds
- [ ] Check for outliers in GPS coordinates
- [ ] Ensure image quality (no blurry images)
- [ ] Standardize image format (convert HEIC to JPG if needed)

### 4.3 Train/Validation Split

```python
from sklearn.model_selection import train_test_split

df = pd.read_csv("metadata.csv")
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

train_df.to_csv("train_metadata.csv", index=False)
val_df.to_csv("val_metadata.csv", index=False)
```

### 4.4 Upload to Hugging Face

Follow the [Hugging Face Dataset Tutorial](https://huggingface.co/docs/datasets/upload_dataset) to upload your dataset.

```python
from datasets import Dataset, DatasetDict, Image as HFImage
from huggingface_hub import HfApi

# Create dataset
def create_hf_dataset(csv_path, image_folder):
    df = pd.read_csv(csv_path)
    df['image'] = df['image_path'].apply(lambda x: os.path.join(image_folder, x))
    
    dataset = Dataset.from_pandas(df)
    dataset = dataset.cast_column("image", HFImage())
    return dataset

train_dataset = create_hf_dataset("train_metadata.csv", "./images/")
test_dataset = create_hf_dataset("val_metadata.csv", "./images/")

dataset_dict = DatasetDict({
    "train": train_dataset,
    "test": test_dataset
})

# Push to Hub
dataset_dict.push_to_hub("YOUR_ORG/YOUR_DATASET_NAME")
```

---

## 5. Phase 3: Model Development

### 5.1 Environment Setup

**Required Libraries:**
```bash
pip install torch torchvision numpy pandas scikit-learn opencv-python
pip install datasets huggingface_hub geopy
```

**Note:** Backend uses `torch==2.9.1`. Test compatibility.

### 5.2 Baseline Model Architecture

The baseline uses **ResNet-18** with modified output layer:

```python
import torch
import torch.nn as nn
import torchvision.models as models

# Load ResNet-18
resnet = models.resnet18(weights=None)  # or pretrained=True for transfer learning

# Modify final layer for GPS regression (2 outputs: lat, lon)
num_features = resnet.fc.in_features
resnet.fc = nn.Linear(num_features, 2)
```

### 5.3 Data Preprocessing

**Image Transforms:**
```python
import torchvision.transforms as transforms

# Training transforms (with augmentation)
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Inference transforms (no augmentation)
inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

**GPS Normalization:**
```python
# During training, normalize GPS coordinates
lat_mean = df['Latitude'].mean()
lat_std = df['Latitude'].std()
lon_mean = df['Longitude'].mean()
lon_std = df['Longitude'].std()

# Normalize
lat_normalized = (lat - lat_mean) / lat_std
lon_normalized = (lon - lon_mean) / lon_std

# IMPORTANT: Denormalize before evaluation!
lat_pred = lat_normalized * lat_std + lat_mean
lon_pred = lon_normalized * lon_std + lon_mean
```

### 5.4 Training Configuration

```python
# Hyperparameters (baseline)
batch_size = 32
learning_rate = 0.001
num_epochs = 10-15
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
```

### 5.5 Training Loop Template

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

for epoch in range(num_epochs):
    # Training
    model.train()
    for images, coords in train_loader:
        images, coords = images.to(device), coords.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, coords)
        loss.backward()
        optimizer.step()
    
    scheduler.step()
    
    # Validation
    model.eval()
    with torch.no_grad():
        # Calculate Haversine distance on validation set
        pass

# Save model
torch.save(model.state_dict(), 'model.pt')
```

### 5.6 Ideas for Improvement

- [ ] **Transfer Learning**: Use pretrained weights (`weights='IMAGENET1K_V1'`)
- [ ] **Better Architecture**: Try ResNet-50, EfficientNet, or Vision Transformers
- [ ] **Data Augmentation**: More aggressive augmentation for robustness
- [ ] **Loss Function**: Try Haversine loss instead of MSE
- [ ] **Ensemble**: Combine multiple models
- [ ] **Multi-scale Input**: Process images at different resolutions

---

## 6. Phase 4: Submission Files

### 6.1 Required Files

You must submit:
1. `model.py` - Model definition
2. `preprocess.py` - Data preprocessing
3. `model.pt` (optional) - Trained weights

### 6.2 preprocess.py Template

```python
import pandas as pd
import torch
from PIL import Image
import torchvision.transforms as transforms
import os

def prepare_data(csv_path: str):
    """
    Args:
        csv_path: Path to CSV file with columns for image path and GPS coordinates
    
    Returns:
        X: Tensor of preprocessed images
        y: Tensor of [latitude, longitude] pairs in DEGREES (raw, not normalized)
    """
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Find image path column
    img_col = None
    for col in ['image_path', 'filepath', 'image', 'path', 'file_name']:
        if col in df.columns:
            img_col = col
            break
    
    # Find coordinate columns
    lat_col = None
    for col in ['Latitude', 'latitude', 'lat']:
        if col in df.columns:
            lat_col = col
            break
    
    lon_col = None
    for col in ['Longitude', 'longitude', 'lon']:
        if col in df.columns:
            lon_col = col
            break
    
    # Get base directory from csv path
    base_dir = os.path.dirname(csv_path)
    
    # Transform for inference
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Process images
    images = []
    for img_path in df[img_col]:
        full_path = os.path.join(base_dir, img_path)
        img = Image.open(full_path).convert('RGB')
        img_tensor = transform(img)
        images.append(img_tensor)
    
    X = torch.stack(images)
    
    # Get coordinates (raw degrees, NOT normalized)
    y = torch.tensor(df[[lat_col, lon_col]].values, dtype=torch.float32)
    
    return X, y
```

### 6.3 model.py Template

```python
import torch
import torch.nn as nn
import torchvision.models as models

class IMG2GPS(nn.Module):
    def __init__(self):
        super(IMG2GPS, self).__init__()
        
        # Load ResNet-18 backbone
        self.backbone = models.resnet18(weights=None)
        
        # Modify final layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, 2)
        
        # HARD-CODE normalization stats from your training data
        self.lat_mean = 39.9522  # Example - replace with your values
        self.lat_std = 0.0015
        self.lon_mean = -75.1932
        self.lon_std = 0.0020
    
    def forward(self, x):
        # Get normalized predictions
        out = self.backbone(x)
        return out
    
    def predict(self, x):
        """
        Returns predictions in raw degrees (denormalized).
        """
        self.eval()
        with torch.no_grad():
            # Get normalized output
            out = self.forward(x)
            
            # Denormalize to raw degrees
            lat = out[:, 0] * self.lat_std + self.lat_mean
            lon = out[:, 1] * self.lon_std + self.lon_mean
            
            return torch.stack([lat, lon], dim=1)

def get_model():
    """Factory function to create model instance."""
    return IMG2GPS()
```

### 6.4 Local Testing

Before submitting, test locally using the provided evaluation script:

```bash
python eval_project_a.py --csv reference/metadata.csv --model model.py --weights model.pt
```

---

## 7. Phase 5: Evaluation & Leaderboard

### 7.1 Leaderboard Link
[Project A: IMG2GPS Leaderboard](https://huggingface.co/spaces/YOUR_LEADERBOARD_LINK)

### 7.2 Submission Process

1. Go to the leaderboard page
2. Enter your **Group ID** and **Alias**
3. Upload your files:
   - `model.py`
   - `preprocess.py`
   - `model.pt` (if needed)
4. Wait for evaluation to complete

### 7.3 Understanding Results

The leaderboard shows **Average Haversine Distance in meters**:
- **< 50m**: Excellent
- **50-100m**: Good
- **100-200m**: Baseline level
- **> 200m**: Needs improvement

---

## 8. Phase 6: Report Writing

### 8.1 Report Structure (5 pages)

**1. Introduction (0.5 page)**
- Problem description
- Approach overview

**2. Data Collection (1 page)**
- Collection procedure and protocol
- Data cleaning and curation process
- Dataset statistics (size, distribution, splits)
- Link to Hugging Face dataset

**3. Model Design (1.5 pages)**
- Architecture choices and justification
- Training procedure
- Hyperparameter tuning
- Iterative improvements

**4. Evaluation (1 page)**
- Metrics used (Haversine distance)
- Results on validation set
- Leaderboard performance
- Comparison with baseline

**5. Exploratory Component (1 page)**
- What novel contribution did you make?
- Options: analysis, technique improvement, ablation studies, etc.

### 8.2 Figures to Include

- [ ] Map showing data collection locations
- [ ] Training/validation loss curves
- [ ] Prediction visualization on sample images
- [ ] Performance comparison table

---

## 9. Technical Specifications

### 9.1 Backend Environment

```
numpy
pandas
torch==2.9.1
torchvision
scikit-learn
opencv-python
```

### 9.2 I/O Contract Summary

**preprocess.py:**
```python
def prepare_data(csv_path: str) -> (X, y)
# X: images as tensor
# y: [lat, lon] pairs in DEGREES (raw)
```

**model.py:**
```python
def get_model() -> model_instance
# OR
class Model/IMG2GPS:
    def predict(batch) -> [lat, lon] in DEGREES
    # OR
    def __call__(batch) -> [lat, lon] in DEGREES
```

### 9.3 Common Pitfalls to Avoid

1. **Forgetting to denormalize**: Output must be raw degrees, not normalized
2. **Wrong column names**: Use exact column names from contract
3. **Missing weights**: Ensure `model.pt` keys match model parameters
4. **Image size mismatch**: Always resize to 224×224
5. **Different transforms**: Use same normalization mean/std as training

---

## 10. Resources & Links

### Course Materials
- [Baseline Notebook](doc/Release_baseline_model.ipynb)
- [Submission Guidelines](doc/Submission_Guideline.pdf)
- [Project Introduction](doc/Final_Project_Intro.pdf)

### External Resources
- [Hugging Face Datasets Tutorial](https://huggingface.co/docs/datasets/upload_dataset)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [torchvision Models](https://pytorch.org/vision/stable/models.html)

### Quick Reference
- **Test Region**: 33rd & Walnut to 34th & Spruce St, Penn Campus
- **Image Size**: 224 × 224 pixels
- **Output Format**: [latitude, longitude] in degrees
- **Metric**: Haversine distance (meters) - lower is better

---

## Checklist Before Final Submission

### Data
- [ ] All images have valid GPS data
- [ ] Data is within test region bounds
- [ ] Dataset uploaded to Hugging Face
- [ ] Train/test split is reasonable

### Model
- [ ] Model beats baseline (< 88m RMSE)
- [ ] Weights saved correctly
- [ ] Model loads without errors

### Submission Files
- [ ] `preprocess.py` follows contract
- [ ] `model.py` follows contract
- [ ] Local evaluation passes
- [ ] Leaderboard submission successful

### Report
- [ ] 5 pages, proper format
- [ ] All sections covered
- [ ] Hugging Face dataset link included
- [ ] Figures and tables included

---

**Good luck with your project! 🎯**
