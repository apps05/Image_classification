# Robust CNN Based Image Classification under Memory Constraints

## By Aprajita Agarwal (BT2024258)

CNN-based vehicle classifier trained under a 5 MB model size constraint.

## Classes
| Index | Class |
|-------|-------|
| 0 | Bus |
| 1 | Truck |
| 2 | Car |
| 3 | Bike |
| 4 | None |

## Requirements
```bash
pip install torch torchvision pillow tqdm
```

## Files
| File | Description |
|------|-------------|
| `vehicle_classifier.py` | Inference class, import this for predictions |
| `student_model.pth` | Trained model weights (4.14 MB) |
| `train_model.py` | Training script |
| `split.py` | Splits dataset into train/val folders |
| `trim.py` | Trims a class folder to a target image count |
| `report.pdf` | Assignment report |

## Running Inference
```python
from vehicle_classifier import VehicleClassifier

classifier = VehicleClassifier(model_path="student_model.pth")
idx = classifier.predict("image.jpg")
print(idx)  # e.g., 2 → Car
```

## Training from Scratch
1. Organise dataset into folders:
```
dataset_all/
    Bus/
    Truck/
    Car/
    Bike/
    None/
```
2. Run split and train:
```bash
python split.py       # creates dataset/train and dataset/val
python train_model.py # trains and saves student_model.pth
```

## Model Details
- Backbone: MobileNetV3 Small (pretrained on ImageNet)
- Input size: 224x224 (handles any input from 32x32 to 256x256)
- Inference: CPU only
- Model size: 4.14 MB
- Validation accuracy: 95.03%
- Robust to illumination changes and partial occlusions via data augmentation