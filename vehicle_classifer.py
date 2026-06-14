import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# -----------------------------
# Class Index Mapping (consistent across all submissions)
# -----------------------------
CLASS_IDX = {
    0: "Bus",
    1: "Truck",
    2: "Car",
    3: "Bike",
    4: "None"
}

# -----------------------------
# Inference Class
# DONT CHANGE THE INTERFACE OF THE CLASS
# -----------------------------
class VehicleClassifier:
    def __init__(self, model_path=None):
        self.device = torch.device("cpu")

        self.model = models.mobilenet_v3_small(weights=None)
        self.model.classifier = nn.Sequential(
        nn.Linear(576, 128),
        nn.Hardswish(),
        nn.Dropout(0.2),
        nn.Linear(128, 5)
        )

        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))

        self.model.eval()

        # Preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])

    def predict(self, image_path: str) -> int:
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(tensor)
            _, predicted = torch.max(outputs, 1)
        return predicted.item()

# -----------------------------
# Example Usage
# -----------------------------
if __name__ == "__main__":
    classifier = VehicleClassifier(model_path="student_model.pth") # load your trained weights
    idx = classifier.predict("test_image.jpg")
    print(f"Predicted Class Index: {idx}, Label: {CLASS_IDX[idx]}")
