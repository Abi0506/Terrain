import torch
from torchvision import transforms
from PIL import Image
import os
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import torch.nn as nn


class_names = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
    'Pasture', 'PermanentCrop', 'River', 'Residential', 'SeaLake'
]

def load_model():
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))
    model.load_state_dict(torch.load("efficientnet_eurosat.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    img_t = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(img_t)
        _, pred = torch.max(output, 1)
    return class_names[pred.item()]
