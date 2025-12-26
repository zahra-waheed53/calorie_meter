import torch
from torchvision import transforms
from PIL import Image
import io
import os

MODEL_PATH = os.getenv("MODEL_PATH", "checkpoints/foodnet.pth")

def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))

    from torchvision import models
    model = models.resnet50(weights=None)

    num_classes = len(checkpoint["classes"])
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    model.classes = checkpoint["classes"]

    return model


def predict_image_class(model, img_bytes):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output, 1)

    return model.classes[predicted.item()]
