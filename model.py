import torch
from torchvision import transforms, models
from PIL import Image
import io
import os

MODEL_PATH = os.getenv("MODEL_PATH", "checkpoints/foodnet.pth")

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location=device)

    model = models.mobilenet_v2(weights=None)
    num_classes = len(checkpoint["classes"])
    model.classifier[1] = torch.nn.Linear(
        model.classifier[1].in_features, num_classes
    )

    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    model.classes = checkpoint["classes"]
    return model

def predict_image_class(model, img_bytes):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)  # move to same device as model

    with torch.no_grad():
        outputs = model(image)
        _, pred = torch.max(outputs, 1)

    return model.classes[pred.item()]
