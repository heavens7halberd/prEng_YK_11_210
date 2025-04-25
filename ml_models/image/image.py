import torch
from torchvision import models, transforms
from PIL import Image
import io

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def classify_image(image):
    try:
        image = Image.open(io.BytesIO(image))
        image = image.convert("RGB")
        input_tensor = preprocess(image).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)

        _, predicted_idx = torch.max(output, 1)

        with open("imagenet_classes.txt") as f:
            labels = [line.strip() for line in f.readlines()]

        return labels[predicted_idx.item()]
    except Exception as e:
        return f"Error: {e}"
