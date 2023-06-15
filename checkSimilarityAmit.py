import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Define the data transformer
image_size = 224
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load the trained model
model = models.resnet50(pretrained=False)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 128)
model.load_state_dict(torch.load("triplet_model.pth"))
model.eval()

# Function to compute Euclidean distance between two images
def compute_distance(image_path1, image_path2):
    image1 = Image.open(image_path1).convert("RGB")
    image2 = Image.open(image_path2).convert("RGB")

    image1 = transform(image1).unsqueeze(0)
    image2 = transform(image2).unsqueeze(0)

    with torch.no_grad():
        output1 = model(image1)
        output2 = model(image2)

    distance = torch.norm(output1 - output2)

    return distance.item()

# Example usage
image_path1 = "dataset/car/cctv_0_45.jpeg"
image_path1 = "dataset/monkey/5.png"
#
image_path2 = "dataset/monkey/4.jpg"

distance = compute_distance(image_path1, image_path2)
print(f"Euclidean distance between the images: {distance}")
