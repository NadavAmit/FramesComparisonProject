import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F

image_size = 224
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


model = models.resnet50(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 128)

#----- Load the trained model ------
model.load_state_dict(
    torch.load("triplet_model2.pth" , map_location=torch.device('cpu')))
model.eval()


def calculate_cosine_similarity(image_path1, image_path2):
    image1 = Image.open(image_path1).convert("RGB")
    image2 = Image.open(image_path2).convert("RGB")
    # Convert images to tensors
    image1_tensor = transform(image1).unsqueeze(0)
    image2_tensor = transform(image2).unsqueeze(0)

    # Get embeddings using the model
    with torch.no_grad():
        output1 = model(image1_tensor)
        output2 = model(image2_tensor)

    # Calculate cosine similarity score
    similarity_score = cosine_similarity(output1, output2)

    similarity_percentage = (similarity_score[0][0] + 1) / 2 * 100

    return similarity_percentage


def compute_distance(image_path1, image_path2):
    image1 = Image.open(image_path1).convert("RGB")
    image2 = Image.open(image_path2).convert("RGB")

    image1 = transform(image1).unsqueeze(0)
    image2 = transform(image2).unsqueeze(0)

    with torch.no_grad():
        output1 = model(image1)
        output2 = model(image2)

    # Normalize the embeddings
    output1 = F.normalize(output1, p=2, dim=1)
    output2 = F.normalize(output2, p=2, dim=1)

    distance = torch.norm(output1 - output2)

    return distance.item()



# -----Start Here------
root_dir = "../dataset"
image_path1 = root_dir + "/people/transformed_1.png"
image_path2 = root_dir + "/girls/transformed_1.png"

distance = compute_distance(image_path1, image_path2)
print(f"Euclidean distance between the images: {distance}")
similarity_percentage = calculate_cosine_similarity(image_path1, image_path2)
print(f"Cosine Similarity Percentage: {similarity_percentage:.2f}%")
