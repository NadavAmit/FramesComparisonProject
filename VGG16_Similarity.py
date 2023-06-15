import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Load pre-trained VGG16 model
vgg16 = models.vgg16(pretrained=True)

# # Remove the last fully connected layer from VGG16
# features = list(vgg16.features.children())
# fc_layers = list(vgg16.classifier.children())[:-1]  # Exclude the last fully connected layer
# features.extend(fc_layers)
#
# # Construct the modified VGG16 model
# vgg16_modified = nn.Sequential(*features)


vgg16.classifier = vgg16.classifier[:-1]

# Set the modified model to evaluation mode
vgg16.eval()

# Preprocess the input image
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    return image

# Calculate the feature vector for an image
def get_feature_vector(image):
    feature_vector = vgg16(image)
    feature_vector = feature_vector.view(feature_vector.size(0), -1)
    return feature_vector

# Compute similarity between two images
def compute_similarity(image_path1, image_path2):
    # Preprocess the images
    image1 = preprocess_image(image_path1)
    image2 = preprocess_image(image_path2)

    # Get feature vectors for the images
    feature_vector1 = get_feature_vector(image1)
    feature_vector2 = get_feature_vector(image2)

    # Compute the cosine similarity between the feature vectors
    similarity = torch.nn.functional.cosine_similarity(feature_vector1, feature_vector2).item()
    return similarity

# Example usage
image_path1 = 'ImageDataSet/SourceImages/1.jpeg'
image_path2 = 'ImageDataSet/SourceImages/4.jpg'
image_path3 = 'ImageDataSet/SourceImages/5.png'

similarity_score = compute_similarity(image_path1, image_path2)
print("Similarity Score:", similarity_score)
similarity_score = compute_similarity(image_path2, image_path3)
print("Similarity Score:", similarity_score)
