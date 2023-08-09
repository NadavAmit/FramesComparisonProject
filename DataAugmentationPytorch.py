import os
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image



my_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.05),
    transforms.RandomGrayscale(p=0.2),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    transforms.RandomRotation(degrees=30),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.0,0.0,0.0],std=[1.0,1.0,1.0])
])

def generate_dataset(image_path, n, output_dir):
    # Load the input image
    image = Image.open(image_path).convert('RGB')

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Apply transforms and save transformed images
    for i in range(n):
        transformed_image = my_transforms(image)
        save_image(transformed_image, os.path.join(output_dir, f"transformed_{i+1}.png"))

image_path = 'dataset/sourceFrames/Smile.png'
numberOfImages = 50  #Number of transformed images to generate
output_dir = 'dataset/smile'  # Directory to save the transformed images

generate_dataset(image_path, numberOfImages, output_dir)