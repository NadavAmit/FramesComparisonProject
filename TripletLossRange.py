import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import random
# Define the data loader
class TripletDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir

        self.classes = sorted(os.listdir(self.root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.image_paths = []
        for cls_name in self.classes:
            class_dir = os.path.join(self.root_dir, cls_name)
            images = sorted(os.listdir(class_dir))
            for image_name in images:
                image_path = os.path.join(class_dir, image_name)
                self.image_paths.append((image_path, cls_name))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        reference_path, reference_class = self.image_paths[idx]

        # Find similar image
        similar_class = None
        while similar_class != reference_class:
            similar_path, similar_class = random.choice(self.image_paths)

        # Find negative (different) image
        negative_class = None
        while negative_class == reference_class or negative_class == None:
            negative_path, negative_class = random.choice(self.image_paths)

        # Load images
        reference_image = Image.open(reference_path).convert("RGB")
        similar_image = Image.open(similar_path).convert("RGB")

        negative_image = Image.open(negative_path).convert("RGB")

        if self.transform is not None:
            reference_image = self.transform(reference_image)
            similar_image = self.transform(similar_image)
            negative_image = self.transform(negative_image)

        return reference_image, similar_image, negative_image

# Define the model
class TripletNet(nn.Module):
    def __init__(self, base_model):
        super(TripletNet, self).__init__()
        self.base_model = base_model

    def forward(self, reference, similar, negative):
        reference_output = self.base_model(reference)
        similar_output = self.base_model(similar)
        negative_output = self.base_model(negative)
        return reference_output, similar_output, negative_output

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set up data loaders
root_dir = "dataset"
image_size = 224
batch_size = 16

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = TripletDataset(root_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Load pre-trained ResNet model
resnet = models.resnet50(pretrained=True)
num_features = resnet.fc.in_features
resnet.fc = nn.Linear(num_features, 128)  # Output embedding size of 128
resnet = resnet.to(device)

Net = TripletNet(resnet)
# Define triplet loss
criterion = nn.TripletMarginLoss()

# Set up optimizer
optimizer = optim.Adam(resnet.parameters(), lr=0.001)

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (reference, similar, negative) in enumerate(dataloader):
        reference = reference.to(device)
        similar = similar.to(device)
        negative = negative.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        reference_output,similar_output,negative_output = Net(reference,similar,negative)


        # Compute triplet loss
        loss = criterion(reference_output, similar_output, negative_output)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {running_loss / 10}")
        running_loss = 0.0

print("Training finished.")

# Save the trained model
torch.save(resnet.state_dict(), "triplet_model.pth")
