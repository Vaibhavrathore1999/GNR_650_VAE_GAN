import torch
import torch.nn as nn
import torchvision.transforms as transforms
from datasets import load_dataset
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Function to display a grid of images
def show_images(images, labels, num_images=16):
    plt.figure(figsize=(8, 8))
    for i in range(num_images):
        plt.subplot(4, 4, i + 1)
        img = images[i].permute(1, 2, 0)  # Change from (C, H, W) to (H, W, C) for displaying
        img = img * 0.5 + 0.5  # Unnormalize the image (assuming normalization of 0.5 mean and std)
        plt.imshow(img)
        plt.title(f"Label: {labels[i].item()}")
        plt.axis("off")
    plt.show()

# Custom dataset class for Million-AID to handle transformations
class MillionAIDDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Access the image directly from the dataset
        image = self.dataset[idx]['image']
        
        # Assuming 'label_1' is the primary label; adjust if you need 'label_2' or 'label_3'
        label = self.dataset[idx]['label_3']
        
        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Function to create DataLoader for Million-AID
def dataloader(batch_size):
    # Load the Million-AID dataset
    dataset = load_dataset("jonathan-roberts1/Million-AID", split="train")

    # Define the transformations to apply to each image
    transform = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Wrap dataset in the custom Dataset class
    transformed_dataset = MillionAIDDataset(dataset, transform=transform)
    
    # Create DataLoader
    data_loader = DataLoader(transformed_dataset, batch_size=batch_size, shuffle=True)
    return data_loader

# Example usage to load data and verify
batch_size = 64
data_loader = dataloader(batch_size)

# Check a single batch to verify everything works
for images, labels in data_loader:
    print("Batch of images:", images.shape)
    print("Batch of labels:", labels)
    show_images(images, labels)
    break
