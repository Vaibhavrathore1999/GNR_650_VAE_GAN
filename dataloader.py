# import torch
# import torch.nn as nn
# import torchvision
# import torchvision.transforms as transforms
# import torchvision.utils as utils
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# def dataloader(batch_size):
#   dataroot="/raid/biplab/sarthak/GNR_650/Project/VAE-GAN-PYTORCH"
#   transform=transforms.Compose([ transforms.Resize(64),transforms.CenterCrop(64),transforms.ToTensor(),transforms.Normalize((0.5),(0.5))])
#   dataset = torchvision.datasets.MNIST(root=dataroot, train=True, transform=transform, download=True)
#   data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
#   return data_loader


import torch
import torch.nn as nn
import torchvision.transforms as transforms
from datasets import load_dataset
# import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from PIL import Image
Image.MAX_IMAGE_PIXELS = None 
# Set the device
# device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')

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
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Wrap dataset in the custom Dataset class
    transformed_dataset = MillionAIDDataset(dataset, transform=transform)
    
    # Create DataLoader
    data_loader = DataLoader(transformed_dataset, batch_size=batch_size, shuffle=True)
    return data_loader
