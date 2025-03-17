# classification/dataloader.py
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Subset

class BinaryDataset(Dataset):
    def __init__(self, dataset, indices, class_mapping, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.class_mapping = class_mapping
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image, label = self.dataset[self.indices[idx]]
        # Convert label to binary (0 or 1) and ensure it's a float
        binary_label = torch.tensor(float(self.class_mapping[label]), dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
            
        return image, binary_label, self.indices[idx]  # Return original index as well

class ImageDataLoader:
    def __init__(self, data_dir, batch_size=32, img_size=(224, 224)):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.img_size = img_size

        self.transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def get_data_loader(self):
        # Load CIFAR-10 dataset
        dataset = datasets.CIFAR10(root=self.data_dir, train=True, download=True)

        # Filter for binary classification (e.g., classes 3 (cat) and 5 (dog))
        binary_classes = [3, 5]  # Class indices for 'cat' and 'dog'
        indices = [i for i, label in enumerate(dataset.targets) if label in binary_classes]
        
        # Create class mapping: 3 -> 0, 5 -> 1
        class_mapping = {3: 0, 5: 1}
        
        # Create binary dataset with transformation
        binary_dataset = BinaryDataset(dataset, indices, class_mapping, self.transform)

        return DataLoader(binary_dataset, batch_size=self.batch_size, shuffle=True)