import numpy as np

from PIL import Image

import torch
from torch.utils.data import Dataset

class VirtualStainingDataset(Dataset):
    def __init__(self, input_images, target_images, transform=None):
        """
        Args:
            root_dir (str): Directory with the input and target images.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.transform = transform
        self.input_images = input_images
        self.target_images = target_images

        assert len(self.input_images) == len(self.target_images), "Number of input and target images must match."

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        # Load the input and target images
        input_image = np.load(self.input_images[idx])
        target_image = np.load(self.target_images[idx])

        # Remove alpha channels
        if input_image.shape[-1] == 4:
            input_image = input_image[:, :, :3]
        if target_image.shape[-1] == 4:
            target_image = target_image[:, :, :3]
            
        # Apply any transformations if specified
        if self.transform:
            augmented = self.transform(image=input_image, target=target_image)
            input_image = augmented['image']                
            target_image = augmented['target']                       

        return input_image, target_image
