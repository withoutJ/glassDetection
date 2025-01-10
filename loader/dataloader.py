import os
from torch.utils.data import Dataset
from PIL import Image
import torch

class ImageDataset(Dataset):
    def __init__(self, dataset_dir, size=None, transform=None):
        self.images_dir = os.path.join(dataset_dir, "images") 
        self.masks_dir = os.path.join(dataset_dir, "masks") 
        self.image_names = sorted(os.listdir(self.images_dir))
        self.mask_names = sorted(os.listdir(self.masks_dir))
        if size:
            self.image_names = self.image_names[:size]
            self.mask_names = self.mask_names[:size]
        self.size = len(self.image_names)
        self.transform = transform

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.image_names[idx])
        mask_path = os.path.join(self.masks_dir, self.mask_names[idx])

        image = self.rgb_loader(img_path)
        mask = self.rgb_loader(mask_path)

        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask
    
    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')