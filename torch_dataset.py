import os
import numpy as np
import torch

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from config import *
from PIL import Image
from typing import List

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        # Do not forget to transpose again when rendering images!
        sample = np.array(sample).transpose((2, 0, 1))
        return torch.tensor(sample, dtype = torch.float32)

class CelebADataset(Dataset):
    def __init__(self, path, transform = None, alpha = 0.1):
        super().__init__()
        self.path = path
        self.transform = transform
        self.alpha = alpha

        # Consider not loading images list in memory, extract image from directory
        self.images = os.listdir(path)
        self.num_images = int(len(self.images) * self.alpha)
        self.images = self.images[:self.num_images]
        # Train dataset should be shuffled on every load to get different initial batches compared to
        # previous training session. Therefore, this should not be seeded!
        self.images = list(np.random.permutation(self.images))

    def __len__(self):
        return self.num_images
    
    def __getitem__(self, idx):
        target = self.images[idx]
        image = Image.open(os.path.join(self.path, target))
        return self.transform(image) / 255.0

def get_images(dataset: str, ids: List[int]) -> torch.Tensor:
    res = torch.empty((len(ids), 3, IMAGE_SIZE, IMAGE_SIZE), dtype = torch.float32)

    transform = transforms.Compose([
        transforms.CenterCrop(CENTER_CROP_SIZE),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        ToTensor()
    ])

    for i, id in enumerate(ids):
        target = os.path.join(dataset, "{:d}.jpg".format(id))
        image = Image.open(target)
        # changing [H x W x C] to [C x H x W] to suit torch format
        image = transform(image)
        res[i] = image / 255.0
    return res

def get_data_loader(path, alpha = 0.1) -> DataLoader:
    transform = transforms.Compose([
        transforms.CenterCrop(CENTER_CROP_SIZE),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        ToTensor()
    ])

    dataset = CelebADataset(path, transform = transform, alpha = alpha)
    # generator kwarg to implement seeding, num_workersr kwarg for parallell loading
    # pin_memory = True speeds up the process of transfering batches to GPU
    loader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True, pin_memory = True)
    return loader