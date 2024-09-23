import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class Dataset(torch.utils.data.Dataset):
    def __init__(self, root, loadSize, fineSize, transform=None):
        self.root = root
        self.paths = self.make_dataset(self.root)
        self.loadSize = loadSize
        self.fineSize = fineSize
        self.transform = transform

    def make_dataset(self, dir):
        images = []
        for fname in os.listdir(dir):
            if any(fname.endswith(ext) for ext in ['.jpg', '.png', '.jpeg']):
                path = os.path.join(dir, fname)
                images.append(path)
        return images

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path).convert('RGB')

        if self.transform:
            img = self.transform(img)
        else:
            img = self.default_transform(img)

        label = 0  # Assuming no labels
        return img, label

    def default_transform(self, img):
        transform = transforms.Compose([
            transforms.Resize(self.fineSize),
            transforms.CenterCrop(self.fineSize),
            transforms.ToTensor(),
        ])
        return transform(img)

    def __len__(self):
        return len(self.paths)
