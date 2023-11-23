from torch.utils.data import Dataset
import pandas as pd
import os
from skimage import io
import torch
from PIL import Image


class MyDataset(Dataset):
    def __init__(self, file, root_dir, transform=None):
        self.annotations = pd.read_csv(file, sep=";", header=None)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = Image.open(path)
        image = image.resize((28, 28))
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))

        if self.transform:
            image = self.transform(image)

        return image, y_label
