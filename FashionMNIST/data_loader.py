import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
from torchvision.io import read_image
import os
import pandas as pd
import matplotlib.pyplot as plt


class MNISTDataset(Dataset):
    """
    Fashion MNIST Dataset Class
    """
    def __init__(self, path, download=True, trans_func=ToTensor(), label_func=Lambda(lambda x: torch.zeros(10, dtype=torch.float).scatter_(0,torch.tensor(x), value=1))):
        # 1. download the dataset
        self.train_data = datasets.FashionMNIST(
            root = path,
            train = True,
            download = download,
            transform = trans_func,
            target_transform = label_func
        )
        self.test_data = datasets.FashionMNIST(
            root = path,
            train = False,
            download = download,
            transform = trans_func,
            target_transform=label_func
        )
        self.transform = trans_func
        self.label_transform = label_func

        # 2. load raw image data and annotations
        self.data_path = os.path.join(path, "FashionMNIST/raw/")
        self.labels = pd.read_csv(os.path.join(self.data_path, "train-labels-idx1-ubyte"), encoding='latin-1')

        # deal with the dataset
        self.labels_map = {
            0: "T-Shirt",
            1: "Trouser",
            2: "Pullover",
            3: "Dress",
            4: "Coat",
            5: "Sandal",
            6: "Shirt",
            7: "Sneaker",
            8: "Bag",
            9: "Ankle Boot",
        }

    def __len__(self):
        """
        : return length the dataset
        """
        return len(self.labels)

    def __getitem__(self, item):
        """
        brief: to get one item from the dataset
        """
        img_name, img_label = self.labels.iloc[item]
        img_path = os.path.join(self.img_path, img_name)
        img_data = read_image(img_path)
        if self.transform:
            img_data = self.transform(img_data)
        if self.label_transform:
            img_label = self.label_transform(img_label)
        return img_data, img_label

    def getTrainData(self):
        return self.train_data

    def getTestData(self):
        return self.test_data

    def randomShowImages(self, n):
        '''
        Randomly pick images and print them out
        :param n: n-by-n layouts
        :return
        '''
        figure = plt.figure(figsize=(8, 8))
        cols, rows = n, n
        for i in range(1, cols * rows + 1):
            sample_idx = torch.randint(len(self.train_data), size=(1,)).item()
            img, label = self.train_data[sample_idx]
            figure.add_subplot(rows, cols, i)
            plt.title(self.labels_map[list(label).index(1)])
            plt.axis("off")
            plt.imshow(img.squeeze(), cmap="gray")
        plt.show()