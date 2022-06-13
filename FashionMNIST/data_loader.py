import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.transforms import ToTensor
import os
import pandas

# download a mnist dataset
def downloadDataset():
    train_data = datasets.FashionMNIST(
        root="./FashionMNIST",
        train=True,
        download=True,
        transform=ToTensor()
    )

    test_data = datasets.FashionMNIST(
        root="./FashionMNIST",
        trian=False,
        download=True,
        transform=ToTensor()
    )
    return train_data, test_data

