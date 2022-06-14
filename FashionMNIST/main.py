"""
A demo for Neural Network Learning
by Hong Zhang
"""
import os
from data_loader import MNISTDataset


if __name__ == '__main__':
    # step 1. initialize the dataset
    dataset = MNISTDataset('./', download = False)
    # step 2. show some samples from Dataset
    dataset.randomShowImages(3)