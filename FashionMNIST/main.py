"""
A demo for Neural Network Learning
by Hong Zhang
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_loader import MNISTDataset
from mNetwork import MNeuralNetwork

if __name__ == '__main__':

    # check if gpu is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # step 1. initialize the dataset
    dataset = MNISTDataset('./', download=False)
    # step 2. show some samples from Dataset
    # dataset.randomShowImages(5)


    # step 3. batch the dataset
    batch_size = 64
    train_loader = DataLoader(dataset.getTrainData(), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset.getTestData(), batch_size=batch_size, shuffle=True)

    # step 4. initialize my network
    mnetwork = MNeuralNetwork().to(device)
    print(mnetwork)

    X = torch.rand(1, 28, 28, device=device)
    logits = mnetwork(X)
    pred_probab = nn.Softmax(dim=1)(logits)
    y_pred = pred_probab.argmax(1)
    print(f"Predicted class: {y_pred}")