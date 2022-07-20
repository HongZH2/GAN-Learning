"""
A demo for Neural Network Learning
by Hong Zhang
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_loader import MNISTDataset
from mNetwork import MNeuralNetwork
from train import train_loop, test_loop
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    # check if gpu is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # step 1. initialize the dataset

    dataset = MNISTDataset('./', download=False)
    # step 2. show some samples from Dataset
    # dataset.randomShowImages(5)

    # step 3. batch the dataset and set some params
    batch_size = 16
    learning_rate = 1e-3
    epochs = 20

    train_loader = DataLoader(dataset.getTrainData(), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset.getTestData(), batch_size=batch_size, shuffle=True)

    # step 4. initialize my network
    mnetwork = MNeuralNetwork().to(device)
    print(mnetwork)

    # step 5.  set the optimizer
    # optimizer = torch.optim.SGD(mnetwork.parameters(), lr=learning_rate, momentum=0.9)
    optimizer = torch.optim.Adam(mnetwork.parameters(), lr=learning_rate) # 差距不大
    # step 6. loss function
    loss_func = nn.CrossEntropyLoss()

    # step 7. start training
    for i in range(1, epochs):
        print(f"Epoch {i}\n-------------------------------")
        train_loop(train_loader, mnetwork, loss_func, optimizer)
        test_loop(test_loader, mnetwork, loss_func)
    print("Done!")

    # step 8. save the model
    torch.save(mnetwork, 'mnn_results_Adam_BS16.pth')