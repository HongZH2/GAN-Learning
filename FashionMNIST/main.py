"""
A demo for Neural Network Learning
by Hong Zhang
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_loader import MNISTDataset
from mNetwork import MNeuralNetwork


# define training loop and test loop
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# check if gpu is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# step 1. initialize the dataset

dataset = MNISTDataset('./', download=False)
# step 2. show some samples from Dataset
# dataset.randomShowImages(5)


# step 3. batch the dataset and set some params
batch_size = 64
learning_rate = 1e-3
epochs = 10

train_loader = DataLoader(dataset.getTrainData(), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset.getTestData(), batch_size=batch_size, shuffle=True)

# step 4. initialize my network
mnetwork = MNeuralNetwork().to(device)
print(mnetwork)

# step 5.  set the optimizer
optimizer = torch.optim.SGD(mnetwork.parameters(), lr=learning_rate)

# step 6. loss function
loss_func = nn.CrossEntropyLoss()



# step 7. start training
for i in range(1, epochs):
    print(f"Epoch {i}\n-------------------------------")
    train_loop(train_loader, mnetwork, loss_func, optimizer)
    test_loop(test_loader, mnetwork, loss_func)
print("Done!")
