import torch

def train_loop(dataloader, model, loss_fn, optimizer):
    # step 1: infer
    size = len(dataloader.dataset)
    all_loss = []
    for batch, (X, Y) in enumerate(dataloader):
        # infer
        pred = model(X)
        loss = loss_fn(pred, Y)

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    # step 1
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # step 2 evaluate all the losses
    with torch.no_grad():
        for X, Y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, Y).item()
            correct += (pred.argmax(1) == Y.argmax(1)).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")