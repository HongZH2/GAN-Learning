
import os
from data_loader import downloadDataset


if __name__ == '__main__':
    train_data, test_data = downloadDataset()
    print(len(train_data))