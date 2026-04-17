"""
For loading in the reviews and sentiments from csv files, or from npz files
"""
import numpy as np
import torch
from torch.utils.data import TensorDataset


def load_AAPL():
    return npz_load("AAPL")

def load_MSFT():
    return npz_load("MSFT")

def load_TSLA():
    return npz_load("TSLA")

def load_NDAQ():
    return npz_load("NDAQ")

def npz_load(ticker):
    """
    Known Tickers:
    stocks = ["AAPL", "MSFT", "TSLA", "NDAQ"]
    """

    data = np.load(f'data/{ticker}.npz')


    x_train = data['X_train']
    y_train = data['y_train']
    x_test = data['X_test']
    y_test = data['y_test']

    # To tensors
    x_train = torch.from_numpy(x_train).float().unsqueeze(-1)  # this makes it like (N, 50, 1)
    x_test  = torch.from_numpy(x_test).float().unsqueeze(-1)
    y_train = torch.from_numpy(y_train).float().unsqueeze(1)
    y_test  = torch.from_numpy(y_test).float().unsqueeze(1)

    train_set = TensorDataset(y_train, x_train)
    test_set = TensorDataset(y_test, x_test)

    return train_set, test_set

if __name__ == "__main__":
    train_set, test_set = load_AAPL()
    label, x = train_set[0]

    print(label.shape)
    print(x.shape)

    label, x = test_set[0]

    print(label.shape)
    print(x.shape)
    