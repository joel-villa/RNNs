"""
For loading in the reviews and sentiments from csv files, or from npz files
"""
import numpy as np
import torch

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
    x_train = torch.from_numpy(x_train).float()
    y_train = torch.from_numpy(y_train).float().unsqueeze(1) # (d,) -> (d, 1)
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).float().unsqueeze(1)

    return x_train, y_train, x_test, y_test

if __name__ == "__main__":
    x_tr, y_tr, x_te, y_te = load_AAPL()
    print(f"x_tr shape: {np.shape(x_tr)}")
    print(f"y_tr shape: {np.shape(y_tr)}")
    print(f"x_te shape: {np.shape(x_te)}")
    print(f"y_te shape: {np.shape(y_te)}")
    x_tr, y_tr, x_te, y_te = load_MSFT()
    print(f"x_tr shape: {np.shape(x_tr)}")
    print(f"y_tr shape: {np.shape(y_tr)}")
    print(f"x_te shape: {np.shape(x_te)}")
    print(f"y_te shape: {np.shape(y_te)}")
    x_tr, y_tr, x_te, y_te = load_TSLA()
    print(f"x_tr shape: {np.shape(x_tr)}")
    print(f"y_tr shape: {np.shape(y_tr)}")
    print(f"x_te shape: {np.shape(x_te)}")
    print(f"y_te shape: {np.shape(y_te)}")
    x_tr, y_tr, x_te, y_te = load_NDAQ()
    print(f"x_tr shape: {np.shape(x_tr)}")
    print(f"y_tr shape: {np.shape(y_tr)}")
    print(f"x_te shape: {np.shape(x_te)}")
    print(f"y_te shape: {np.shape(y_te)}")