import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def save(ticker):
    """
    Fetch the data for the given ticker (about 300 samples should be sufficient)
    """
    # Step 1: Fetch historical stock price data using Yahoo Finance
    data = yf.download(ticker, start="2024-04-01", end="2026-01-01")
    closing_prices = data['Close'].values.reshape(-1, 1)

    # Step 3: Normalize the closing prices to scale between 0 and 1
    scaler = MinMaxScaler()
    scaled_prices = scaler.fit_transform(closing_prices)

    # Step 4: Prepare feature sequences using a sliding window approach
    # Each input consists of 50 consecutive days of stock prices
    window_size = 50
    X, y = [], []
    for i in range(window_size, len(scaled_prices)):
        X.append(scaled_prices[i-window_size:i, 0])
        y.append(scaled_prices[i, 0])
    X, y = np.array(X), np.array(y)

    # Step 5: Split the data into training and testing sets (80-20 ratio)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=123 
    )

    print(np.shape(X_train))
    print(np.shape(y_train))

    # Step 6: Save data
    np.savez_compressed(
        f'data/{ticker}.npz',
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test
    )


if __name__ == "__main__":
    stocks = ["AAPL", "MSFT", "TSLA", "NDAQ"]
    for s in stocks:
        save(s)