import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# Step 1: Fetch historical stock price data using Yahoo Finance
ticker = "AAPL" # Apple Inc.
# data = yf.download(ticker, start="2024-04-01" , end="2025-04-01")
data = yf.download(ticker, start="2025-03-01" , end="2025-04-01", interval="60m")
closing_prices = data['Close'].values.reshape(-1, 1)

# Step 3: Normalize the closing prices and sentiment scores to scale between 
# 0 and 1
scaler = MinMaxScaler()
scaled_prices = scaler.fit_transform(closing_prices.reshape(-1, 1)).flatten()

# Step 4: Prepare feature sequences using a sliding window approach
# Each input consists of 50 consecutive time intervals of stock prices, plus an 
# additional time interval as label
window_size = 50
X, y = [], []
for i in range(window_size, len(scaled_prices) - 1):
    X.append(scaled_prices[i-window_size:i])
    y.append(scaled_prices[i + 1])

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
