"""
For evaluating a model
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf


def test_model(model, test_set, device, ticker):
    # Step 1: Fetch historical stock price data using Yahoo Finance
    data = yf.download(ticker, start="2023-8-01", end="2025-04-01")
    closing_prices = data['Close'].values.reshape(-1, 1)

    # Step 3: Normalize the closing prices to scale between 0 and 1
    scaler = MinMaxScaler()
    scaled_prices = scaler.fit_transform(closing_prices)

    
    # Loading data
    model.to(device)

    y_test, X_test = test_set.tensors

    X_test = X_test.to(device)


    # Step 8: Make predictions on the test dataset
    predictions = model.predict(X_test)

    predictions = predictions.cpu().numpy()
    y_test = y_test.cpu().numpy()

    # Step 9: Inverse transform the predictions and true labels to get actual price values
    predictions = scaler.inverse_transform(predictions)
    y_test = scaler.inverse_transform(y_test)

    # Step 10: Evaluate model performance using MAE and MAPE metrics
    mae = np.mean(np.abs(predictions - y_test))
    mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")