import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Step 1: Fetch historical stock price data using Yahoo Finance
ticker = "AAPL" # Apple Inc.
data = yf.download(ticker, start="2024-04-01", end="2025-04-01")
closing_prices = data['Close'].values.reshape(-1, 1)

# Step 2: Analyze sentiment scores from financial news (placeholder text used here)
analyzer = SentimentIntensityAnalyzer()
news_data = ["Sample news article text"] # Replace with real-time news from a financial API
sentiments = [analyzer.polarity_scores(text)['compound'] for text in news_data]
sentiments = np.array(sentiments).reshape(-1, 1)

# Step 3: Normalize the closing prices and sentiment scores to scale between 0 and 1
scaler = MinMaxScaler()
scaled_prices = scaler.fit_transform(closing_prices)
scaled_sentiments = scaler.fit_transform(sentiments)

# Step 4: Prepare feature sequences using a sliding window approach
# Each input consists of 60 consecutive days of stock prices and sentiment scores
window_size = 60
X, y = [], []
for i in range(window_size, len(scaled_prices)):
    X.append(np.column_stack((scaled_prices[i-window_size:i, 0],
                              scaled_sentiments[i-window_size:i, 0])))
    y.append(scaled_prices[i, 0])
X, y = np.array(X), np.array(y)

# Step 5: Split the data into training and testing sets (80-20 ratio)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Step 6: Define the LSTM model architecture
model = Sequential()
model.add(LSTM(64, return_sequences=True,
               input_shape=(window_size, 2))) # Two features: price and sentiment
model.add(Dropout(0.2)) # Regularization to prevent overfitting
model.add(LSTM(32)) # Second LSTM layer
model.add(Dense(1)) # Output layer with a single neuron for regression
model.compile(optimizer='adam', loss='mse')

# Step 7: Train the model on the training dataset
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

# Step 8: Make predictions on the test dataset
predictions = model.predict(X_test)

# Step 9: Inverse transform the predictions and true labels to get actual price values
predictions = scaler.inverse_transform(predictions)
y_test = scaler.inverse_transform([y_test])

# Step 10: Evaluate model performance using MAE and MAPE metrics
mae = np.mean(np.abs(predictions - y_test))
mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")