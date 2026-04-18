"""
For evaluating a model
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler


def model_performance(model, y_test, X_test):
    # Step 3: Normalize the closing prices to scale between 0 and 1
    scaler = MinMaxScaler()

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