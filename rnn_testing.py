"""
For evaluating a model
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from preprocessing import get_scaler_and_prices


def test_model(model, test_set, device, ticker):

    scaler, _ = get_scaler_and_prices(ticker)

    
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
    return mae, mape

def test_session(session, test_set, ticker):
    scaler, _ = get_scaler_and_prices(ticker)

    y_test, X_test = test_set.tensors

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    predictions = session.run([output_name], {input_name: X_test})[0]
    predictions = predictions.numpy()
    y_test = y_test.numpy()

    predictions = scaler.inverse_transform(predictions)
    y_test = scaler.inverse_transform(y_test)

    mae = np.mean(np.abs(predictions - y_test))
    mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    return mae, mape

