#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 12:58:24 2024

@author: sunjim
"""

"""


Explanation:
Data Preparation:

Load the stock price data and normalize it using MinMaxScaler.
Create a dataset where each input vector contains 30 days of stock prices (excluding the date) and the output is the next day's opening and closing prices.
Model Building:

A Sequential model is created with a SimpleRNN layer followed by a Dense layer.
The RNN layer has 50 units and takes input with shape (look_back, 5), where look_back is 30 days and 5 features (Volume, Open, Close, High, Low).
Model Training:

The model is compiled with the Adam optimizer and mean squared error loss function.
The model is trained for 50 epochs with a batch size of 16.
Making Predictions:

Predictions are made on the test set and are inverse transformed to get the actual stock prices.
Comparison:

The predicted opening and closing prices are compared with the actual opening and closing prices.

"""

import io
import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU, Bidirectional
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
import joblib
from datetime import datetime, timedelta


trained_best_file = "maotai/maotai_20250808_best.h5"
train_ratio = 0.99
trained_scaler_file = "maotai/maotai_20250808_scaler.save"
train_csv_path = 'maotai/600519_20250808.csv'
predict_column = 0
shuffle_data = False

def _main(retrain=False, load_trained=False):
    look_back = 35
    hidden_layer = 95

    X, y = prepare_train_data(train_csv_path, look_back)
    if retrain:
        if load_trained:
            train_by_data(X, y, hidden_layer, look_back, trained_best_file)
        else:
            train_by_data(X, y, hidden_layer, look_back)
    
    # pure test 
    train_size = int(len(X) * train_ratio)
    X_test = X[train_size:][-10:]
    y_test = y[train_size:][-10:]
    
    do_predict(X_test, y_test, trained_best_file, hidden_layer, look_back)
    do_real_predict('maotai/stock_real_price.csv', trained_best_file, hidden_layer, look_back)
    
#    do_predict(X_test, y_test, 'train_result/trained_best_in_val.h5')
#    do_real_predict('real_latest_stock_price.csv', 'train_result/trained_best_in_val.h5')

def choose_look_back():
    for look_back in range(5, 100, 5):
        X, y = prepare_train_data(train_csv_path, look_back)
        train_by_data(X, y, look_back)

def choose_model():
    result = {}
    best_loss = float('inf')
    best_params = None

    for look_back in range(5, 100, 5):
        X, y = prepare_train_data(train_csv_path, look_back)
        for units in range(5, 100, 5):
            print(f"hidden_units={units}")
            ret = train_by_data(X, y, units, look_back)
            result[str(look_back)+':'+str(units)]=ret

            # Track the best performing combination
            current_val_loss = ret
            if current_val_loss < best_loss:
                best_loss = current_val_loss
                best_params = (look_back, units)
                print(f"🎯 New best! look_back={look_back}, hidden_units={units}, val_loss={current_val_loss:.6f}")

    # Print all results
    print("\n" + "="*60)
    print("ALL RESULTS:")
    print("="*60)   
    for hidden_unit in result:
        print ('hidden ', hidden_unit,':', f"loss:{result[hidden_unit]}, val_loss:{result[hidden_unit]}")
    
    # Print the best combination
    print("\n" + "="*60)
    print("BEST COMBINATION:")
    print("="*60)
    if best_params:
        best_look_back, best_hidden_units = best_params
        print(f"Best look_back: {best_look_back}")
        print(f"Best hidden_units: {best_hidden_units}")
        print(f"Best validation loss: {best_loss:.6f}")

def prepare_train_data(data_path, look_back):
    # Load your data
    # Assume `data` is a DataFrame with columns: 'Date', 'Volume', 'Open', 'Close', 'High', 'Low'
    data = pd.read_csv(data_path)

    data = uniform_data(data)

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    joblib.dump(scaler, trained_scaler_file) 

    # Add 'Date' back to the scaled data
#    scaled_data = np.concatenate((dates.values.reshape(-1, 1), scaled_data), axis=1)
    
    return create_dataset(scaled_data, look_back)



# Prepare the dataset
def create_dataset(data, look_back=30):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i + look_back])  # Skip 'Date' column for input features
        y.append(data[i + look_back, predict_column])  # Next day's swing percentage 
    return np.array(X), np.array(y)


def train_by_data(X, y, hidden_unit=50, look_back=25, saved_weights=''):
    # Shuffle the data before splitting
    if shuffle_data:
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        # Split the data into training and testing sets
        train_size = int(len(X_shuffled) * train_ratio)
        X_train, X_test = X_shuffled[:train_size], X_shuffled[train_size:]
        y_train, y_test = y_shuffled[:train_size], y_shuffled[train_size:]
    else:
        train_size = int(len(X) * train_ratio)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
    
    logging = TensorBoard()
    checkpoint = ModelCheckpoint(trained_best_file, monitor='val_loss',
                                 save_weights_only=True, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=55, verbose=1, mode='auto')
    
    model = create_model(hidden_unit, look_back, X_train.shape[-1])   
    if os.path.isfile(saved_weights):
        model.load_weights(saved_weights)
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    for cnt in range(1):
        # Train the model
        history = model.fit(X_train, y_train, epochs=512, batch_size=64, validation_data=(X_test, y_test),
                  callbacks=[logging, checkpoint, early_stopping])
        # Get the best validation loss
        best_val_loss = min(history.history['val_loss'])
        best_val_loss_epoch = history.history['val_loss'].index(best_val_loss) + 1

        print(f"Final loss: {history.history['loss'][-1]:.6f}, Final val_loss: {history.history['val_loss'][-1]:.6f}")
        print(f"Best val_loss: {best_val_loss:.6f} at epoch {best_val_loss_epoch}")

        return best_val_loss
    

def create_model(hidden_units=50, look_back=25, input_layer=5):
    # Build the RNN model
    model = Sequential()
    model.add(LSTM(hidden_units, input_shape=(look_back, input_layer), return_sequences=False))  # 5 features: Volume, Open, Close, High, Low
    model.add(Dense(1))  # Predicting 2 values: next day Open and Close prices
    return model

def uniform_data(data):
    # Convert Date column to datetime and extract weekday
    data['Date'] = pd.to_datetime(data['Date'])
    data['Weekday'] = data['Date'].dt.dayofweek  # 0=Monday, 6=Sunday
    
    data = data.drop(columns=['Date'])
    data = data.drop(columns=['code'])
    return data

def do_real_predict(real_data_path, saved_weights, hidden_units=50, look_back=25):
    
    scaler = joblib.load(trained_scaler_file)
    # Load your data
    # Assume `data` is a DataFrame with columns: 'Date', 'Volume', 'Open', 'Close', 'High', 'Low'
    data = pd.read_csv(real_data_path)

    # Drop the 'Date' column for normalization and later use it for features
    data = data[-look_back:]
    print(data)
    dates = data['Date']
    last_day = datetime.strptime(np.array(dates)[-1] , '%m/%d/%y') + timedelta(1)

    data = uniform_data(data)
    scaled_data = scaler.transform(data)
    
    # Add 'Date' back to the scaled data
    #scaled_data = np.concatenate((dates.values.reshape(-1, 1), scaled_data), axis=1)
    
    # Make predictions
    model = create_model(hidden_units, look_back, scaled_data.shape[-1])

    # Build the model first
    dummy_input = np.zeros((1, look_back, scaled_data.shape[-1]))
    model(dummy_input)

    model.load_weights(saved_weights)
    predictions = model.predict(np.array([scaled_data]))
    
    # Inverse transform the predictions to get actual values
    #predicted_prices = scaler.inverse_transform(np.concatenate((np.zeros((predictions.shape[0], 1)), predictions, np.zeros((predictions.shape[0], (data.shape[-1]-3)))), axis=1))[:, [1, 2]]
    predicted_prices = inverse_transform_predictions(predictions, scaler)

    print(f"Predicted {last_day.strftime('%m/%d/%y')} open: {round(predicted_prices[0], 2)}")
    
def inverse_transform_predictions(predictions, scaler, original_features=None):
    # Get the actual number of features the scaler was trained on
    if original_features is None:
        original_features = scaler.n_features_in_

    # Create a full feature array with zeros for non-predicted features
    full_predictions = np.zeros((predictions.shape[0], original_features))
    full_predictions[:, predict_column] = predictions.flatten()  # Put predictions in Open/Close positions
    return scaler.inverse_transform(full_predictions)[:, predict_column]  # Return only Open/Close


def do_predict(X_test, y_test, saved_weights, hidden_units=50, look_back=25):
    model = create_model(hidden_units, look_back, X_test.shape[-1])

    # Build the model first
    dummy_input = np.zeros((1, look_back, X_test.shape[-1]))
    model(dummy_input)

    model.load_weights(saved_weights)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    scaler = joblib.load(trained_scaler_file)
    
    # Inverse transform the predictions to get actual values
    predicted_prices = inverse_transform_predictions(predictions, scaler)

    # Inverse transform the actual values for comparison
    #actual_prices = scaler.inverse_transform(np.concatenate((np.zeros((y_test.shape[0], 1)), y_test, np.zeros((y_test.shape[0], (X_test.shape[-1]-3)))), axis=1))[:, [1, 2]]
    actual_prices = inverse_transform_predictions(y_test, scaler)

    # Print the results
    for i in range(len(predicted_prices)):
        print(f"Predicted open: {round(predicted_prices[i], 2)}, Actual open: {round(actual_prices[i], 2)} ({round(predicted_prices[i] - actual_prices[i], 2)})")  

if __name__ == "__main__":
    _main(retrain=True)
    #choose_model()