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
from tensorflow.keras.layers import Dense, SimpleRNN
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
import joblib
from datetime import datetime, timedelta


trained_best_file = "trained_best_one_day.h5"
train_ratio = 0.9
trained_scaler_file = "one_day_scaler.save"

def _main(retrain=False, load_trained=False):
    look_back = 55
    hidden_layer = 25
    train_csv_path = 'adsk_stock_prices_train.csv'
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
    do_real_predict('adsk_stock_real_price.csv', trained_best_file, hidden_layer, look_back)
    
#    do_predict(X_test, y_test, 'train_result/trained_best_in_val.h5')
#    do_real_predict('real_latest_stock_price.csv', 'train_result/trained_best_in_val.h5')

def choose_look_back():
    for look_back in range(5, 100, 5):
        train_csv_path = 'adsk_stock_prices.csv'
        X, y = prepare_train_data(train_csv_path, look_back)
        train_by_data(X, y, look_back)

def choose_model():
    result = {}
    
    for look_back in range(5, 100, 5):
        train_csv_path = 'adsk_stock_prices.csv'
        X, y = prepare_train_data(train_csv_path, look_back)
        for units in range(5, 100, 5):
            print(f"hidden_units={units}")
            ret = train_by_data(X, y, units, look_back)
            result[str(look_back)+':'+str(units)]=ret
    
    for hidden_unit in result:
   #     print (result[hidden_unit])
        print ('hidden ', hidden_unit,':', f"loss:{result[hidden_unit]['loss'][-1]}, val_loss:{result[hidden_unit]['val_loss'][-1]}")

def prepare_train_data(data_path, look_back):
    # Load your data
    # Assume `data` is a DataFrame with columns: 'Date', 'Volume', 'Open', 'Close', 'High', 'Low'
    data = pd.read_csv(data_path)

    # Drop the 'Date' column for normalization and later use it for features
#    dates = data['Date']
    data = data.drop(columns=['Date'])

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
        y.append(data[i + look_back, [1, 2]])  # Next day's opening and closing prices
    return np.array(X), np.array(y)


def train_by_data(X, y, hidden_unit=50, look_back=25, saved_weights=''):
    # Split the data into training and testing sets
    train_size = int(len(X) * train_ratio)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    logging = TensorBoard()
    checkpoint = ModelCheckpoint(trained_best_file, monitor='val_loss',
                                 save_weights_only=True, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=55, verbose=1, mode='auto')
    
    # my track
    best_loss_file = 'train_result/test_loss_best.npy'
    best_loss = {'train_loss': 10000000000.0, 'val_loss': 10000000000.0}
    if os.path.isfile(best_loss_file):
        best_loss = np.load(best_loss_file, allow_pickle='TRUE').item()
    
    model = create_model(hidden_unit, look_back, X_train.shape[-1])   
    if os.path.isfile(saved_weights):
        model.load_weights(saved_weights)
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    for cnt in range(1):
        # Train the model
        history = model.fit(X_train, y_train, epochs=255, batch_size=32, validation_data=(X_test, y_test),
                  callbacks=[logging, checkpoint, early_stopping])
        print(f"history loss:{history.history['loss'][-1]}, val_loss:{history.history['val_loss'][-1]}")
        return history.history
    

def create_model(hidden_units=50, look_back=25, input_layer=5):
    # Build the RNN model
    model = Sequential()
    model.add(SimpleRNN(hidden_units, input_shape=(look_back, input_layer), return_sequences=False))  # 5 features: Volume, Open, Close, High, Low
    model.add(Dense(2))  # Predicting 2 values: next day Open and Close prices
    return model


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
    
    data = data.drop(columns=['Date'])
    scaled_data = scaler.transform(data)
    
    # Add 'Date' back to the scaled data
    #scaled_data = np.concatenate((dates.values.reshape(-1, 1), scaled_data), axis=1)
    
    # Make predictions
    model = create_model(hidden_units, look_back, scaled_data.shape[-1])
    model.load_weights(saved_weights)
    predictions = model.predict(np.array([scaled_data]))
    
    # Inverse transform the predictions to get actual values
    predicted_prices = scaler.inverse_transform(np.concatenate((np.zeros((predictions.shape[0], 1)), predictions, np.zeros((predictions.shape[0], (data.shape[-1]-3)))), axis=1))[:, [1, 2]]
    
    print(f"Predicted {last_day.strftime('%m/%d/%y')} Open: ${round(predicted_prices[0, 0], 2)}")
    print(f"Predicted {last_day.strftime('%m/%d/%y')} Close: ${round(predicted_prices[0, 1], 2)}")
    

def do_predict(X_test, y_test, saved_weights, hidden_units=50, look_back=25):
    model = create_model(hidden_units, look_back, X_test.shape[-1])
    model.load_weights(saved_weights)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    scaler = joblib.load(trained_scaler_file)
    
    # Inverse transform the predictions to get actual values
    predicted_prices = scaler.inverse_transform(np.concatenate((np.zeros((predictions.shape[0], 1)), predictions, np.zeros((predictions.shape[0], (X_test.shape[-1]-3)))), axis=1))[:, [1, 2]]
    
    # Inverse transform the actual values for comparison
    actual_prices = scaler.inverse_transform(np.concatenate((np.zeros((y_test.shape[0], 1)), y_test, np.zeros((y_test.shape[0], (X_test.shape[-1]-3)))), axis=1))[:, [1, 2]]
    
    # Print the results
    for i in range(len(predicted_prices)):
        print(f"Predicted Open: {round(predicted_prices[i, 0], 2)}, Actual Open: {actual_prices[i, 0]} ({round(predicted_prices[i, 0] - actual_prices[i, 0], 2)})")
  #      print(f"Predicted Close: {round(predicted_prices[i, 1], 2)}, Actual Close: {actual_prices[i, 1]} ({round(predicted_prices[i, 1] - actual_prices[i, 1], 2)})")