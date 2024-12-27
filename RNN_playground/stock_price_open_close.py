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


trained_best_file = "trained_best.h5"
train_ratio = 0.9
next_n_days = 2

def _main(retrain=False):
    look_back = 30
    train_csv_path = 'adsk_stock_prices.csv'
    X, y = prepare_train_data(train_csv_path, look_back)
    train_by_data(X, y, look_back, trained_best_file)
    
    # pure test 
    train_size = int(len(X) * train_ratio)
#    X_test = X[train_size:][-10:]
#    y_test = y[train_size:][-10:]
    X_test = X[train_size:][-1:]
    y_test = y[train_size:][-1:]
    do_predict_test(X_test, y_test, look_back, trained_best_file)
    
    do_real_predict('real_latest_stock_price.csv', look_back, trained_best_file)
    

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
    joblib.dump(scaler, 'scaler.save') 

    # Add 'Date' back to the scaled data
#    scaled_data = np.concatenate((dates.values.reshape(-1, 1), scaled_data), axis=1)
    
    return create_dataset(scaled_data, look_back)



# Prepare the dataset
def create_dataset(data, look_back=30):
    X, y = [], []
    for i in range(len(data) - look_back - next_n_days):
        X.append(data[i:i + look_back])  # Skip 'Date' column for input features
        y.append(data[i + look_back:i + look_back + next_n_days, [1, 2]])  # Next day's opening and closing prices
    return np.array(X), np.array(y).reshape(-1, 2*next_n_days)


def train_by_data(X, y, look_back, saved_weights=''):
    # Split the data into training and testing sets
    train_size = int(len(X) * train_ratio)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    logging = TensorBoard()
    checkpoint = ModelCheckpoint(trained_best_file, monitor='val_loss',
                                 save_weights_only=True, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=1, mode='auto')
    
    # my track
    best_loss_file = 'train_result/test_loss_best.npy'
    best_loss = {'train_loss': 10000000000.0, 'val_loss': 10000000000.0}
    if os.path.isfile(best_loss_file):
        best_loss = np.load(best_loss_file, allow_pickle='TRUE').item()
    
    model = create_model(look_back)   
    if os.path.isfile(saved_weights):
        model.load_weights(saved_weights)
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    for cnt in range(1):
        # Train the model
        history = model.fit(X_train, y_train, epochs=1, batch_size=16, validation_data=(X_test, y_test),
                  callbacks=[logging, checkpoint, early_stopping])
        # save best
        if best_loss['train_loss'] > history.history['loss'][0]:
             best_loss['train_loss'] = history.history['loss'][0]
             model.save_weights('train_result/trained_overfit.h5')
             np.save(best_loss_file, best_loss)
             
        if best_loss['val_loss'] > history.history['val_loss'][0]:
             print(f"Best look_back={look_back}")
             best_loss['val_loss'] = history.history['val_loss'][0]
             model.save_weights('train_result/trained_best_in_val.h5')
             np.save(best_loss_file, best_loss)
    

def create_model(look_back):
    # Build the RNN model
    model = Sequential()
    model.add(SimpleRNN(50, input_shape=(look_back, 5), return_sequences=False))  # 5 features: Volume, Open, Close, High, Low
    model.add(Dense(2*next_n_days))  # Predicting 2 values: next day Open and Close prices
    return model


def do_real_predict(real_data_path, look_back, saved_weights):
    model = create_model(look_back)
    model.load_weights(saved_weights)
    
    scaler = joblib.load('scaler.save')
    # Load your data
    # Assume `data` is a DataFrame with columns: 'Date', 'Volume', 'Open', 'Close', 'High', 'Low'
    data = pd.read_csv(real_data_path)

    # Drop the 'Date' column for normalization and later use it for features
    data = data[-look_back:]
    print(data)
    dates = data['Date']
    data = data.drop(columns=['Date'])
    scaled_data = scaler.transform(data)
    
    # Add 'Date' back to the scaled data
    #scaled_data = np.concatenate((dates.values.reshape(-1, 1), scaled_data), axis=1)
    
    # Make predictions
    predictions = model.predict(np.array([scaled_data]))
    
    # Inverse transform the predictions to get actual values
    predicted_prices = convert_readable_predict(predictions)
    
    # Print the results
    for i in range(next_n_days):
        print(f"Predicted day{i + 1}: Open: {round(predicted_prices[i, 0], 2)}, Close:{round(predicted_prices[i, 1], 2)}") 
    
def convert_readable_predict(predictions):
    print("before: {predictions}")
    print(predictions)
    print(predictions.shape)
    predictions = predictions.reshape(next_n_days, 2)
    
    print("after: {predictions.reshape(2, 2)}")
    print(predictions)
    print(predictions.shape)
    
    scaler = joblib.load('scaler.save')
    
    # Inverse transform the predictions to get actual values
    predicted_prices = scaler.inverse_transform(np.concatenate((np.zeros((predictions.shape[0], 1)), predictions, np.zeros((predictions.shape[0], 2))), axis=1))[:, [1, 2]]

    return predicted_prices
    

def do_predict_test(X_test, y_test, look_back, saved_weights):
    model = create_model(look_back)
    model.load_weights(saved_weights)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Inverse transform the predictions to get actual values
    predicted_prices = convert_readable_predict(predictions)
    
    # Inverse transform the actual values for comparison
    actual_prices = convert_readable_predict(y_test)
    
    # Inverse transform the predictions to get actual values
    #predicted_prices = scaler.inverse_transform(np.concatenate((np.zeros((predictions.shape[0], 1)), predictions, np.zeros((predictions.shape[0], 2))), axis=1))[:, [1, 2]]
    
    # Inverse transform the actual values for comparison
    #actual_prices = scaler.inverse_transform(np.concatenate((np.zeros((y_test.shape[0], 1)), y_test, np.zeros((y_test.shape[0], 2))), axis=1))[:, [1, 2]]
    
    # Print the results
    for i in range(next_n_days):
        print(f"Predicted day{i + 1}: Open: {round(predicted_prices[i, 0], 2)}, Close:{round(predicted_prices[i, 1], 2)}") 
        print(f"Actual day{i + 1}:    Open: {actual_prices[i, 0]}, Close: {actual_prices[i, 1]}")