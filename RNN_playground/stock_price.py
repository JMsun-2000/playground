#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 10:10:43 2024

@author: sunjim
"""

"""
Question:
use python to code a RNN to deal with one stock next day open price predict. The input vector is daily opening price, closing price, highest price and lowest price. Recurrent latest 30 days. output is the next day opening price
Certainly! Here's an example of how you can implement a Recurrent Neural Network (RNN) using Python and TensorFlow/Keras to predict the next day's opening price of a stock based on the past 30 days of opening, closing, highest, and lowest prices.

Anwser:
First, you need to install TensorFlow if you haven't already:

pip install tensorflow
Next, you can use the following code to create and train the RNN model:


Explanation:
Data Preparation:

Load the stock price data and normalize it using MinMaxScaler.
Create a dataset where each input vector contains 30 days of stock prices, and the output is the next day's opening price.
Model Building:

A Sequential model is created with a SimpleRNN layer followed by a Dense layer.
The RNN layer has 50 units and takes input with shape (look_back, 4), where look_back is 30 days and 4 features (Open, Close, High, Low).
Model Training:

The model is compiled with the Adam optimizer and mean squared error loss function.
The model is trained for 50 epochs with a batch size of 16.
Making Predictions:

Predictions are made on the test set and are inverse transformed to get the actual stock prices.
Comparison:

The predicted opening prices are compared with the actual opening prices.
This is a basic implementation. In practice, you may need to fine-tune the model architecture, hyperparameters, and data preprocessing steps based on your specific dataset and requirements.

"""

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
from sklearn.preprocessing import MinMaxScaler

# Load your data
# Assume `data` is a DataFrame with columns: 'Open', 'Close', 'High', 'Low'
data = pd.read_csv('stock_prices.csv')

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[['Open', 'Close', 'High', 'Low']])

# Prepare the dataset
def create_dataset(data, look_back=30):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i + look_back])
        y.append(data[i + look_back][0])  # Next day's opening price
    return np.array(X), np.array(y)

look_back = 30
X, y = create_dataset(scaled_data, look_back)

# Split the data into training and testing sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build the RNN model
model = Sequential()
model.add(SimpleRNN(50, input_shape=(look_back, 4), return_sequences=False))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

# Make predictions
predictions = model.predict(X_test)

# Inverse transform the predictions to get actual values
predicted_open_prices = scaler.inverse_transform(np.concatenate((predictions, np.zeros((predictions.shape[0], 3))), axis=1))[:, 0]

# Compare with actual values
actual_open_prices = scaler.inverse_transform(np.concatenate((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], 3))), axis=1))[:, 0]

# Print the results
for i in range(len(predicted_open_prices)):
    print(f"Predicted: {predicted_open_prices[i]}, Actual: {actual_open_prices[i]}")