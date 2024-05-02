import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import os

# Load a single stock data using Pandas
data = pd.read_csv("C:/Users/Eddie/Documents/GitHub/risk_n_alysis_project/archive/Stocks/tsla.us.txt")

# Split data into training and testing portions with an 80/20 split
training_size = int(len(data)*0.80)
data_len = len(data)
train, test = data[0:training_size],data[training_size:data_len]

# Pick out the "Open" prices
train = train.loc[:, ["Open"]].values

# Normalize the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train)

# Predict the 40th value of a 40-length series
end_len = len(train_scaled)
X_train = []
y_train = []
timesteps = 40
for i in range(timesteps, end_len):
    X_train.append(train_scaled[i - timesteps:i, 0])
    y_train.append(train_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Each data point needs to be nested in it's own array for Keras to work properly
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
print("X_train --> ", X_train.shape)
print("y_train shape --> ", y_train.shape)

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import SimpleRNN
from keras.layers import Dropout

# Build the model using SimpleRNN layers
regressor = Sequential()

regressor.add(SimpleRNN(units = 50, activation = "tanh", return_sequences = True, input_shape = (X_train.shape[1],1)))
regressor.add(Dropout(0.2))

regressor.add(SimpleRNN(units = 50, activation = "tanh", return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(SimpleRNN(units = 50, activation = "tanh", return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(SimpleRNN(units = 50))
regressor.add(Dropout(0.2))

# The model output is a scalar
regressor.add(Dense(units = 1))

# Compile the model
regressor.compile(optimizer= "adam", loss = "mean_squared_error")

# Train the model
epochs = 100 
batch_size = 20
regressor.fit(X_train, y_train, epochs = epochs, batch_size = batch_size)
real_price = test.loc[:, ["Open"]].values

# Prepare the testing data
dataset_total = pd.concat((data["Open"], test["Open"]), axis = 0)
inputs = dataset_total[len(dataset_total) - len(test) - timesteps:].values.reshape(-1,1)
inputs = scaler.transform(inputs)

X_test = []
for i in range(timesteps, 412):
    X_test.append(inputs[i-timesteps:i, 0])
X_test = np.array(X_test)

# Again reshape the data so that each point is in a nested array
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Now predict the future stock prices
predict = regressor.predict(X_test)
predict = scaler.inverse_transform(predict)

# Plot the data
plt.plot(real_price, color = "red", label = "Real Stock Price")
plt.plot(predict, color = "black", label = "Predict Stock Price")
plt.title("Stock Price Prediction")
plt.xlabel("Time")
plt.ylabel("Tesla Stock Price")
plt.legend()
plt.show()
