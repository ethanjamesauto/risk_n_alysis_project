import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import os

#ETHAN CHANGE THIS SHIG TO YOUR FILE PATH
data = pd.read_csv("C:/Users/Eddie/Documents/GitHub/risk_n_alysis_project/archive/Stocks/tsla.us.txt")

training_size = int(len(data)*0.80)
data_len = len(data)
train, test = data[0:training_size],data[training_size:data_len]


train = train.loc[:, ["Open"]].values
#POOP IN 10
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train)

#POOP IN 11
end_len = len(train_scaled)
X_train = []
y_train = []
timesteps = 40
for i in range(timesteps, end_len):
    X_train.append(train_scaled[i - timesteps:i, 0])
    y_train.append(train_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)


#POOP IN 12
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
print("X_train --> ", X_train.shape)
print("y_train shape --> ", y_train.shape)

#POOP IN 14
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import SimpleRNN
from keras.layers import Dropout

#POOP IN 15
regressor = Sequential()

regressor.add(SimpleRNN(units = 50, activation = "tanh", return_sequences = True, input_shape = (X_train.shape[1],1)))
regressor.add(Dropout(0.2))

regressor.add(SimpleRNN(units = 50, activation = "tanh", return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(SimpleRNN(units = 50, activation = "tanh", return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(SimpleRNN(units = 50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units = 1))

#POOP IN 16
regressor.compile(optimizer= "adam", loss = "mean_squared_error")

#POOP IN 17
epochs = 100 
batch_size = 20

#POOP IN 18
regressor.fit(X_train, y_train, epochs = epochs, batch_size = batch_size)

#POOP IN 19
test.head()

#POOP IN 20
real_price = test.loc[:, ["Open"]].values
print("Real Price Shape --> ", real_price.shape)

#POOP IN 22
dataset_total = pd.concat((data["Open"], test["Open"]), axis = 0)
inputs = dataset_total[len(dataset_total) - len(test) - timesteps:].values.reshape(-1,1)
inputs = scaler.transform(inputs)

#POOP IN 23
X_test = []

for i in range(timesteps, 412):
    X_test.append(inputs[i-timesteps:i, 0])
X_test = np.array(X_test)

print("X_test shape --> ", X_test.shape)

#POOP IN 24
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predict = regressor.predict(X_test)
predict = scaler.inverse_transform(predict)

#POOP IN 25
plt.plot(real_price, color = "red", label = "Real Stock Price")
plt.plot(predict, color = "black", label = "Predict Stock Price")
plt.title("Stock Price Prediction")
plt.xlabel("Time")
plt.ylabel("Tesla Stock Price")
plt.legend()
plt.show()
