# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 17:17:22 2022

@author: 857238
"""
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from matplotlib import pyplot as plt
import sklearn.model_selection as model_selection
from keras.datasets import boston_housing
from keras import models
from keras import layers
import numpy as np
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

fileName = 'new_stroke_detection_data.csv'
print("fileName: ", fileName)
raw_data = open(fileName, 'rt')
data = np.loadtxt(raw_data, usecols = (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18),
                  delimiter=",")

x = data[:, :18]
y = data[:, 18]


x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, train_size=0.75,test_size=0.25, random_state=101)


(train_data, train_targets), (test_data, test_targets) = (x_train, y_train), (x_test, y_test)
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape = (1,18) ))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1))

model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

history = model.fit(train_data, train_targets, epochs=100, batch_size=1)

mae_history = history.history['mae']
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)
predicted_prices = model.predict(test_data)

import matplotlib.pyplot as plt
loss = history.history['loss']
epochs = range(1, len(mae_history) + 1)

fig, ax = plt.subplots()
ax.plot(epochs, loss, 'bo', label='Training loss')
ax.set(xlabel='Epochs', ylabel='Loss',
       title='Training and validation loss');
ax.legend()

plt.show()