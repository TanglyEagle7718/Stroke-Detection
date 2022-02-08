# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 11:05:08 2022

@author: 857238
"""

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from sklearn.preprocessing import OneHotEncoder
from keras import models
from keras import layers
import sklearn.model_selection as model_selection
from sklearn.preprocessing import StandardScaler
from keras import optimizers

"""
Get data from already cleaned csv file
"""
fileName = 'new_stroke_detection_data.csv'
print("fileName: ", fileName)
raw_data = open(fileName, 'rt')
data = np.loadtxt(raw_data, usecols = (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18),
                  delimiter=",")
x = data[:, :18]
y = data[:, 18]


"""
Split data into training and test data
"""
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y,train_size=0.75,test_size=0.25, random_state=101)

"""
Builds network
"""

model = models.Sequential()
model.add(layers.Dense(18, activation='relu', input_shape = (18,) ))
model.add(layers.Dense(9, activation='relu'))
model.add(layers.Dense(1))

"""
Configures model
"""
x_val = x_train[:1500]
partial_x_train = x_train[1500:]

y_val = y_train[:1500]
partial_y_train = y_train[1500:]

model.compile(optimizer=optimizers.RMSprop(lr=0.005), loss='mse', metrics=['mae','accuracy'])
history = model.fit(partial_x_train, partial_y_train, epochs= 10, batch_size=1, validation_data=(x_val,y_val))
mae_history = history.history['mae']
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

test_mae_score = model.evaluate(x_test, y_test)

'''
Plotting model
'''
epochs = range(1, len(acc) + 1)
fig, ax = plt.subplots()
ax.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
ax.plot(epochs, val_loss, 'b', label='Validation loss')
ax.set(xlabel='Epochs', ylabel='Loss',
       title='Training and validation loss');
ax.legend()

fig1, ax1 = plt.subplots()
ax1.set_ylim(0.94, 0.99)
ax1.plot(epochs, acc, 'ro', label='Training acc')
ax1.plot(epochs, val_acc, 'b', label='Validation acc')
ax1.set(xlabel='Epochs', ylabel='Accuracy', title='Training and validation accuracy');
ax1.legend()

plt.show()

print ("test_mae_score : " + str(test_mae_score))
