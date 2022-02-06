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
model.add(layers.Dense(44, activation='relu'))
model.add(layers.Dense(1))

from keras import optimizers

model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

x_val = x_train[:18]
partial_x_train = x_train[18:]

y_val = y_train[:18]
partial_y_train = y_train[18:]

history = model.fit(partial_x_train, partial_y_train, epochs=50, batch_size=1, validation_data=(x_val, y_val))

#mae_history = history.history['mae']
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)
predicted_prices = model.predict(test_data)

results = model.evaluate(partial_x_train, partial_y_train)
print ("train:", results)
results = model.evaluate(x_val, y_val)
print ("validation:", results)
results = model.evaluate(partial_x_train, partial_y_train)
print ("all data", results)

history_dict = history.history
print("history dict.keys():", history_dict.keys())


import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# =============================================================================
# # "bo" is for "blue dot"
# plt.plot(epochs, loss, 'bo', label='Training loss')
# # b is for "solid blue line"
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# 
# plt.show()
# =============================================================================

"""
plot accuracy
"""
fig, ax = plt.subplots()
ax.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
ax.plot(epochs, val_loss, 'b', label='Validation loss')
ax.set(xlabel='Epochs', ylabel='Loss',
       title='Training and validation loss');
ax.legend()
acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']

fig1, ax1 = plt.subplots()
ax1.plot(epochs, acc, 'ro', label='Training acc')
ax1.plot(epochs, val_acc, 'b', label='Validation acc')
ax1.set(xlabel='Epochs', ylabel='Loss',
       title='Training and validation accuracy');
ax1.legend()

plt.show()

"""
predict
"""

testPrediction = model.predict(x_test)