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
model.add(layers.Dense(1, activation='sigmoid'))


"""
Configures model
"""

x_val = x_train[:400]
partial_x_train = x_train[400:]

y_val = y_train[:400]
partial_y_train = y_train[400:]

model.compile(optimizer=optimizers.RMSprop(lr=0.001), loss='mse', metrics=['mae','accuracy'])
history = model.fit(partial_x_train, partial_y_train, epochs= 50, batch_size=1024, validation_data=(x_val,y_val))
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
ax1.set_ylim(0.75, 0.99)
ax1.plot(epochs, acc, 'ro', label='Training acc')
ax1.plot(epochs, val_acc, 'b', label='Validation acc')
ax1.set(xlabel='Epochs', ylabel='Accuracy', title='Training and validation accuracy');
ax1.legend()

plt.show()

print ("test_mae_score : " + str(test_mae_score))

def comments():
    # =============================================================================
    # model = models.Sequential()
    # model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
    # model.add(layers.Dense(16, activation='relu'))
    # model.add(layers.Dense(1, activation='sigmoid'))
    # 
    # =============================================================================
    """
    TRAIN
    We will now train our model for 20 epochs (20 iterations over all samples in
     the x_train and y_train tensors), in mini-batches of 512 samples. At this
     same time we will monitor loss and accuracy on the 10,000 samples that we
     set apart. This is done by passing the validation data as the validation_data
     argument:
    history = model.fit(partial_x_train,
                        partial_y_train,
                        epochs=20,
                        batch_size=512,
                        validation_data=(x_val, y_val))



    RESULTS
    This object has a member history, which is a dictionary containing data about
     everything that happened during training
    It contains 4 entries: one per metric that was being monitored, during 
    training and during validation.

    results = model.evaluate(partial_x_train, partial_y_train)
    print ("train:", results)
    results = model.evaluate(x_val, y_val)
    print ("validation:", results)
    results = model.evaluate(x_test, y_test)
    print ("all data", results)

    history_dict = history.history
    print("history dict.keys():", history_dict.keys())

    """
