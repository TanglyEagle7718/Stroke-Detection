# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from sklearn.preprocessing import OneHotEncoder

def createData():
    fileName = 'new_stroke_detection_data.csv'
    print("fileName: ", fileName)
    raw_data = open(fileName, 'rt')
    data = np.loadtxt(raw_data, usecols = (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18),
                      delimiter=",")
    x = data[:, :18]
    y = data[:, 18]
    finalY = y
    return x, y, finalY

def standardize(arr):
    meanValues = np.mean(arr, axis = 0)
    standardDev = np.std(arr, axis = 0)
    return (arr-meanValues)/standardDev, meanValues, standardDev

def activation(X, W):
  z = np.dot(X,W)
  sigma = 1/(1+np.power( np.e,(-1*z) ))
  return sigma

def calcGradient(X,Y,W):
  sigma = activation(X,W)
  return np.dot(sigma-Y, X)/len(Y)
  
def calcCost(X,W,Y):
  sigma = activation(X,W)
  sum = -1*((Y*np.log(sigma) + (1-Y)*np.log(1-sigma)))
  return np.mean(sum)

############################################################
# Create figure objects
fig = plt.figure()
ax = fig.add_axes([0.1,0.1,0.8,0.8])# [left, bottom, width, height]

#Creates the weights
np.set_printoptions(precision=4)
X, Y, finalY = createData()
numRows = X.shape[0]
numCols = X.shape[1]
W = np.array(np.zeros(numCols))

# set learning rate - the list is if we want to try multiple LR's
lr = 0.3

#set up the cost array for graphing
costArray = []
costArray.append(calcCost(X, W, Y))

#initalize while loop flags
finished = False
count = 0
while (not finished and count <100000):
    gradient = calcGradient(X,Y,W)
    #5 update weights
    W = W - (lr*(gradient))
    #print("weights: ", W)
    costArray.append(calcCost(X, W, Y))
    lengthOfGradientVector = np.linalg.norm(gradient)
    if (lengthOfGradientVector < .000001): 
        finished=True
    print(count)
    count+=1
print("________________________________________________")
print("Final weights: ", W)
print("Final Cost:", costArray[-1])
print("Count = ", count)

ax.set_title("Cost as weights are changed")
ax.set_xlabel("iteration")
ax.set_ylabel("Costs")
ax.plot(np.arange(len(costArray)), costArray, "ro", label = "cost")
ax.legend()   

#print(W_array)
pred = activation(X,W)

def comparePredActual(pred,actual):
  #pred = np.argmax(pred, axis=1)
  pred = np.argmax(pred)
  pred = pred + 1
  correct = (pred == actual)
  correctCount = np.sum(correct)
  return correct,correctCount

correct, correctCount = comparePredActual(pred,finalY)
print("Number of correct predictions: ", correctCount)
print("Number of incorrect predictions:", (len(pred) - correctCount))

             
        

    