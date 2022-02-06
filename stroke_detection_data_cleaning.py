# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 16:14:47 2022

@author: 836666
"""

import numpy as np
from sklearn.preprocessing import OneHotEncoder

fileName = 'healthcare-dataset-stroke-data.csv'
print("fileName: ", fileName)
raw_data = open(fileName, 'rt')
data = np.loadtxt(raw_data, usecols = (6,10,1,2,3,4,5,7,8,9,11), skiprows = 1, delimiter=",", dtype=str)

x = data[:, :10]
y = data[:, 10]

#standardizing
def standardize(x):
    x = x.astype(np.float64)
    mean = np.mean(x, axis=0)
    std_dev = np.std(x, axis=0)
    return (x - mean)/std_dev
#age
x[:, 3] = standardize(x[:, 3])
#average glucose level
x[:, 8] = standardize(x[:, 8])

#boolean masking
#gender
x[:, 2][x[:, 2] == 'Female'] = 1
x[:, 2][x[:, 2] == 'Male'] = 0
#ever married
x[:, 6][x[:, 6] == 'Yes'] = 1
x[:, 6][x[:, 6] == 'No'] = 0
#residence type
x[:, 7][x[:, 7] == 'Urban'] = 1
x[:, 7][x[:, 7] == 'Rural'] = 0

#ohe for work type and smoking status
ohe = OneHotEncoder(categories = 'auto')
#work type
work_type_ohe = ohe.fit_transform(x[:,0:1]).toarray()
#smoking status
smoking_status_ohe = ohe.fit_transform(x[:,1:2]).toarray()
#adding ohe columns (first 3 are work type, second 3 are smoking status)
x = x[:, 2:]
x = np.column_stack((work_type_ohe, smoking_status_ohe, x))

#normalizing bmi
#changing all 'N/A' to nan
x[:, 16][x[:, 16] == 'N/A'] = np.nan

x = x.astype(np.float64)

bmi_mean = np.nanmean(x[:, 16], axis = 0)
x[:, 16][np.isnan(x[:, 16])] = bmi_mean

bmi_max = np.max(x[:, 16], axis = 0)
bmi_min = np.min(x[:, 16], axis = 0)
bmi_mean = np.mean(x[:, 16], axis = 0)

bmis_normed = (x[:, 16] - bmi_mean)/(bmi_max - bmi_min)
x = np.column_stack((x[:, :16], bmis_normed))

#adding bias
bias = np.ones((x.shape[0], 1))
x = np.column_stack((bias, x))
