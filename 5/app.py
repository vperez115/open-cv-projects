# Functions to read and show images.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.linalg import eigh 

d0 = pd.read_csv('C:/Users/Victo/Desktop/open-cv-projects/5/mnist_train.csv')

print(d0.head(5)) # print first five rows of d0.

# save the labels into a variable l.
l = d0['label']

# Drop the label feature and store the pixel data in d.
d = d0.drop("label",axis=1)
print(d.shape)
print(l.shape)

def display():
    # display or plot a number.
    plt.figure(figsize=(7,7))
    idx = 1

    grid_data = d.iloc[idx].to_numpy().reshape(28,28)  # reshape from 1d to 2d pixel array
    plt.imshow(grid_data, interpolation = "none", cmap = "gray")
    plt.show()

    print(l[idx])
# display()

def label():
    # Pick first 15K data-points to work on for time-effeciency.
    # Excercise: Perform the same analysis on all of 42K data-points.

    labels = l.head(15000)
    data = d.head(15000)

    print("the shape of sample data = ", data.shape)

    # Data-preprocessing: Standardizing the data

    standardized_data = StandardScaler().fit_transform(data)
    print(standardized_data.shape)

    #find the co-variance matrix which is : A^T * A
    sample_data = standardized_data

    # matrix multiplication using numpy
    covar_matrix = np.matmul(sample_data.T , sample_data)

    print ( "The shape of variance matrix = ", covar_matrix.shape)

    # finding the top two eigen-values and corresponding eigen-vectors. 
    # for projecting onto a 2-Dim space.

    # the parameter 'eigvals' is defined (low value to heigh value). 
    # eigh function will return the eigen values in asending order.
    # this code generates only the top 2 (782 and 783) eigenvalues.
    values, vectors = eigh(covar_matrix, subset_by_index=[782, 783])

    print("Shape of eigen vectors = ",vectors.shape)
    # converting the eigen vectors into (2,d) shape for easyness of further computations.
    vectors = vectors.T

    print("Updated shape of eigen vectors = ",vectors.shape)
    # here the vectors[1] represent the eigen vector corresponding 1st principal eigen vector.
    # here the vectors[0] represent the eigen vector corresponding 2nd principal eigen vector.

label()
