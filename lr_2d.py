
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


# load the data
X = []
Y = []

# for line in open('data/data_2d.csv'):
#     x1, x2, y = line.split(',')
#     X.append([float(x1), float(x2), 1]) # add the bias term
#     Y.append(float(y))

# df = pd.read_csv('data/data_2d.csv', names = ['x1', 'x2', 'y'])
df = pd.read_csv('data/data_2d.csv', sep=',', header=None, prefix='Col')
# print(df)
# X = df['Col0'] + df['Col1']

X = pd.concat([df['Col0'], df['Col1']], axis = 1)
Y = df['Col2']
# Y = np.array(y)
# # let's turn X and Y into numpy arrays since that will be useful later
X = np.array(X)
Y = np.array(Y)


# let's plot the data to see what it looks like
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], Y)
plt.show()


# apply the equations we learned to calculate a and b
# numpy has a special method for solving Ax = b
# so we don't use x = inv(A)*b
# note: the * operator does element-by-element multiplication in numpy
#       np.dot() does what we expect for matrix multiplication
w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
Yhat = np.dot(X, w)


# determine how good the model is by computing the r-squared
d1 = Y - Yhat
d2 = Y - Y.mean()
r2 = 1 - d1.dot(d1) / d2.dot(d2)
print("the r-squared is:", r2)
