
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


# load the data
X = []
Y = []


df = pd.read_csv('data/data_2d.csv', sep=',', header=None, prefix='Col')

X = pd.concat([df['Col0'], df['Col1']], axis=1)
Y = df['Col2']

# see what data looks like
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(np.array(X)[:,0], np.array(X)[:,1], np.array(Y))
plt.show()


# use np function to solve the question to find out weights
w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
Yhat = np.dot(X, w)


# determine how good the model is by computing the r-squared
d1 = Y - Yhat
d2 = Y - Y.mean()
r2 = 1 - d1.dot(d1) / d2.dot(d2)
print("the r-squared is:", r2)
