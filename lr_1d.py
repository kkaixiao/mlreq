import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# X, Y = [], []
# for line in open('data/data_1d.csv'):
#     x, y = line.split(',')
#     X.append(float(x))
#     Y.append(float(y))
#
# X, Y = np.array(X), np.array(Y)

df = pd.read_csv('data/data_1d.csv', names=['X', 'Y'])

X, Y = df["X"], df["Y"]

# print('X', X, type(X))

# plt.scatter(X, Y)
# plt.grid(alpha=0.2)
# plt.show()

denominator = X.dot(X) - X.mean()*X.sum()

a = (X.dot(Y) - Y.mean()*X.sum()) / denominator
b = (Y.mean() * X.dot(X) - X.mean() * X.dot(Y)) / denominator

# print(a,b)
Yhat = a*X + b

plt.scatter(X, Y)
plt.plot(X, Yhat, color='red')
plt.grid(alpha=0.2)
plt.show()

residual = Y - Yhat
total = Y - Y.mean()
r2 = 1 - residual.dot(residual) / total.dot(total)
print(r2)

