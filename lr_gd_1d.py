
import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_csv('data/data_1d.csv', names=['X', 'Y'])

X, Y = df["X"], df["Y"]

# plt.scatter(X, Y)
# plt.grid(alpha=0.2)
# plt.show()

a = b = 0

learningRate, epochs, n = 0.0001, 100000, len(X)

cost, prevDerivativeA, derivativeA = 0.0000001, float('inf'), 0

# if change of 'derivativeA' can not converge to the set value of 'cost',
# stop iteration after times of epochs
while abs(prevDerivativeA - derivativeA) > cost and epochs > 0:
    Yhat = a*X + b
    prevDerivativeA = derivativeA
    derivativeA = -2 * (X * (Y - Yhat)).mean()
    derivativeB = -2 * (Y - Yhat).mean()
    a -= learningRate * derivativeA
    b -= learningRate * derivativeB
    # print(derivativeA, derivativeB)
    epochs -= 1

Yhat = a*X + b

plt.scatter(X, Y)
plt.plot(X, Yhat, color='red')
plt.grid(alpha=0.2)
plt.show()

residual = Y - Yhat
total = Y - Y.mean()
r2 = 1 - residual.dot(residual) / total.dot(total)
print(r2)





