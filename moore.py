# shows how linear regression analysis can be applied to moore's law

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

X = []
Y = []

# some numbers show up as 1,170,000,000 (commas)
# some numbers have references in square brackets after them


df = pd.read_csv('data/moore.csv', sep='\t', header=None, prefix='X')

non_decimal = re.compile(r'[^\d]+')
func = lambda x: int(non_decimal.sub('', x.split('[')[0]))

df.X2 = df.X2.apply(func)    # X (year)
df.X1 = df.X1.apply(func)    # Y

X = df.X2.to_numpy()
Y = np.log(df.X1.to_numpy())


plt.scatter(X, Y)
plt.show()

Y = np.log(Y)
plt.scatter(X, Y)
plt.show()

# copied from lr_1d.py
denominator = X.dot(X) - X.mean() * X.sum()
a = ( X.dot(Y) - Y.mean()*X.sum() ) / denominator
b = ( Y.mean() * X.dot(X) - X.mean() * X.dot(Y) ) / denominator

# let's calculate the predicted Y
Yhat = a*X + b

plt.scatter(X, Y)
plt.plot(X, Yhat)
plt.show()

# determine how good the model is by computing the r-squared
d1 = Y - Yhat
d2 = Y - Y.mean()
r2 = 1 - d1.dot(d1) / d2.dot(d2)
print("a:", a, "b:", b)
print("the r-squared is:", r2)

# how long does it take to double?
# log(transistorcount) = a*year + b
# transistorcount = exp(b) * exp(a*year)
# 2*transistorcount = 2 * exp(b) * exp(a*year) = exp(ln(2)) * exp(b) * exp(a * year) = exp(b) * exp(a * year + ln(2))
# a*year2 = a*year1 + ln2
# year2 = year1 + ln2/a
print("time to double:", np.log(2)/a, "years")
