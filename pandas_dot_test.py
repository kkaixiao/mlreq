import pandas as pd

df1 = pd.DataFrame([[1,2], [0,1]])
df2 = pd.DataFrame([[4,1], [2,3]])

print(df1)
print(df2)
print(df1.dot(df2))