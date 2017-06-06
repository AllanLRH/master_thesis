import numpy as np
import pandas as pd


np.random.seed(0)
df = pd.DataFrame(
    data=np.random.randint(0, 10, size=(7, 3)),
    index=[['u01', 'u01', 'u01', 'u02', 'u02', 'u03', 'u03'], ['C', 'C', 'C', 'C', 'T', 'T', 'T']],
    columns=['foo', 'bar', 'baz'])
df.index.names = ['user', 'comType']

print('df', df, sep=" =\n\n", end="\n"*5)

mask = (df.baz > 8) | (df.baz < 2)
print('mask', mask, sep=" =\n\n", end="\n"*5)

print('df.loc[mask]', df.loc[mask], sep=" =\n\n", end="\n"*5)
# Procuces same result as the line above
# print('df.iloc[np.where(mask)[0]]', df.iloc[np.where(mask)[0]], sep=" =\n\n", end="\n"*5)

df2 = df.drop(mask.index[mask.values])  # Drops all of user user u01, one of user u02
print('df2', df2, sep=" =\n\n", end="\n"*5)

df3 = df.drop(mask)  # Doesn't do anything
print('df3', df3, sep=" =\n\n", end="\n"*5)
