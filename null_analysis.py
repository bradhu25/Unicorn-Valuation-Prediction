import pandas as pd

df = pd.read_csv('two_or_more.csv')

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

print(df.shape)

print(df.isna().sum())