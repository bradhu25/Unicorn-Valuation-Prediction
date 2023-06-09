import pandas as pd

df = pd.read_csv('PitchBook - Morningstar - Unicorn prediction data.csv')

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

print(df.shape)

print(df.isna().sum())