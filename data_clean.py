import pandas as pd

df = pd.read_csv('PitchBook - Morningstar - Unicorn prediction data.csv')

print(df[df['Round Entity ID'] == 587392]['Deal Type 2'])

# print(df[df['Round Entity ID'] == 587392]['Deal Type 2'])

for i in df.shape[0]:
    print

print(df['Deal Type 2'])

if df['Deal Type 2'].startswith('Series A'):
    df['Deal Type 2'] = 'Series A'

print(df[df['Round Entity ID'] == 587392]['Deal Type 2'])

# df['Emerging Spaces'] = df['Emerging Spaces'].fillna(0)
# df = df.dropna()

# for dates_col in ['First Financing Date', 'Last Known Valuation Date']:
#     df[dates_col] = pd.DatetimeIndex(df[dates_col])
#     df[dates_col]= df[dates_col].dt.year + ((df[dates_col].dt.dayofyear - 1) / 365)

# df = df.replace(',','', regex=True)

# for float_col in ['Last Known Valuation', 'First Financing Valuation', 'First Financing Size']:
#     df[float_col] = df[float_col].astype(float)

# df['Unicorn Status'] = 1 * (df['Last Known Valuation'] >= 1000)

df.to_csv('cleaned_data.csv', index=False)