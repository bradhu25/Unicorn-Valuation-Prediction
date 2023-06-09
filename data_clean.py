import pandas as pd
import math

df = pd.read_csv('PitchBook - Morningstar - Unicorn prediction data.csv')

df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# creating uniform series names in Deal Type 2
# adding IsUnicorn column

series_names = ['Series A', 'Series B', 'Series C', 'Series D', 'Series E', 'Series 1', 'Series 2', 'Series 3']
series_updates = {'Series 1': 'Series A', 'Series 2': 'Series B', 'Series 3': 'Series C'}

unicorns = set()
for i in range(df.shape[0]):
    if df.loc[i, 'First Unicorn Round Flag']:
        unicorns.add(df.loc[i, 'Business Entity ID'])

df['IsUnicorn'] = 0

for i in range(df.shape[0]):
    if i % 10000 == 0:
        print(i, ' of ', df.shape[0])
    if isinstance(df.loc[i, 'Deal Type 2'], str):
        for series in series_names:
            if str(df.loc[i, 'Deal Type 2']).startswith(series):
                if series in series_updates:
                    df.loc[i, 'Deal Type 2'] = series_updates[series]
                else:
                    df.loc[i, 'Deal Type 2'] = series

    if isinstance(df.loc[i, 'Deal Type 2'], float):
        if math.isnan(df.loc[i, 'Deal Type 2']):
            if df.loc[i, 'First Unicorn Round Flag'] and (df.loc[i, 'Deal Type'] == 'Early Stage VC'):
                df.loc[i, 'Deal Type 2'] = 'Series B'
            else:
                df.loc[i, 'Deal Type 2'] = 'Series A'

    if df.loc[i, 'Business Entity ID'] in unicorns:
        df.loc[i, 'IsUnicorn'] = 1

df.to_csv('cleaned_data.csv', index=False)

# df['Emerging Spaces'] = df['Emerging Spaces'].fillna(0)
# df = df.dropna()

# for dates_col in ['First Financing Date', 'Last Known Valuation Date']:
#     df[dates_col] = pd.DatetimeIndex(df[dates_col])
#     df[dates_col]= df[dates_col].dt.year + ((df[dates_col].dt.dayofyear - 1) / 365)

# df = df.replace(',','', regex=True)

# for float_col in ['Last Known Valuation', 'First Financing Valuation', 'First Financing Size']:
#     df[float_col] = df[float_col].astype(float)

# df['Unicorn Status'] = 1 * (df['Last Known Valuation'] >= 1000)

# df.to_csv('cleaned_data.csv', index=False)