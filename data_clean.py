import pandas as pd
import math

df = pd.read_csv('PitchBook - Morningstar - Unicorn prediction data.csv')

# remove extra empty columns
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# keep only features we are including
identifiers = ['Round Entity ID',
               'Business Entity ID']

labels_features = ['First Unicorn Round Flag']

round_features = ['Close Date',
             'Close Quarter',
             ' Deal Size (millions) ',
             'Mega Deal?',
             'Pre Value (millions)',
             'Post Value (millions)',
             'Traditional VC Investor Count',
             'Non-Traditional VC Investor Count',
             'US Investor Count',
             'Europe Investor Count',
             'Female Founder Count',
             'Investor Count',
             'CVC Investor Involvement',
             'PE Investor Involvement',
             'Hedge Fund Investor Involvement',
             'Asset Manager Investor Involvement',
             'Government/SWF Investor Involvement',
             'Number of Lead Investors on Deal',
             'Number of Non-Lead Investors on Deal',
             'Number of New Investors',
             'Number of New Lead Investors',
             'Number of Follow On Investors',
             'Number of Lead Follow On Investors',
             'At Least One Lead Investor is New and Got Board Seat',
             'Crossover Investor was a Lead Investor',
             'Notable Investor Count',
             'Notable Investor Involvement',
             'VC Raised to Date']

business_features = ['Industry Sector',
                     'Industry Group',
                     'Industry Code',
                     'Global Region',
                     'City',
                     'Country',
                     'Founding Year']

round_identifiers = ['Deal Type',
                     'Deal Type 2',
                     'VC Deal Number']


# only keep above columns
keep = identifiers + labels_features + round_features + business_features + round_identifiers
df = df[keep]

# fix var types
df[' Deal Size (millions) '] = pd.to_numeric(df[' Deal Size (millions) '], errors='coerce')

# fil na values with mean
fill_cols = ['Traditional VC Investor Count', 
             'Non-Traditional VC Investor Count', 
             'US Investor Count', 
             'Europe Investor Count', 
             'Female Founder Count', 
             'Investor Count', 
             'Number of Lead Investors on Deal', 
             'Number of Non-Lead Investors on Deal', 
             'Number of New Investors', 
             'Number of New Lead Investors', 
             'Number of Follow On Investors',
             'Number of Lead Follow On Investors',
             'At Least One Lead Investor is New and Got Board Seat', 
             'Crossover Investor was a Lead Investor', 
             'Notable Investor Count',
             ' Deal Size (millions) ',
             'Pre Value (millions)',
             'Post Value (millions)',
             'VC Raised to Date',
             'Notable Investor Involvement']

for col in fill_cols:
    col_avg = df[col].mean()
    df[col] = df[col].fillna(col_avg)


# change yes/no to 1/0
yes_no_cols = ['CVC Investor Involvement', 
               'PE Investor Involvement', 
               'Hedge Fund Investor Involvement', 
               'Asset Manager Investor Involvement', 
               'Government/SWF Investor Involvement']

for col in yes_no_cols:
    df[col] = df[col].eq('Yes').mul(1)

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

# date as integer
df['Close Date'] = pd.DatetimeIndex(df['Close Date'])
df['Close Date']= df['Close Date'].dt.year + ((df['Close Date'].dt.dayofyear - 1) / 365)

# remove angel rounds
print(df.shape)
df = df[df['Deal Type'] != 'Angel (individual)']
print(df.shape)

# remove na
print(df.shape)
df = df.dropna()
print(df.shape)

df.to_csv('cleaned_data.csv', index=False)