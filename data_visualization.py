import pandas as pd
from util import scatter_plot

df = pd.read_csv('cleaned_data.csv')

scatter_plot(df['First Financing Date'], df['First Financing Valuation'], './plots/first_financing.png')

scatter_plot(df['Last Known Valuation Date'], df['Last Known Valuation'], './plots/last_valuation.png')