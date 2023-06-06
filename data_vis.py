import pandas as pd
from util import scatter_plot, histogram, density_histogram

df = pd.read_csv('cleaned_data.csv')

scatter_plot(df['First Financing Date'], df['First Financing Valuation'], './plots/first_financing.png')
scatter_plot(df['Last Known Valuation Date'], df['Last Known Valuation'], './plots/last_valuation.png')

# maybe we don't need these and just keep density histograms 
histogram(df['First Financing Date'], bins=15, save_path='./plots/first_financing_date_hist.png')
histogram(df['First Financing Valuation'], bins=15, save_path='./plots/first_financing_val_hist.png')
histogram(df['First Financing Size'], bins=15, save_path='./plots/first_financing_size_hist.png')

density_histogram(df['First Financing Date'], bins=15, save_path='./plots/first_financing_date_dhist.png')
density_histogram(df['First Financing Valuation'], bins=15, save_path='./plots/first_financing_val_dhist.png')
density_histogram(df['First Financing Size'], bins=15, save_path='./plots/first_financing_size_dhist.png')
