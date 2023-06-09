import pandas as pd
from matplotlib import pyplot as plt
from util import scatter_plot, histogram, density_histogram

df = pd.read_csv('cleaned_data.csv')

uni = df.loc[df['First Unicorn Round Flag'] == 1]
deal = (uni['Deal Type'].value_counts())
deal = deal.reindex(['Seed Round', 'Early Stage VC', 'Later Stage VC'])
plt.figure()
plt.bar(deal.index, deal.values)
plt.savefig('./plots/deal_type.png')
plt.close()

uni_early_stage = uni.loc[uni['Deal Type'] == 'Early Stage VC']
deal2 = (uni_early_stage['Deal Type 2'].value_counts())
deal2 = deal2.reindex(['Series A', 'Series B'])
plt.bar(deal2.index, deal2.values)
plt.savefig('./plots/early_stage_vc.png')
plt.close()

seed = df.loc[df['Deal Type'] == 'Seed Round']
early_stage = df.loc[df['Deal Type'] == 'Early Stage VC']
early_stage_a = early_stage[early_stage['Deal Type 2'] == 'Series A']
early = pd.concat([seed, early_stage_a])

one_deal = early.loc[early['VC Deal Number'] == 1]
two_deals = early.loc[early['VC Deal Number'] == 2]
three_deals = early.loc[early['VC Deal Number'] == 3]
four_deals = early.loc[early['VC Deal Number'] == 4]
five_or_more_deals = early.loc[early['VC Deal Number'] >= 5]

exactly_one_deal = one_deal.shape[0] - two_deals.shape[0]
exactly_two_deals = two_deals.shape[0] - three_deals.shape[0]
exactly_three_deals = three_deals.shape[0] - four_deals.shape[0]
exactly_four_deals = four_deals.shape[0] - five_or_more_deals.shape[0]
five_or_more_deals = five_or_more_deals.shape[0]

deals = {'one': exactly_one_deal, 
         'two': exactly_two_deals, 
         'three': exactly_three_deals, 
         'four': exactly_four_deals,
         'five or more': five_or_more_deals}

plt.bar(range(len(deals)), list(deals.values()), tick_label=list(deals.keys()))
plt.savefig('./plots/early_exact_counts.png')
plt.close()

three_or_more = pd.read_csv('three_or_more_flattened.csv')
two_or_more = pd.read_csv('two_or_more_stripped.csv')

data_frames = {'three': three_or_more, 'two': two_or_more}

for name in data_frames:
    data = data_frames[name]
    print(data.shape)
    unicorn_dist = (data['IsUnicorn'].value_counts())
    print(name, '/n', unicorn_dist)
    plt.bar(unicorn_dist.index, unicorn_dist.values)
    save_path = './plots/unicorn_dist_' + name + '.png'
    plt.savefig(save_path)
    plt.close()

# scatter_plot(df['First Financing Date'], df['First Financing Valuation'], './plots/first_financing.png')
# scatter_plot(df['Last Known Valuation Date'], df['Last Known Valuation'], './plots/last_valuation.png')

# # maybe we don't need these and just keep density histograms 
# histogram(df['First Financing Date'], bins=15, save_path='./plots/first_financing_date_hist.png')
# histogram(df['First Financing Valuation'], bins=15, save_path='./plots/first_financing_val_hist.png')
# histogram(df['First Financing Size'], bins=15, save_path='./plots/first_financing_size_hist.png')

# density_histogram(df['First Financing Date'], bins=15, save_path='./plots/first_financing_date_dhist.png')
# density_histogram(df['First Financing Valuation'], bins=15, save_path='./plots/first_financing_val_dhist.png')
# density_histogram(df['First Financing Size'], bins=15, save_path='./plots/first_financing_size_dhist.png')