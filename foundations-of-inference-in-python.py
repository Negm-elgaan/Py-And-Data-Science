# Select a random starting row number, not including the last 90 rows
initial_row_number = np.random.choice(range(len(btc_sp_df) - 90))

# Use initial_row_number to select the next 90 rows from that row number
sample_df = btc_sp_df.iloc[initial_row_number:(initial_row_number + 90)]
#############################
# Select a random starting row number, not including the last 90 rows
initial_row_number = np.random.choice(range(btc_sp_df.shape[0] - 90))

# Use initial_row_number to select the next 90 rows from that row number
sample_df = btc_sp_df.iloc[initial_row_number:(initial_row_number + 90)]

# Use sample_df to compute the percent increase in Close_SP500
sp500_pct_change = (sample_df.iloc[0]['Close_SP500'] - sample_df.iloc[-1]['Close_SP500']) / sample_df.iloc[0]['Close_SP500']

# Use sample_df to compute the percent increase in Close_BTC
btc_pct_change = (sample_df.iloc[0]['Close_BTC'] - sample_df.iloc[-1]['Close_BTC']) / sample_df.iloc[0]['Close_BTC']

print('SP500: ', sp500_pct_change, '\n', 'BTC: ', btc_pct_change)