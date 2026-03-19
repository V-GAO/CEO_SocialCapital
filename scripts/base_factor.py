import pandas as pd
import s3fs
import numpy as np

db = wrds.connect(wrds_username = 'vincent0358')

def construct_portfolio(self, factor_df, returns_df, n_groups=3):
    """

    Construct portfolio based on sustainability factors (i.e. quintiles created before)

    Parameters:
    factor_df
    returns_df
    """
    portfolio_returns = pd.DataFrame(index=returns_df.index) # check output

    for i, date in enumerate(returns_df.index):
        if date not in factor_df.index or date == returns_df.index[0]:
            continue

        prev_dates = factor_df.index[factor_df.index < date]

        if len(prev_dates) == 0:
            continue

        prev_date = prev_dates[-1]

        factor_values = factor_df.loc[prev_date].dropna()

        if len(factor_values) < n_groups: # use 3 as n_groups or number of groups
            continue

        sorted_stocks = factor_values.sort_values(ascending=False)
        
        # divide into groups
        group_size = max(1, len(sorted_stocks) // n_groups)

        # top groups / top quintile
        high_factor_stocks = sorted_stocks.iloc[:group_size].index.tolist() # gives IDs of stocks

        # bottom groups / bottom quintile
        low_factor_stocks = sorted_stocks.iloc[-group_size:].index.tolist()
