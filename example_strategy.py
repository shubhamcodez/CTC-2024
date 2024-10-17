import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import itertools


class Strategy:

  def __init__(self) -> None:
    self.capital : float = 100_000_000
    self.portfolio_value : float = 0

    self.start_date : datetime = datetime(2024, 1, 1)
    self.end_date : datetime = datetime(2024, 3, 30)

    self.options : pd.DataFrame = pd.read_csv("data/cleaned_options_data.csv")
    self.options["day"] = self.options["ts_recv"].apply(lambda x: x.split("T")[0])

    self.underlying = pd.read_csv("data/underlying_data_hour.csv")
    self.underlying.columns = self.underlying.columns.str.lower()

  @classmethod
  def parse_option_symbol(cls,symbol):
    # Example: SPX 240119P04700000
    numbers = symbol.split(' ')[3]
    expiry_date = numbers[:6]  # Expiry date in YYMMDD format
    option_type = numbers[6]  # P for put, C for call
    strike_price = float(numbers[7:]) / 1000  # Strike price divided by 1000
    return expiry_date, option_type, strike_price

  @classmethod
  def third_friday_of_month(cls,date):
    # First day of the month
    first_day_of_month = date.replace(day=1)

    # Find the first Friday of the month
    first_friday_offset = (4 - first_day_of_month.weekday()) % 7  # 4 corresponds to Friday
    first_friday = first_day_of_month + timedelta(days=first_friday_offset)

    # The third Friday is 14 days after the first Friday
    third_friday = first_friday + timedelta(weeks=2)

    # If the current date is after the third Friday, move to the next month's third Friday
    if date > third_friday:
        # Move to the next month's first day
        next_month = (date.month % 12) + 1
        year = date.year + (1 if date.month == 12 else 0)
        first_day_of_next_month = datetime(year, next_month, 1)

        # Recalculate the third Friday for the next month
        first_friday_offset = (4 - first_day_of_next_month.weekday()) % 7
        first_friday = first_day_of_next_month + timedelta(days=first_friday_offset)
        third_friday = first_friday + timedelta(weeks=2)

    return third_friday

  @classmethod
  def get_best_spreads(cls, df):
      # Reset index to preserve original indices
      df = df.reset_index().rename(columns={'index': 'idx_1'})

      # Add a temporary key column for cross join
      df['key'] = 1

      # Perform a self-merge to create all combinations within the same option type
      df_merged = df.merge(df, on='key', suffixes=('_1', '_2'))

      # Remove temporary key column
      df_merged = df_merged.drop(columns=['key'])

      # Filter out self-pairs and duplicate pairs
      df_merged = df_merged[df_merged['idx_1_1'] < df_merged['idx_1_2']]

      # Only consider pairs with the same option type
      df_merged = df_merged[df_merged['typ_1'] == df_merged['typ_2']]

      # Exclude pairs with the same strike price
      df_merged = df_merged[df_merged['strike_1'] != df_merged['strike_2']]

      # Calculate strike difference
      df_merged['strike_diff'] = df_merged['strike_1'] - df_merged['strike_2']
      df_merged['abs_strike_diff'] = df_merged['strike_diff'].abs()

      # Initialize necessary columns
      df_merged['long_spread_price'] = np.nan
      df_merged['short_spread_price'] = np.nan
      df_merged['long_idxs_buy'] = np.nan
      df_merged['long_idxs_sell'] = np.nan
      df_merged['short_idxs_buy'] = np.nan
      df_merged['short_idxs_sell'] = np.nan
      df_merged['long_szs'] = np.nan
      df_merged['short_szs'] = np.nan

      # Create masks for conditions
      mask_typ_C = df_merged['typ_1'] == 'C'
      mask_typ_P = df_merged['typ_1'] == 'P'
      mask_strike_diff_neg = df_merged['strike_diff'] < 0
      mask_strike_diff_pos = df_merged['strike_diff'] > 0

      # Case 1: Call options where strike_diff < 0
      mask_case1 = mask_typ_C & mask_strike_diff_neg
      df_merged.loc[mask_case1, 'long_spread_price'] = df_merged.loc[mask_case1, 'ask_px_00_1'] - df_merged.loc[mask_case1, 'bid_px_00_2']
      df_merged.loc[mask_case1, 'short_spread_price'] = -df_merged.loc[mask_case1, 'bid_px_00_1'] + df_merged.loc[mask_case1, 'ask_px_00_2']
      df_merged.loc[mask_case1, 'long_idxs_buy'] = df_merged.loc[mask_case1, 'idx_1_1']
      df_merged.loc[mask_case1, 'long_idxs_sell'] = df_merged.loc[mask_case1, 'idx_1_2']
      df_merged.loc[mask_case1, 'short_idxs_buy'] = df_merged.loc[mask_case1, 'idx_1_2']
      df_merged.loc[mask_case1, 'short_idxs_sell'] = df_merged.loc[mask_case1, 'idx_1_1']
      df_merged.loc[mask_case1, 'long_szs'] = df_merged.loc[mask_case1, ['ask_sz_00_1', 'bid_sz_00_2']].min(axis=1)
      df_merged.loc[mask_case1, 'short_szs'] = df_merged.loc[mask_case1, ['bid_sz_00_1', 'ask_sz_00_2']].min(axis=1)

      # Case 2: Call options where strike_diff > 0
      mask_case2 = mask_typ_C & mask_strike_diff_pos
      df_merged.loc[mask_case2, 'long_spread_price'] = df_merged.loc[mask_case2, 'ask_px_00_2'] - df_merged.loc[mask_case2, 'bid_px_00_1']
      df_merged.loc[mask_case2, 'short_spread_price'] = -df_merged.loc[mask_case2, 'bid_px_00_2'] + df_merged.loc[mask_case2, 'ask_px_00_1']
      df_merged.loc[mask_case2, 'long_idxs_buy'] = df_merged.loc[mask_case2, 'idx_1_2']
      df_merged.loc[mask_case2, 'long_idxs_sell'] = df_merged.loc[mask_case2, 'idx_1_1']
      df_merged.loc[mask_case2, 'short_idxs_buy'] = df_merged.loc[mask_case2, 'idx_1_1']
      df_merged.loc[mask_case2, 'short_idxs_sell'] = df_merged.loc[mask_case2, 'idx_1_2']
      df_merged.loc[mask_case2, 'long_szs'] = df_merged.loc[mask_case2, ['bid_sz_00_1', 'ask_sz_00_2']].min(axis=1)
      df_merged.loc[mask_case2, 'short_szs'] = df_merged.loc[mask_case2, ['ask_sz_00_1', 'bid_sz_00_2']].min(axis=1)

      # Case 3: Put options where strike_diff < 0
      mask_case3 = mask_typ_P & mask_strike_diff_neg
      df_merged.loc[mask_case3, 'long_spread_price'] = -df_merged.loc[mask_case3, 'bid_px_00_1'] + df_merged.loc[mask_case3, 'ask_px_00_2']
      df_merged.loc[mask_case3, 'short_spread_price'] = df_merged.loc[mask_case3, 'ask_px_00_1'] - df_merged.loc[mask_case3, 'bid_px_00_2']
      df_merged.loc[mask_case3, 'long_idxs_buy'] = df_merged.loc[mask_case3, 'idx_1_2']
      df_merged.loc[mask_case3, 'long_idxs_sell'] = df_merged.loc[mask_case3, 'idx_1_1']
      df_merged.loc[mask_case3, 'short_idxs_buy'] = df_merged.loc[mask_case3, 'idx_1_1']
      df_merged.loc[mask_case3, 'short_idxs_sell'] = df_merged.loc[mask_case3, 'idx_1_2']
      df_merged.loc[mask_case3, 'long_szs'] = df_merged.loc[mask_case3, ['bid_sz_00_1', 'ask_sz_00_2']].min(axis=1)
      df_merged.loc[mask_case3, 'short_szs'] = df_merged.loc[mask_case3, ['ask_sz_00_1', 'bid_sz_00_2']].min(axis=1)

      # Case 4: Put options where strike_diff > 0
      mask_case4 = mask_typ_P & mask_strike_diff_pos
      df_merged.loc[mask_case4, 'long_spread_price'] = -df_merged.loc[mask_case4, 'ask_px_00_2'] + df_merged.loc[mask_case4, 'ask_px_00_1']
      df_merged.loc[mask_case4, 'short_spread_price'] = df_merged.loc[mask_case4, 'ask_px_00_2'] - df_merged.loc[mask_case4, 'bid_px_00_1']
      df_merged.loc[mask_case4, 'long_idxs_buy'] = df_merged.loc[mask_case4, 'idx_1_1']
      df_merged.loc[mask_case4, 'long_idxs_sell'] = df_merged.loc[mask_case4, 'idx_1_2']
      df_merged.loc[mask_case4, 'short_idxs_buy'] = df_merged.loc[mask_case4, 'idx_1_2']
      df_merged.loc[mask_case4, 'short_idxs_sell'] = df_merged.loc[mask_case4, 'idx_1_1']
      df_merged.loc[mask_case4, 'long_szs'] = df_merged.loc[mask_case4, ['ask_sz_00_1', 'bid_sz_00_2']].min(axis=1)
      df_merged.loc[mask_case4, 'short_szs'] = df_merged.loc[mask_case4, ['bid_sz_00_1', 'ask_sz_00_2']].min(axis=1)

      # Calculate profits
      df_merged['long_profit'] = df_merged['abs_strike_diff'] - df_merged['long_spread_price']
      df_merged['short_profit'] = -df_merged['abs_strike_diff'] - df_merged['short_spread_price']

      # Determine the more profitable spread type
      mask_long_better = df_merged['long_profit'] >= df_merged['short_profit']
      df_merged['profit'] = np.where(mask_long_better, df_merged['long_profit'], df_merged['short_profit'])
      df_merged['buy_idx'] = np.where(mask_long_better, df_merged['long_idxs_buy'], df_merged['short_idxs_buy']).astype(int)
      df_merged['sell_idx'] = np.where(mask_long_better, df_merged['long_idxs_sell'], df_merged['short_idxs_sell']).astype(int)
      df_merged['max_size'] = np.where(mask_long_better, df_merged['long_szs'], df_merged['short_szs']).astype(int)

      df_merged['adj_profit'] = np.where(mask_long_better, df_merged['profit'] / df_merged['long_spread_price'] * df_merged['max_size'],
                                                           df_merged['profit'] / df_merged['abs_strike_diff'] * df_merged['max_size'] )

      # Select relevant columns
      df_results = df_merged[[
          'strike_1', 'strike_2', 'strike_diff', 'long_spread_price', 'short_spread_price',
          'long_profit', 'short_profit', 'profit','adj_profit', 'buy_idx', 'sell_idx', 'max_size', 'typ_1'
      ]].rename(columns={'typ_1': 'typ'})

      # Sort and remove duplicates
      df_results = df_results.sort_values(by=['profit','adj_profit','strike_diff'], ascending=[False,False,True])
      df_results = df_results.drop_duplicates(subset=['buy_idx'], keep='first')
      df_results = df_results.drop_duplicates(subset=['sell_idx'], keep='first')

      # Reset index for clarity
      df_results.reset_index(drop=True, inplace=True)

      return df_results

  def generate_orders(self) -> pd.DataFrame:
    #given that we are supposed to trade options day by day, we iterate over every day in the orders

    def convert_timestamp_utc(x,tz = timedelta(hours = 5)):
      return datetime.strptime(x[:19], "%Y-%m-%d %H:%M:%S") + tz

    #get daily open price, given that the options will be settled on open price on third friday of the week
    daily_underlying_open = self.underlying[self.underlying['date'].apply(convert_timestamp_utc).apply(lambda x: x.hour) == 14]
    daily_underlying_open['datetime'] = pd.to_datetime(daily_underlying_open['date'])
    daily_underlying_rolling_stds = daily_underlying_open['open'].rolling(window = 10).std()



    all_orders = []

    for day, group in self.options.groupby('day'):
       current_date = pd.to_datetime(day)
       print('current date: %s'%current_date.date())
       current_expiry = Strategy.third_friday_of_month(current_date)
       current_expiry_str = current_expiry.strftime('%y%m%d')

       current_time_to_expiry = current_expiry - current_date
       if current_time_to_expiry.days <= 0:
        current_expiry = Strategy.third_friday_of_month(current_date + timedelta(days = 15))
        current_expiry_str = current_expiry.strftime('%y%m%d')
        current_time_to_expiry = current_expiry - current_date

       #get trailing daily open price stds
       #get current idx
       current_idx = daily_underlying_open['datetime'].apply(lambda x: x.date()) == current_date.date()
       current_rolling_std = daily_underlying_rolling_stds.loc[current_idx].iloc[0]
       current_underlying_open = daily_underlying_open.loc[current_idx]['open'].iloc[0]

       #trade options that expires within 4 days

       #if current_time_to_expiry > timedelta(days=4) or current_time_to_expiry.days <= 0:
       #   continue

       if np.isnan(current_rolling_std):
        continue

       #scaled it to match the remaining expiry
       current_std_scaled = (current_rolling_std * (current_time_to_expiry / timedelta(days = 1))**0.5)

       strike_upper = current_underlying_open + 3 * current_std_scaled
       strike_lower = current_underlying_open - 3 * current_std_scaled

       temp = group['symbol'].apply(Strategy.parse_option_symbol)
       group['expiry']= temp.apply(lambda x: x[0])
       group['typ'] = temp.apply(lambda x: x[1])
       group['strike']= temp.apply(lambda x: x[2])
       #generate orders for out of money calls

       itm_put_orders = group[(group['strike'] >= strike_upper) &
                               (group['typ'] == 'P') &
                               (group['expiry'] == current_expiry_str)]


       itm_call_orders = group[(group['strike'] <= strike_lower) &
                               (group['typ'] == 'C') &
                               (group['expiry'] == current_expiry_str)]

       pairs_call = Strategy.get_best_spreads(itm_call_orders).iloc[:2]
       pairs_put = Strategy.get_best_spreads(itm_put_orders).iloc[:2]

       if pairs_call.size != 0 and np.sum(pairs_call['profit'] > 0) !=  0:
        best_pairs_call = pairs_call.loc[pairs_call['adj_profit'].idxmax()]

        buy_orders = itm_call_orders.loc[best_pairs_call['buy_idx']]
        buy_orders['action'] = 'B'
        buy_orders['order_size'] = best_pairs_call['max_size']

        sell_orders = itm_call_orders.loc[best_pairs_call['sell_idx']]
        sell_orders['action'] = 'S'
        sell_orders['order_size'] = best_pairs_call['max_size']

        all_orders.append(buy_orders[['ts_recv','symbol', 'action', 'order_size']])
        all_orders.append(sell_orders[['ts_recv','symbol', 'action', 'order_size']])

       '''
       if pairs_put.size != 0 and np.sum(pairs_put['profit'] > 0) !=  0:
        best_pairs_put = pairs_put.loc[pairs_put['adj_profit'].idxmax()]

        buy_orders = itm_put_orders.loc[best_pairs_put['buy_idx']]
        buy_orders['action'] = 'B'
        buy_orders['order_size'] = best_pairs_put['max_size']

        sell_orders = itm_put_orders.loc[best_pairs_put['sell_idx']]
        sell_orders['action'] = 'S'
        sell_orders['order_size'] = best_pairs_put['max_size']

        all_orders.append(buy_orders[['ts_recv','symbol', 'action', 'order_size']])
        all_orders.append(sell_orders[['ts_recv','symbol', 'action', 'order_size']])
       '''




    result = pd.DataFrame(all_orders)
    result.rename(columns = {
                             'ts_recv': 'datetime',
                             'symbol': 'option_symbol'}, inplace = True)
    #temp change
    return result.reset_index(drop = True)

