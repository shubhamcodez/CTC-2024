import pandas as pd
from datetime import datetime, timedelta

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

  def generate_orders(self) -> pd.DataFrame:
    orders = []
    daily_capital = self.capital * 0.10  # 10% of capital per day
    
    current_date = self.start_date
    while current_date <= self.end_date:
      # Filter options expiring within a week
      expiry_date = current_date + timedelta(days=7)
      daily_options = self.options[
        (self.options["day"] == current_date.date()) & 
        (self.options["expiration"] <= expiry_date) & 
        (self.options["expiration"] > current_date)
      ]
      
      if daily_options.empty:
        current_date += timedelta(days=1)
        continue
      
      # Get current SPX price
      current_spx = self.underlying[self.underlying["date"] == current_date.strftime("%Y-%m-%d")]["close"].iloc[0]
      
      # Select options for Strangle
      otm_call = daily_options[(daily_options["type"] == "call") & (daily_options["strike"] > current_spx)].sort_values("strike").iloc[0]
      otm_put = daily_options[(daily_options["type"] == "put") & (daily_options["strike"] < current_spx)].sort_values("strike", ascending=False).iloc[0]
      
      # Select options for Iron Condor
      further_otm_call = daily_options[(daily_options["type"] == "call") & (daily_options["strike"] > otm_call["strike"])].sort_values("strike").iloc[0]
      further_otm_put = daily_options[(daily_options["type"] == "put") & (daily_options["strike"] < otm_put["strike"])].sort_values("strike", ascending=False).iloc[0]
      
      # Calculate order sizes based on available capital and option prices
      strangle_capital = daily_capital * 0.5
      iron_condor_capital = daily_capital * 0.5
      
      strangle_size = int(strangle_capital / (otm_call["ask_px_00"] + otm_put["ask_px_00"]))
      iron_condor_size = int(iron_condor_capital / ((further_otm_call["ask_px_00"] - otm_call["bid_px_00"]) + (otm_put["bid_px_00"] - further_otm_put["ask_px_00"])))
      
      # Generate orders
      orders.extend([
        {"datetime": current_date, "option_symbol": otm_call["symbol"], "action": "B", "order_size": strangle_size},
        {"datetime": current_date, "option_symbol": otm_put["symbol"], "action": "B", "order_size": strangle_size},
        {"datetime": current_date, "option_symbol": otm_call["symbol"], "action": "S", "order_size": iron_condor_size},
        {"datetime": current_date, "option_symbol": further_otm_call["symbol"], "action": "B", "order_size": iron_condor_size},
        {"datetime": current_date, "option_symbol": otm_put["symbol"], "action": "S", "order_size": iron_condor_size},
        {"datetime": current_date, "option_symbol": further_otm_put["symbol"], "action": "B", "order_size": iron_condor_size},
      ])
      
      current_date += timedelta(days=1)
    
    return pd.DataFrame(orders)
