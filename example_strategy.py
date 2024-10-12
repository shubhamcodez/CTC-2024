import pandas as pd
from datetime import datetime, timedelta

class Strategy:
  
    def __init__(self) -> None:
        self.capital: float = 100_000_000
        self.portfolio_value: float = 0

        self.start_date: datetime = datetime(2024, 1, 1)
        self.end_date: datetime = datetime(2024, 3, 30)
  
        self.options: pd.DataFrame = pd.read_csv("data/options_data.csv")
        self.options["day"] = self.options["ts_recv"].apply(lambda x: x.split("T")[0])

        self.underlying = pd.read_csv("data/underlying_data_hour.csv")
        self.underlying.columns = self.underlying.columns.str.lower()

        # Convert date column to datetime
        self.underlying['date'] = pd.to_datetime(self.underlying['date'])

        self.positions = []  # Store positions generated from the create_positions method
        self.current_positions = {}  # Dictionary to track current positions and their entry prices

        # Risk management parameters
        self.max_loss_percentage = 0.1  # Set max loss at 10% of capital
        self.days_before_expiry_to_close = 2  # Number of days to close positions before expiry

    def place_order(self, order_type: str, datetime: str, option_symbol: str, action: str, order_size: int, option_kind: str):
        """Helper function to create an order dictionary."""
        return {
            "datetime": datetime,
            "option_symbol": option_symbol,
            "action": action,
            "order_size": order_size,
            "order_type": order_type,  # Include order type for reference
            "option_kind": option_kind   # Indicates if it's a Put or Call
        }

    def create_positions(self):
        prices_df = self.underlying  # Using underlying data as prices for this example
        daily_prices = prices_df.groupby(prices_df['date'].dt.date)

        for date, daily_data in daily_prices:
            opening_price = daily_data.iloc[0]['open']
            date_with_tz = pd.to_datetime(date).tz_localize('UTC')

            valid_options = self.options[(
                self.options['expiry'] > date_with_tz) & 
                (self.options['expiry'] != date_with_tz + timedelta(days=1))
            ] 

            # Filter options
            put_options = valid_options[valid_options['type'] == 'Put']
            call_options = valid_options[valid_options['type'] == 'Call']
            strikes = put_options['strike'].sort_values().unique()

            # Check for Short Iron Condor setup
            if len(strikes) >= 4:
                long_put_1 = strikes[-1]  # Lowest strike for the put bought
                short_put_1 = strikes[-2]  # Highest strike for the put sold
                short_call_1 = strikes[-3]  # Lowest strike for the call sold
                long_call_2 = strikes[-4]    # Highest strike for the call bought

                # Create orders for Short Iron Condor
                short_iron_condor_orders = [
                    self.place_order("Short Iron Condor", put_options[put_options['strike'] == short_put_1].iloc[0]['ts_recv'], short_put_1, "Sell", 1, "Put"),
                    self.place_order("Short Iron Condor", call_options[call_options['strike'] == short_call_1].iloc[0]['ts_recv'], short_call_1, "Sell", 1, "Call"),
                    self.place_order("Short Iron Condor", put_options[put_options['strike'] == long_put_1].iloc[0]['ts_recv'], long_put_1, "Buy", 1, "Put"),
                    self.place_order("Short Iron Condor", call_options[call_options['strike'] == long_call_2].iloc[0]['ts_recv'], long_call_2, "Buy", 1, "Call")
                ]

                self.positions.append({
                    'date': date,
                    'type': 'Short Iron Condor',
                    'orders': short_iron_condor_orders,
                    'opening_price': opening_price
                })

                # Track current positions
                for order in short_iron_condor_orders:
                    self.current_positions[order["option_symbol"]] = {
                        "order_size": order["order_size"],
                        "entry_price": opening_price
                    }

            # Check for Long Straddle setup
            if len(call_options) > 0 and len(put_options) > 0:
                atm_strike = min(strikes, key=lambda x: abs(x - opening_price))

                # Create orders for Long Straddle
                long_straddle_orders = [
                    self.place_order("Long Straddle", put_options[put_options['strike'] == atm_strike].iloc[0]['ts_recv'], atm_strike, "Buy", 1, "Put"),
                    self.place_order("Long Straddle", call_options[call_options['strike'] == atm_strike].iloc[0]['ts_recv'], atm_strike, "Buy", 1, "Call")
                ]

                self.positions.append({
                    'date': date,
                    'type': 'Long Straddle',
                    'orders': long_straddle_orders,
                    'opening_price': opening_price
                })

                # Track current positions
                for order in long_straddle_orders:
                    self.current_positions[order["option_symbol"]] = {
                        "order_size": order["order_size"],
                        "entry_price": opening_price
                    }

            # Risk Management Checks
            self.close_positions()

    def close_positions(self):
        for option_symbol, details in list(self.current_positions.items()):
            entry_price = details["entry_price"]
            current_price = self.get_current_price(option_symbol)  # Fetch current price using timestamps

            # Check if the loss threshold is exceeded
            if (entry_price - current_price) / entry_price > self.max_loss_percentage:
                print(f"Closing position for {option_symbol} due to exceeded loss threshold.")
                self.close_order(option_symbol)

            # Check if the position should be closed before expiry
            expiration_date = self.get_expiration_date(option_symbol)
            if (expiration_date - datetime.now()).days < self.days_before_expiry_to_close:
                print(f"Closing position for {option_symbol} before expiration.")
                self.close_order(option_symbol)

    def close_order(self, option_symbol):
        order_size = self.current_positions[option_symbol]["order_size"]
        # Implement logic to close the order here
        print(f"Closing order: {option_symbol} with size: {order_size}")
        # Reset the position after closing
        self.current_positions.pop(option_symbol)

    def get_current_price(self, option_symbol):
        """Fetch the current price of the option using the timestamp of the order."""
        timestamp = pd.to_datetime(option_symbol)  # Use option_symbol as timestamp
        # Find the price closest to the timestamp in the underlying data
        price_row = self.underlying.loc[
            self.underlying['date'].dt.date <= timestamp.date()
        ].iloc[-1]  # Get the last price before or at the timestamp
        return price_row['close']  # Return the close price

    def get_expiration_date(self, option_symbol):
        """Assumes the expiration date is available in the options DataFrame."""
        # Get expiration date from options data
        return pd.to_datetime(self.options[self.options['symbol'] == option_symbol]['expiry'].iloc[0])  # Placeholder for expiration date retrieval

    def generate_orders(self) -> pd.DataFrame:
        orders = []

        # Iterate through all positions and their corresponding orders
        for position in self.positions:
            for order in position['orders']:
                # Retrieve details from the order
                option_symbol = order["option_symbol"]  # Unique identifier for the option
                expiration_date = self.get_expiration_date(option_symbol)  # Get the expiration date
                formatted_expiration_date = expiration_date.strftime("%Y%m%d")  # Format the date to YYYYMMDD
                option_type = order["option_kind"]  # Option type (Call or Put)
                strike_price = self.options[self.options['symbol'] == option_symbol]['strike'].iloc[0]  # Get strike price from options data

                # Construct the option symbol in the required format
                formatted_option_symbol = f"SPX{formatted_expiration_date}{option_type[0].upper()}{int(strike_price)}"

                # Append each order to the orders list with necessary attributes
                orders.append({
                    "Datetime": order["datetime"],            # Timestamp of the order
                    "Option Symbol": formatted_option_symbol,  # The formatted option symbol
                    "Action": order["action"],                # Action to take: "Buy" or "Sell"
                    "Order Size": order["order_size"]         # Size of the order
                })

        # Create a DataFrame from the orders list
        return pd.DataFrame(orders)