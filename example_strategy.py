import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import norm
from scipy.integrate import quad
from scipy.optimize import minimize
import pytz
from tqdm import tqdm

class Strategy:
    def __init__(self) -> None:
        self.capital: float = 100_000_000
        self.portfolio_value: float = 0
        self.start_date: datetime = datetime(2024, 1, 1, tzinfo=pytz.UTC)
        self.end_date: datetime = datetime(2024, 3, 30, tzinfo=pytz.UTC)
        
        # Load and preprocess data
        self.load_and_preprocess_data()
        
        # Initialize Heston parameters
        self.kappa = 2.0
        self.theta = 0.04
        self.sigma = 0.3
        self.rho = -0.7
        self.v0 = 0.04

        # Calibrate Heston parameters using historical data
        self.calibrate_heston_parameters()

        self.positions = {}
        self.orders = []

    def load_and_preprocess_data(self):
        # Load options data
        self.options = pd.read_csv("data/cleaned_options_data.csv")
        self.options["ts_recv"] = pd.to_datetime(self.options["ts_recv"], utc=True)
        
        # Extract information from symbol
        self.options['expiry'] = pd.to_datetime(self.options['symbol'].str.extract('(\d{6})')[0], format='%y%m%d', utc=True)
        self.options['type'] = self.options['symbol'].str[-9].map({'C': 'Call', 'P': 'Put'})
        self.options['strike'] = self.options['symbol'].str[-8:].astype(float) / 1000
        
        # Calculate days till expiry
        self.options['days_till_expiry'] = (self.options['expiry'] - self.options['ts_recv']).dt.days
        
        # Create a 'day' column for filtering
        self.options["day"] = self.options["ts_recv"].dt.date
        
        # Load underlying data
        self.underlying = pd.read_csv("data/spx_minute_level_data_jan_mar_2024.csv")
        self.underlying.columns = ["ms_of_day", "price", "date"]
        self.underlying["date"] = pd.to_datetime(self.underlying["date"].astype(str), format="%Y%m%d")
        self.underlying["time"] = pd.to_timedelta(self.underlying["ms_of_day"], unit='ms')
        self.underlying["datetime"] = (self.underlying["date"] + self.underlying["time"]).dt.tz_localize('UTC')
        self.underlying = self.underlying.sort_values("datetime")

    def calibrate_heston_parameters(self):
        print("Calibrating Heston model parameters using historical data...")
        
        # Use data up to the start date for calibration
        historical_options = self.options[self.options['ts_recv'] < self.start_date]
        
        if historical_options.empty:
            print("No historical data available for calibration. Using default parameters.")
            return
        
        # Select a subset of historical options for calibration
        calibration_options = historical_options.sample(min(1000, len(historical_options)))
        
        # Define the objective function
        def objective(params):
            self.kappa, self.theta, self.sigma, self.rho, self.v0 = params
            total_error = 0
            for _, option in calibration_options.iterrows():
                S = self.get_underlying_price(option['ts_recv'])
                K = option['strike']
                T = (option['expiry'] - option['ts_recv']).total_seconds() / (365 * 24 * 3600)
                r = 0.03  # Assumed risk-free rate
                market_price = (option['bid_px_00'] + option['ask_px_00']) / 2
                model_price = self.heston_price(S, K, T, r, option['type'])
                total_error += (market_price - model_price)**2
            return total_error

        # Initial guess and bounds
        initial_guess = [self.kappa, self.theta, self.sigma, self.rho, self.v0]
        bounds = [(0.1, 10), (0.01, 0.5), (0.1, 1), (-0.99, 0.99), (0.01, 0.5)]

        # Perform the optimization
        result = minimize(objective, initial_guess, method='L-BFGS-B', bounds=bounds)

        # Update the parameters
        self.kappa, self.theta, self.sigma, self.rho, self.v0 = result.x

        print(f"Calibrated parameters: kappa={self.kappa:.4f}, theta={self.theta:.4f}, "
              f"sigma={self.sigma:.4f}, rho={self.rho:.4f}, v0={self.v0:.4f}")

    def heston_characteristic_function(self, u, S, K, T, r, v):
        i = complex(0, 1)
        
        a = self.kappa * self.theta
        b = self.kappa - self.rho * self.sigma * i * u
        
        d = np.sqrt(b**2 + self.sigma**2 * (u**2 + i*u))
        g = (b - d) / (b + d)
        
        C = r * i * u * T + a / self.sigma**2 * (
            (b - d) * T - 2 * np.log((1 - g * np.exp(-d * T)) / (1 - g))
        )
        D = (b - d) / self.sigma**2 * ((1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T)))
        
        return np.exp(C + D * v + i * u * np.log(S))

    def heston_price(self, S, K, T, r, option_type):
        v = self.v0  # Use the calibrated initial variance
        i = complex(0, 1)  # Define the imaginary unit
    
        def integrand(u, S, K, T, r, v, flag):
            if flag:
                return np.real(np.exp(-i*u*np.log(K)) * self.heston_characteristic_function(u-i, S, K, T, r, v) / (i*u))
            else:
                return np.real(np.exp(-i*u*np.log(K)) * self.heston_characteristic_function(u, S, K, T, r, v) / (i*u))
    
        P1 = 0.5 + 1/np.pi * quad(integrand, 0, 100, args=(S, K, T, r, v, True))[0]
        P2 = 0.5 + 1/np.pi * quad(integrand, 0, 100, args=(S, K, T, r, v, False))[0]
    
        if option_type == "Call":
            return S * P1 - K * np.exp(-r * T) * P2
        else:
            return K * np.exp(-r * T) * (1 - P2) - S * (1 - P1)

    def calculate_delta(self, S, K, T, r, option_type):
        epsilon = 0.0001
        price1 = self.heston_price(S - epsilon, K, T, r, option_type)
        price2 = self.heston_price(S + epsilon, K, T, r, option_type)
        return (price2 - price1) / (2 * epsilon)

    def get_underlying_price(self, datetime):
        # Convert the input datetime to UTC to match the underlying data format
        datetime_utc = datetime.astimezone(pytz.UTC)
        
        # Select the price based on the most recent datetime
        price = self.underlying[self.underlying["datetime"] <= datetime_utc]["price"].iloc[-1]
        return price

    def generate_orders(self) -> pd.DataFrame:
        for _, option in tqdm(self.options.iterrows(), total=self.options.shape[0]):
            ts_recv = option['ts_recv']
            underlying_price = self.get_underlying_price(ts_recv)
            strike = option["strike"]
            T = (option["expiry"] - ts_recv).total_seconds() / (365 * 24 * 3600)  # in years
            market_price = (option['bid_px_00'] + option['ask_px_00']) / 2
            theoretical_price = self.heston_price(underlying_price, strike, T, 0.03, option["type"])

            # Buy condition
            if market_price < theoretical_price * 0.8:  # If market price is less than 80% of theoretical
                order = {
                    "symbol": option['symbol'],
                    "type": option["type"],
                    "strike": strike,
                    "expiry": option['expiry'],
                    "market_price": market_price,
                    "theoretical_price": theoretical_price,
                    "action": "B"  # Buy
                }
                self.orders.append(order)

            # Sell condition
            if option['symbol'] in self.positions and market_price > theoretical_price * 1.2:
                order = {
                    "symbol": option['symbol'],
                    "type": option["type"],
                    "strike": strike,
                    "expiry": option['expiry'],
                    "market_price": market_price,
                    "theoretical_price": theoretical_price,
                    "action": "S"  # Sell
                }
                self.orders.append(order)

        # Save orders to a CSV file
        orders_df = pd.DataFrame(self.orders)
        orders_df.to_csv("generated_orders.csv", index=False)

        return orders_df

# Example usage:
if __name__ == "__main__":
    strategy = Strategy()
    orders = strategy.generate_orders()
    print(orders)
