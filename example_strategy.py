import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import norm
from scipy.integrate import quad
from scipy.optimize import minimize
import pytz

class Strategy:
    def __init__(self) -> None:
        self.capital: float = 100_000_000
        self.portfolio_value: float = 0
        self.start_date: datetime = datetime(2024, 1, 1, tzinfo=pytz.UTC)
        self.end_date: datetime = datetime(2024, 1, 3, tzinfo=pytz.UTC)
        
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
        self.options["ts_recv"] = pd.to_datetime(self.options["ts_recv"])
        self.options["day"] = self.options["ts_recv"].dt.date
        
        # Updated symbol parsing
        def parse_symbol(symbol):
            try:
                expiration = pd.to_datetime(symbol[3:9], format="%y%m%d")
                strike = float(symbol[9:-1]) / 1000
                option_type = 'Call' if symbol[-1] == 'C' else 'Put'
                return pd.Series([expiration, strike, option_type])
            except:
                return pd.Series([pd.NaT, np.nan, ''])

        symbol_data = self.options["symbol"].apply(parse_symbol)
        self.options[["expiration", "strike", "option_type"]] = symbol_data
        
        # Load underlying data
        self.underlying = pd.read_csv("data/spx_minute_level_data_jan_mar_2024.csv")
        self.underlying.columns = ["ms_of_day", "price", "date"]
        self.underlying["date"] = pd.to_datetime(self.underlying["date"].astype(str), format="%Y%m%d")
        self.underlying["time"] = pd.to_timedelta(self.underlying["ms_of_day"], unit='ms')
        self.underlying["datetime"] = (self.underlying["date"] + self.underlying["time"])
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
                T = (option['expiration'] - option['ts_recv']).total_seconds() / (365 * 24 * 3600)
                r = 0.03  # Assumed risk-free rate
                market_price = (option['bid_px_00'] + option['ask_px_00']) / 2
                model_price = self.heston_price(S, K, T, r, option['option_type'])
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
        price = self.underlying[self.underlying["datetime"] <= datetime]["price"].iloc[-1]
        return price

    def generate_orders(self) -> pd.DataFrame:
        for date in pd.date_range(self.start_date, self.end_date):
            date = date.replace(tzinfo=pytz.UTC)
            day_options = self.options[self.options["day"] == date.date()]
            if day_options.empty:
                continue
            
            grouped_options = day_options.groupby("ts_recv")
            
            for ts_recv, options_group in grouped_options:
                underlying_price = self.get_underlying_price(ts_recv)
                
                target_expiry = date + timedelta(days=30)
                eligible_options = options_group[
                    (options_group["expiration"] > target_expiry) & 
                    (options_group["expiration"] <= target_expiry + timedelta(days=15)) &
                    (options_group["strike"].between(0.95 * underlying_price, 1.05 * underlying_price))
                ]
                
                if eligible_options.empty:
                    continue
                
                call = eligible_options[eligible_options["option_type"] == "Call"].iloc[0]
                put = eligible_options[eligible_options["option_type"] == "Put"].iloc[0]
                
                for option in [call, put]:
                    symbol = option["symbol"]
                    option_type = option["option_type"]
                    strike = option["strike"]
                    expiry = option["expiration"]
                    T = (expiry - date).total_seconds() / (365 * 24 * 3600)
                    r = 0.03  # Assumed risk-free rate
                    
                    theoretical_price = self.heston_price(underlying_price, strike, T, r, option_type)
                    market_price = (option["bid_px_00"] + option["ask_px_00"]) / 2
                    
                    if theoretical_price > market_price * 1.01:
                        action = "B"
                        size = min(10, int(option["ask_sz_00"]))
                    elif theoretical_price < market_price * 0.99:
                        action = "S"
                        size = min(10, int(option["bid_sz_00"]))
                    else:
                        continue
                    
                    self.orders.append({
                        "datetime": ts_recv,
                        "option_symbol": symbol,
                        "action": action,
                        "order_size": size
                    })
                    
                    if symbol not in self.positions:
                        self.positions[symbol] = 0
                    self.positions[symbol] += size if action == "B" else -size
                
                total_delta = 0
                for symbol, position in self.positions.items():
                    option = self.options[self.options["symbol"] == symbol].iloc[0]
                    delta = self.calculate_delta(underlying_price, option["strike"], 
                                                 (option["expiration"] - date).total_seconds() / (365 * 24 * 3600), 
                                                 r, option["option_type"])
                    total_delta += position * delta * 100
                
                hedge_size = int(-total_delta / 100)
                if hedge_size != 0:
                    atm_call = eligible_options[(eligible_options["option_type"] == "Call") & 
                                                (eligible_options["strike"] == eligible_options["strike"].min())].iloc[0]
                    atm_put = eligible_options[(eligible_options["option_type"] == "Put") & 
                                               (eligible_options["strike"] == eligible_options["strike"].min())].iloc[0]
                    
                    self.orders.append({
                        "datetime": ts_recv,
                        "option_symbol": atm_call["symbol"],
                        "action": "B" if hedge_size > 0 else "S",
                        "order_size": abs(hedge_size)
                    })
                    
                    self.orders.append({
                        "datetime": ts_recv,
                        "option_symbol": atm_put["symbol"],
                        "action": "S" if hedge_size > 0 else "B",
                        "order_size": abs(hedge_size)
                    })
        
        return pd.DataFrame(self.orders)

# Instantiate the strategy
strategy = Strategy()

# Generate orders
orders = strategy.generate_orders()

# Print the first few orders
print("\nFirst few generated orders:")
print(orders.head())

# Print some statistics about the orders
print("\nOrder statistics:")
print(f"Total number of orders: {len(orders)}")
print(f"Number of buy orders: {len(orders[orders['action'] == 'B'])}")
print(f"Number of sell orders: {len(orders[orders['action'] == 'S'])}")
print(f"Average order size: {orders['order_size'].mean():.2f}")