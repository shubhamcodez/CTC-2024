# CTC 2024 Derivatives Case Strategy

## Cornell Trading Competition - Derivatives Case

The Cornell Trading Competition (CTC) 2024 challenges participants to craft a trading strategy for European-style SPX options. The objective is to manage portfolio risk across diverse market conditions, aiming for consistent positive returns while showcasing effective hedging and risk management.

### Key Competition Details

- **Initial Capital**: $100,000,000
- **Underlying Asset**: S&P 500 Index
- **Instruments**: SPX Options (European-style, cash-settled)
- **Contract Multiplier**: 100
- **Expiration**: 3rd Friday of each month
- **Training Data**: 3 months of SPX options NBBO data and SPY data
- **Testing**: Three distinct 3-month periods (varying by average VIX)

### Evaluation Metrics

1. Total Return
2. Max Drawdown
3. Sharpe Ratio
4. Consistency Score
5. Final Score (combines total return, risk management, and consistency)

## Our Strategy

Our approach targets pricing inefficiencies in deep in-the-money (ITM) options while maintaining a balanced risk profile:

1. **Daily Analysis**: Calculate the underlying's rolling standard deviation to estimate expected price movements.

2. **Strike Selection**: Focus on options with strikes 3 standard deviations from the current price, targeting deep ITM options.

3. **Spread Creation**:
   - **Calls**: Buy lower strike, sell higher strike.
   - **Puts**: Buy higher strike, sell lower strike.

4. **Profit Calculation**: Assess potential profit for each spread, considering both absolute profit and profit adjusted for spread price and maximum trade size.

5. **Trade Execution**: Select and execute the most profitable spreads (up to two each for calls and puts), adhering to order size limits and margin requirements.

6. **Risk Management**:
   - Focus on ITM options to mitigate the risk of worthless expiration.
   - Use spreads to cap maximum loss.
   - Avoid trading options close to expiration (within 4 days).

7. **Consistency**: Apply this process daily across various market conditions to achieve consistent returns while effectively managing risk.

Our strategy balances profit potential with rigorous risk management, aligning with the competition's goal of consistent positive returns rather than high-risk approaches.

## Implementation

The core strategy is encapsulated in the `Strategy` class within the provided Python script. Key methods include:

- `parse_option_symbol`: Extracts option details from the symbol.
- `third_friday_of_month`: Calculates expiration dates.
- `get_best_spreads`: Identifies the most profitable option spreads.
- `generate_orders`: Analyzes the market and generates trading orders.

To execute the strategy:
1. Ensure required data files (`cleaned_options_data.csv` and `underlying_data_hour.csv`) are in the `data/` directory.
2. Run the main script.

## Performance Results

### Our Performance

| Metric          | Our Strategy |
|-----------------|--------------|
| Max Drawdown    | 0.01335554   |
| Overall Return  | 98.9015495%  |
| Sharpe Ratio    | 884.6000     |
| Overall Score   | 6550778.4846 |

### Baseline Performance

| Metric          | Baseline     |
|-----------------|--------------|
| Max Drawdown    | 2.65715524   |
| Overall Return  | -367.179916% |
| Sharpe Ratio    | 0.03547925   |
| Overall Score   | -4.90271252  |

Our strategy significantly outperforms the baseline across all metrics, demonstrating its effectiveness in managing risk while generating substantial returns.

## Repository Structure

- `data/`: Contains options data, S&P underlying prices, and a sample strategy output.
- `backtester.py`: Code for backtesting strategies.
- `example_strategy.py`: Reference implementation for strategy structure.
- `starter_code.py`: Skeleton code for strategy development.
- `run_backtester.ipynb`: Jupyter notebook for importing strategies and running the backtester.

## Note on Allowed Libraries

The following libraries are permitted for use in your strategy:

- numpy
- matplotlib
- seaborn
- pandas
- scipy
- statsmodels

*Important*: The usage of machine learning and deep learning techniques is not permitted in this competition.

## Disclaimer

This strategy is designed for the Cornell Trading Competition and may not be suitable for real-world trading without further refinement and risk assessment. Always conduct thorough research and consider seeking professional advice before engaging in options trading.
