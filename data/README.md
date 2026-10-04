# Data Directory

This directory is used to cache downloaded market data and store any synthetic fallback datasets.

## Contents

- **price_data.csv** *(auto-generated)*: Cached adjusted closing prices downloaded via `yfinance`.
- **synthetic_returns.csv** *(auto-generated if needed)*: Synthetic return data used as a fallback when live data is unavailable.

## Data Source

Historical adjusted closing prices are fetched from Yahoo Finance via the `yfinance` library.

Default ticker symbols and weights:

| Ticker | Asset       | Weight |
|--------|-------------|--------|
| AAPL   | Apple Inc.  | 30%    |
| MSFT   | Microsoft   | 25%    |
| GOOGL  | Alphabet    | 20%    |
| AMZN   | Amazon      | 15%    |
| JPM    | JPMorgan    | 10%    |

## Notes

- Data is fetched for approximately 3 years of daily history.
- All prices are adjusted closing prices to account for dividends and splits.
- Missing values are forward-filled then back-filled.
- This data is for **educational purposes only** and is not intended for production use.
