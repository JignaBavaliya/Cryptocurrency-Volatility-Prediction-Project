# Exploratory Data Analysis (EDA) Report

## Dataset overview
- Rows (after cleaning + feature engineering): **71,240**
- Cryptocurrencies: **55**
- Date range: **2013-06-04** to **2022-10-16**

## Key checks
- OHLC consistency enforced (high/low vs open/close).
- `volume` and `marketCap` zeros treated as missing and imputed per crypto.

## Visualizations (see `plots/`)
- Records per crypto
- Target distribution (future 7d vol)
- Correlation heatmap
- Liquidity vs future vol
- (Optional) Bitcoin close + historical vol

## Observations (typical for crypto markets)
- Volatility is heavy-tailed: spikes exist (risk periods).
- Liquidity and volatility often show non-linear relationships.
