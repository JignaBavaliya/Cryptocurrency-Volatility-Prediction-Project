# High-Level Design (HLD)

## Goal
Predict cryptocurrency volatility levels from historical market data (OHLC, volume, market cap).

## System components
1. **Data Ingestion**
   - Input: CSV file (`data/dataset.csv`)
   - Output: typed dataframe with parsed dates and crypto identifiers

2. **Data Processing**
   - Replace invalid zeros in `volume` and `marketCap` with missing values
   - Time-consistent imputation per cryptocurrency (forward fill, back fill)
   - OHLC consistency validation

3. **Feature Engineering**
   - Returns & log returns
   - Rolling historical volatility (7/14/30)
   - Liquidity ratios (volume/marketCap) + rolling means
   - ATR(14), Bollinger width(20), momentum windows
   - Calendar features

4. **Modeling**
   - Time-based train/test split (no random shuffling)
   - Walk-forward cross validation on training dates
   - Ridge Regression with scaling + one-hot crypto identity

5. **Evaluation**
   - RMSE, MAE, R² on unseen dates

6. **Serving (Local)**
   - Streamlit app loads saved model and produces predictions for selected crypto/date.
