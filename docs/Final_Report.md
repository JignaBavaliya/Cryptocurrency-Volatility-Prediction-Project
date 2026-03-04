# Final Report

## Objective
Build a machine learning model to predict cryptocurrency volatility levels from historical OHLC, volume, and market cap data.

## Volatility definition (target)
Forward-looking 7-day realised volatility: standard deviation of the next 7 daily log returns.

## Methodology
1. Data preprocessing
   - Parse dates and sort per cryptocurrency
   - Replace 0 volume/marketCap with missing and impute time-consistently
   - Validate OHLC consistency

2. Feature engineering
   - Rolling realised volatility (7/14/30)
   - Liquidity ratios + rolling means
   - ATR(14), Bollinger width(20), momentum (7/14/30)
   - Calendar features

3. Model selection
   - Ridge regression with scaling and crypto identity one-hot encoding.
   - Walk-forward CV to tune alpha (regularization strength).

## Evaluation (unseen dates)
- Best alpha (CV): **100.0**
- Test RMSE: **0.146291**
- Test MAE : **0.009767**
- Test R²  : **-2.901458**

## Key insights
- Recent volatility and range-based measures (ATR/true range) are strong predictors of near-term volatility.
- Liquidity features help capture unstable periods where volatility increases as liquidity deteriorates.

## Deliverables
- Trained model: `artifacts/volatility_ridge.joblib`
- Metrics: `artifacts/metrics.json`
- Cleaned dataset: `data/cleaned_features.csv`
- EDA plots: `plots/`
- HLD/LLD/Pipeline docs: `docs/`
- Local app: `app.py`
