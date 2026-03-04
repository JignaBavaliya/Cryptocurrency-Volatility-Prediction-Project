# Low-Level Design (LLD)

## Folder structure
- `src/`
  - `config.py` : constants, feature lists
  - `preprocess.py` : cleaning + imputations + OHLC checks
  - `features.py` : feature engineering + target creation
  - `eda.py` : EDA plots + EDA report generation
  - `train.py` : training, walk-forward tuning, evaluation, artifact saving
- `artifacts/` : trained model + metrics json
- `plots/` : EDA images

## Function-level design

### preprocess.py
- `load_data(path) -> DataFrame`
- `clean_data(df) -> DataFrame`
  - parse dates
  - replace zeros with NaN (volume/marketCap)
  - per-crypto ffill/bfill and median fill
  - enforce OHLC constraints

### features.py
- `add_features(df) -> DataFrame`
  - groupby crypto rolling computations (vol, liquidity, momentum, ATR, Bollinger width)
- `add_target(df, horizon=7) -> DataFrame`
  - forward-looking realised volatility over next horizon days

### train.py
- `make_time_split(df, test_frac=0.2)`
- `walk_forward_cv_splits(train_df, n_splits=3)`
- `tune_ridge_alpha(X_train, y_train, splits, alphas)`
- `train_final_model(X_train, y_train, best_alpha)`
- `evaluate(model, X_test, y_test)`

### eda.py
- Generates:
  - records per crypto
  - target distribution histogram
  - correlation heatmap
  - liquidity vs volatility scatter
  - optional BTC series plots
