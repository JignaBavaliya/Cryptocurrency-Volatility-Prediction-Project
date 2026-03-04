# Cryptocurrency Volatility Prediction

This project builds a machine learning model to predict **future 7-day realised volatility** for cryptocurrencies using historical OHLC, volume, and market cap data.

## What this contains (as required in the assignment)
- Source code (modular Python scripts)
- Cleaned dataset + engineered features
- EDA report + plots
- HLD & LLD documents
- Pipeline architecture documentation
- Final report (method, metrics, insights)
- Trained model artifact
- Local deployment app (Streamlit)

## Quickstart
```bash
pip install -r requirements.txt
python -m src.train
python -m src.eda
streamlit run app.py
```

## Model target (volatility definition)
Forward-looking 7-day realised volatility:
std of daily log returns over the next 7 days.

## Latest metrics
- Best alpha (CV): 100.0
- Test RMSE: 0.146291
- Test MAE : 0.009767
- Test R²  : -2.901458
