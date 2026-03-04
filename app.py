import streamlit as st
import pandas as pd
import numpy as np
import joblib
from src.preprocess import load_data, clean_data
from src.features import add_features, add_target
from src.config import FEATURE_COLS, CAT_COLS, TARGET_COL

st.set_page_config(page_title="Crypto Volatility Predictor", layout="centered")
st.title("Cryptocurrency Volatility Prediction")
st.caption("Predicts forward-looking 7-day realised volatility using historical OHLC + liquidity features.")

root = "."
data_path = "data/dataset.csv"
model_path = "artifacts/volatility_ridge.joblib"

@st.cache_data
def load_prepared():
    df = load_data(data_path)
    df = clean_data(df)
    df = add_features(df)
    df = add_target(df, horizon=7)
    df_model = df.dropna(subset=FEATURE_COLS + [TARGET_COL]).copy()
    return df_model

@st.cache_resource
def load_model():
    return joblib.load(model_path)

df_model = load_prepared()
model = load_model()

cryptos = sorted(df_model["crypto_name"].unique())
crypto = st.selectbox("Select cryptocurrency", cryptos)

df_c = df_model[df_model["crypto_name"] == crypto].sort_values("date").copy()
min_date = df_c["date"].min().date()
max_date = df_c["date"].max().date()

date = st.date_input("Select date (must be within available range)", value=max_date, min_value=min_date, max_value=max_date)

row = df_c[df_c["date"].dt.date == date]
if row.empty:
    st.error("No row found for that date after feature/target creation. Choose another date.")
else:
    X = row[CAT_COLS + FEATURE_COLS]
    pred = float(model.predict(X)[0])
    st.metric("Predicted future 7-day volatility", f"{pred:.6f}")

    st.subheader("Row features used")
    st.dataframe(row[["date","crypto_name"] + FEATURE_COLS + [TARGET_COL]].tail(1))
