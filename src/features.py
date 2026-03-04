import numpy as np
import pandas as pd

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["crypto_name","date"]).reset_index(drop=True)
    g = df.groupby("crypto_name", group_keys=False)

    df["log_close"] = np.log(df["close"].clip(lower=1e-12))
    df["log_return_1"] = g["log_close"].diff()
    df["return_1"] = g["close"].pct_change()

    df["price_range"] = df["high"] - df["low"]
    df["price_range_pct"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)

    df["liq_ratio"] = df["volume"] / df["marketCap"].replace(0, np.nan)
    df["log_volume"] = np.log(df["volume"].clip(lower=1e-12))
    df["log_market_cap"] = np.log(df["marketCap"].clip(lower=1e-12))

    prev_close = g["close"].shift(1)
    tr1 = (df["high"] - df["low"]).abs()
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    df["true_range"] = np.maximum(tr1, np.maximum(tr2, tr3))
    df["atr_14"] = g["true_range"].rolling(14, min_periods=14).mean().reset_index(level=0, drop=True)

    for w in [7, 14, 30]:
        df[f"vol_return_{w}"] = g["log_return_1"].rolling(w, min_periods=w).std().reset_index(level=0, drop=True)
        df[f"liquidity_{w}"] = g["liq_ratio"].rolling(w, min_periods=w).mean().reset_index(level=0, drop=True)
        df[f"momentum_{w}"] = g["return_1"].rolling(w, min_periods=w).mean().reset_index(level=0, drop=True)

    mid20 = g["close"].rolling(20, min_periods=20).mean().reset_index(level=0, drop=True)
    std20 = g["close"].rolling(20, min_periods=20).std().reset_index(level=0, drop=True)
    upper = mid20 + 2*std20
    lower = mid20 - 2*std20
    df["bb_width_20"] = (upper - lower) / mid20.replace(0, np.nan)

    df["dow"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    return df

def add_target(df: pd.DataFrame, horizon: int = 7) -> pd.DataFrame:
    df = df.copy()
    g = df.groupby("crypto_name", group_keys=False)
    future_ret = g["log_return_1"].shift(-1)
    df["target_future_vol_7d"] = future_ret.groupby(df["crypto_name"]).rolling(horizon, min_periods=horizon).std().reset_index(level=0, drop=True)
    return df
