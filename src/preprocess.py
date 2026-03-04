import pandas as pd
import numpy as np

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    elif "timestamp" in df.columns:
        df["date"] = pd.to_datetime(df["timestamp"], errors="coerce").dt.date
        df["date"] = pd.to_datetime(df["date"])
    else:
        raise ValueError("No date/timestamp column found.")
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["volume","marketCap"]:
        if col in df.columns:
            df.loc[df[col] == 0, col] = np.nan

    df = df.sort_values(["crypto_name","date"]).reset_index(drop=True)

    for col in ["volume","marketCap"]:
        if col in df.columns:
            df[col] = df.groupby("crypto_name")[col].ffill().bfill()
            df[col] = df[col].fillna(df[col].median())

    # OHLC consistency
    def ok(r):
        return (r["high"] >= r["low"]) and (r["high"] >= r["open"]) and (r["high"] >= r["close"]) and (r["low"] <= r["open"]) and (r["low"] <= r["close"])
    df = df[df.apply(ok, axis=1)].copy()
    return df
