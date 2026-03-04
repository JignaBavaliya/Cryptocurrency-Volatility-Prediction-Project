import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .config import FEATURE_COLS

def run_eda(df_model: pd.DataFrame, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    counts = df_model["crypto_name"].value_counts().head(20).sort_values()
    plt.figure(figsize=(8,6))
    counts.plot(kind="barh")
    plt.title("Top 20 cryptocurrencies by number of records")
    plt.xlabel("Record count")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "eda_records_per_crypto.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(7,5))
    plt.hist(df_model["target_future_vol_7d"], bins=80)
    plt.title("Distribution: future 7-day realised volatility target")
    plt.xlabel("Future 7d realised volatility")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "eda_target_distribution.png"), dpi=200)
    plt.close()

    corr = df_model[FEATURE_COLS].corr().values
    plt.figure(figsize=(10,8))
    plt.imshow(corr, aspect="auto")
    plt.xticks(range(len(FEATURE_COLS)), FEATURE_COLS, rotation=90, fontsize=6)
    plt.yticks(range(len(FEATURE_COLS)), FEATURE_COLS, fontsize=6)
    plt.colorbar()
    plt.title("Correlation heatmap (engineered features)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "eda_corr_heatmap.png"), dpi=200)
    plt.close()

    sample = df_model.sample(n=min(25000, len(df_model)), random_state=42)
    plt.figure(figsize=(7,5))
    plt.scatter(sample["liquidity_7"], sample["target_future_vol_7d"], s=4)
    plt.title("Liquidity (rolling mean) vs future volatility")
    plt.xlabel("liquidity_7")
    plt.ylabel("Future 7d realised volatility")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "eda_liquidity_vs_future_vol.png"), dpi=200)
    plt.close()
