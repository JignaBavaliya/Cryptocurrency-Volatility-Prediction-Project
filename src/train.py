import os, json, math
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from .preprocess import load_data, clean_data
from .features import add_features, add_target
from .config import FEATURE_COLS, CAT_COLS, TARGET_COL, RANDOM_SEED

def make_time_split(df: pd.DataFrame, test_frac: float = 0.2):
    unique_dates = np.sort(df["date"].unique())
    cut = unique_dates[int(len(unique_dates) * (1 - test_frac))]
    train = df[df["date"] <= cut].copy()
    test  = df[df["date"] >  cut].copy()
    return train, test, cut

def walk_forward_splits(train_df: pd.DataFrame, n_splits: int = 3):
    train_dates = np.sort(train_df["date"].unique())
    tscv = TimeSeriesSplit(n_splits=n_splits)
    date_to_idx = {}
    dvals = train_df["date"].values
    for i, d in enumerate(dvals):
        date_to_idx.setdefault(d, []).append(i)

    splits = []
    for tr_d_idx, va_d_idx in tscv.split(train_dates):
        tr_dates = set(train_dates[tr_d_idx])
        va_dates = set(train_dates[va_d_idx])
        tr_idx = np.array(sorted([i for d in tr_dates for i in date_to_idx.get(d, [])]))
        va_idx = np.array(sorted([i for d in va_dates for i in date_to_idx.get(d, [])]))
        splits.append((tr_idx, va_idx))
    return splits

def tune_alpha(pipe_factory, X_train, y_train, splits, alphas):
    best_alpha, best_rmse = None, float("inf")
    for a in alphas:
        rmses=[]
        for tr_idx, va_idx in splits:
            m = pipe_factory(a)
            m.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
            pred = m.predict(X_train.iloc[va_idx])
            rmses.append(math.sqrt(mean_squared_error(y_train.iloc[va_idx], pred)))
        rmse=float(np.mean(rmses))
        if rmse < best_rmse:
            best_rmse, best_alpha = rmse, a
    return best_alpha, best_rmse

def main():
    root = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(root, "data", "dataset.csv")
    artifacts_dir = os.path.join(root, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)

    df = load_data(data_path)
    df = clean_data(df)
    df = add_features(df)
    df = add_target(df, horizon=7)

    df_model = df.dropna(subset=FEATURE_COLS + [TARGET_COL]).copy()

    train_df, test_df, cut = make_time_split(df_model, test_frac=0.2)

    X_train = train_df[CAT_COLS + FEATURE_COLS]
    y_train = train_df[TARGET_COL]
    X_test  = test_df[CAT_COLS + FEATURE_COLS]
    y_test  = test_df[TARGET_COL]

    preprocess = ColumnTransformer([
        ("num", StandardScaler(), FEATURE_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
    ])

    def pipe_factory(alpha):
        return Pipeline([
            ("prep", preprocess),
            ("model", Ridge(alpha=alpha, random_state=RANDOM_SEED)),
        ])

    splits = walk_forward_splits(train_df, n_splits=3)
    alphas = [0.1, 1.0, 10.0, 100.0, 1000.0]
    best_alpha, cv_rmse = tune_alpha(pipe_factory, X_train, y_train, splits, alphas)

    model = pipe_factory(best_alpha)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    metrics = {
        "best_alpha": best_alpha,
        "cv_rmse": float(cv_rmse),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        "test_mae": float(mean_absolute_error(y_test, pred)),
        "test_r2": float(r2_score(y_test, pred)),
        "train_end_date": str(pd.to_datetime(cut).date()),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
    }

    joblib.dump(model, os.path.join(artifacts_dir, "volatility_ridge.joblib"))
    with open(os.path.join(artifacts_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
