"""
Baseline, statistical, and ML forecasting models + walk-forward
multi-horizon evaluation for UAC care-load / discharge series.
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

HORIZONS = [1, 7, 14]

# ---------- metrics ----------

def mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
        "MAPE": mape(y_true, y_pred),
    }


# ---------- baseline models ----------

def naive_forecast(history, horizon):
    """Persistence: last observed value repeated for the horizon."""
    last_val = history.iloc[-1]
    return np.full(horizon, last_val)


def moving_average_forecast(history, horizon, window=7):
    avg = history.iloc[-window:].mean()
    return np.full(horizon, avg)


# ---------- statistical models ----------

def sarima_forecast(history, horizon, order=(2, 1, 2), seasonal_order=(1, 0, 1, 7)):
    try:
        model = SARIMAX(
            history,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fit = model.fit(disp=False)
        fc = fit.forecast(horizon)
        return fc.values
    except Exception:
        return naive_forecast(history, horizon)


def exp_smoothing_forecast(history, horizon):
    try:
        model = ExponentialSmoothing(
            history, trend="add", seasonal="add", seasonal_periods=7,
            initialization_method="estimated",
        )
        fit = model.fit()
        fc = fit.forecast(horizon)
        return fc.values
    except Exception:
        return naive_forecast(history, horizon)


# ---------- ML models ---------------

def _ml_feature_cols(df, target):
    return [
        c for c in df.columns
        if c.startswith(f"{target}_lag") or c.startswith(f"{target}_roll")
        or c in ["dow", "month", "is_weekend", "net_pressure_rollmean7"]
    ]


def ml_forecast(train_df, full_df, target, horizon, model_cls, **kwargs):
    """
    Recursive forecasting: train a regressor on lag/rolling/calendar
    features, then roll forward one step at a time, feeding each
    prediction back in to compute the next step's lag features.
    """
    feat_cols = _ml_feature_cols(train_df, target)
    train = train_df.dropna(subset=feat_cols + [target])
    X_train, y_train = train[feat_cols], train[target]

    model = model_cls(**kwargs)
    model.fit(X_train, y_train)

    history = full_df[target].copy()
    net_pressure_hist = full_df["net_pressure"].copy()
    preds = []
    cursor = train_df.index[-1]

    for step in range(horizon):
        next_date = cursor + pd.Timedelta(days=1)
        row = {}
        for lag in [1, 7, 14]:
            idx = next_date - pd.Timedelta(days=lag)
            row[f"{target}_lag{lag}"] = history.get(idx, history.iloc[-1])
        for window in [7, 14]:
            vals = history.reindex(
                pd.date_range(next_date - pd.Timedelta(days=window), periods=window)
            )
            row[f"{target}_rollmean{window}"] = vals.mean()
            row[f"{target}_rollstd{window}"] = vals.std()
        row["dow"] = next_date.dayofweek
        row["month"] = next_date.month
        row["is_weekend"] = int(next_date.dayofweek in [5, 6])
        """"
        Net pressure (transfers minus discharges) has no per-step forecast of
        its own, so once the 7-day lookback window shifts entirely past the
        last real observation (horizons beyond ~7 days), every value in the
        window would otherwise be NaN and break sklearn's predict(). Forward-fill
        unknown future days with the most recently observed net pressure value —
        a reasonable "holds near its current level" assumption — so the feature
        stays defined at any horizon.
        """
        np_vals = net_pressure_hist.reindex(
            pd.date_range(next_date - pd.Timedelta(days=7), periods=7)
        ).fillna(net_pressure_hist.iloc[-1])
        row["net_pressure_rollmean7"] = np_vals.mean()

        X_next = pd.DataFrame([row])[feat_cols]
        if X_next.isna().any(axis=None):
            """
            Last-resort safety net: fill any unexpected NaN with the
            corresponding training feature's mean rather than let
            sklearn raise, since a dashboard crash is worse than a
            slightly degraded single-step estimate.
            """
            X_next = X_next.fillna(X_train.mean())
        pred = model.predict(X_next)[0]
        preds.append(pred)

        history.loc[next_date] = pred
        cursor = next_date

    return np.array(preds)


# ---------- walk-forward multi-horizon evaluation ----------

MODEL_REGISTRY = {
    "Naive Persistence": lambda tr, full, target, h: naive_forecast(tr[target], h),
    "Moving Average (7d)": lambda tr, full, target, h: moving_average_forecast(tr[target], h),
    "SARIMA": lambda tr, full, target, h: sarima_forecast(tr[target], h),
    "Exponential Smoothing": lambda tr, full, target, h: exp_smoothing_forecast(tr[target], h),
    "Random Forest": lambda tr, full, target, h: ml_forecast(
        tr, full, target, h, RandomForestRegressor, n_estimators=300, max_depth=8, random_state=42
    ),
    "Gradient Boosting": lambda tr, full, target, h: ml_forecast(
        tr, full, target, h, GradientBoostingRegressor, n_estimators=300, max_depth=3,
        learning_rate=0.05, random_state=42
    ),
}


def walk_forward_evaluate(
    df, target, models=None, horizons=HORIZONS,
    n_splits=12, step_days=14, min_train_days=365,
):
    """
    Rolling-origin walk-forward validation. At each split, train on
    data up to a cutoff, forecast max(horizons) days ahead, and score
    against the actual values at each horizon.
    """
    if models is None:
        models = list(MODEL_REGISTRY.keys())

    max_h = max(horizons)
    n = len(df)
    last_possible_cutoff = n - max_h
    first_cutoff = min_train_days
    if last_possible_cutoff <= first_cutoff:
        raise ValueError("Not enough data for requested walk-forward config")

    cutoffs = sorted(list(range(last_possible_cutoff, first_cutoff, -step_days))[:n_splits])
   
    records = []
    for cutoff in cutoffs:
        train_df = df.iloc[:cutoff]
        actual_future = df[target].iloc[cutoff: cutoff + max_h]
        if len(actual_future) < max_h:
            continue

        for name in models:
            fn = MODEL_REGISTRY[name]
            try:
                preds = fn(train_df, df.iloc[: cutoff + max_h], target, max_h)
            except Exception:
                continue
            for h in horizons:
                y_true = actual_future.values[:h]
                y_pred = np.asarray(preds[:h])
                m = compute_metrics(y_true, y_pred)
                records.append({
                    "model": name,
                    "cutoff_date": df.index[cutoff - 1],
                    "horizon": h,
                    **m,
                })

    return pd.DataFrame(records)


def summarize_results(results_df):
    summary = (
        results_df.groupby(["model", "horizon"])[["MAE", "RMSE", "MAPE"]]
        .mean()
        .reset_index()
        .sort_values(["horizon", "MAE"])
    )
    return summary


def forecast_future(df, target, model_name, horizon, ci_alpha=0.1):
    """
    Produce a future forecast (beyond the end of df) with a simple
    residual-based confidence interval for display purposes.
    """
    fn = MODEL_REGISTRY[model_name]
    preds = fn(df, df, target, horizon)
    """
    crude uncertainty band from *recent* day-over-day volatility only
    (using the full history would let the 2025 structural break dominate
    the spread and produce unrealistically wide bands for the current regime)
    """
    resid_std = df[target].diff().iloc[-120:].std()
    from scipy import stats
    z = stats.norm.ppf(1 - ci_alpha / 2)
    growing_sd = resid_std * np.sqrt(np.arange(1, horizon + 1))
    lower = preds - z * growing_sd
    upper = preds + z * growing_sd

    future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=horizon)
    return pd.DataFrame({
        "date": future_dates,
        "forecast": preds,
        "lower": np.clip(lower, 0, None),
        "upper": upper,
    }).set_index("date")
