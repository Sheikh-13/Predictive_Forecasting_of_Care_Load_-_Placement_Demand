"""
Operational KPI calculations for the UAC forecasting dashboard/report:
- Forecast Accuracy (%)
- Surge Lead Time
- Capacity Breach Probability
- Forecast Stability Index
"""
import numpy as np
import pandas as pd


def forecast_accuracy_pct(mape_value):
    """Simple, interpretable inverse of MAPE, floored at 0."""
    if mape_value is None or np.isnan(mape_value):
        return np.nan
    return max(0.0, 100 - mape_value)


def capacity_breach_probability(forecast_df, capacity_threshold):
    """
    Given a forecast with 'forecast','lower','upper' columns and a
    capacity threshold, estimate the probability of breach on each
    forecast day assuming a normal distribution implied by the CI band,
    then return the probability of at least one breach across the horizon.
    """
    from scipy import stats
    z90 = stats.norm.ppf(0.95)  # matches the 90% CI (alpha=0.1) used in forecast_future
    sigma = (forecast_df["upper"] - forecast_df["forecast"]) / z90
    sigma = sigma.replace(0, np.nan)
    prob_breach_daily = 1 - stats.norm.cdf(
        capacity_threshold, loc=forecast_df["forecast"], scale=sigma.fillna(1e-6)
    )
    prob_no_breach_any_day = np.prod(1 - np.clip(prob_breach_daily, 0, 1))
    prob_breach_any_day = 1 - prob_no_breach_any_day
    return {
        "daily_breach_prob": pd.Series(prob_breach_daily, index=forecast_df.index),
        "any_day_breach_prob": float(prob_breach_any_day),
    }


def surge_lead_time(history, forecast_df, surge_multiple=1.15, lookback=30):
    """
    Days of advance warning a forecast gives before care load is
    projected to exceed `surge_multiple` times the recent baseline
    (mean of the last `lookback` observed days).
    """
    baseline = history.iloc[-lookback:].mean()
    surge_level = baseline * surge_multiple
    breach_days = forecast_df.index[forecast_df["forecast"] >= surge_level]
    if len(breach_days) == 0:
        return None  # no surge projected within the forecast horizon
    first_breach = breach_days[0]
    lead_days = (first_breach - forecast_df.index[0]).days + 1
    return {
        "baseline": baseline,
        "surge_level": surge_level,
        "first_breach_date": first_breach,
        "lead_time_days": lead_days,
    }


def forecast_stability_index(results_df, model_name, horizon):
    """
    How much a model's error varies across different walk-forward
    origins (lower = more stable/robust). Returned as the coefficient
    of variation of MAE across walk-forward splits, and as a 0-100
    'stability score' (100 = perfectly stable).
    """
    subset = results_df[(results_df["model"] == model_name) & (results_df["horizon"] == horizon)]
    if len(subset) < 2 or subset["MAE"].mean() == 0:
        return {"cv": np.nan, "stability_score": np.nan}
    cv = subset["MAE"].std() / subset["MAE"].mean()
    score = max(0.0, 100 * (1 - min(cv, 1.0)))
    return {"cv": float(cv), "stability_score": float(score)}


def model_robustness_table(results_df, horizons):
    rows = []
    for model in results_df["model"].unique():
        for h in horizons:
            fsi = forecast_stability_index(results_df, model, h)
            rows.append({"model": model, "horizon": h, **fsi})
    return pd.DataFrame(rows)