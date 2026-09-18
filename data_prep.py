"""
Data loading, cleaning, and feature engineering for the UAC Care Load
Forecasting project.
"""
import pandas as pd

RAW_PATH = "HHS_Unaccompanied_Alien_Children_Program.csv"

COLS = {
    "Date": "date",
    "Children apprehended and placed in CBP custody*": "apprehended",
    "Children in CBP custody": "cbp_custody",
    "Children transferred out of CBP custody": "transferred_to_hhs",
    "Children in HHS Care": "hhs_care",
    "Children discharged from HHS Care": "discharged",
}


def load_raw(path=RAW_PATH):
    df = pd.read_csv(path)
    df = df.dropna(subset=["Date"]).copy()
    df = df.rename(columns=COLS)
    for c in df.columns:
        if c == "date":
            continue
        df[c] = (
            df[c].astype(str).str.replace(",", "", regex=False).str.strip() # remove thousands-separator commas
        )
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], format="%B %d, %Y")
    df = df.sort_values("date").drop_duplicates(subset="date")
    df = df.set_index("date")
    return df


def build_daily_series(df):
    """
    Source data is reported irregularly (roughly 4-6 days/week, with
    real gaps around holidays and reporting lapses). Reindex to a
    continuous daily calendar and interpolate the flow/stock columns
    so downstream lag/rolling features and models see a clean series.
    """
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="D")
    daily = df.reindex(full_idx)
    daily.index.name = "date"

    # linear interpolation for gaps, both directions at the edges
    daily = daily.interpolate(method="linear", limit_direction="both")

    # flag which days were actually reported vs interpolated
    daily["is_reported"] = daily.index.isin(df.index)
    return daily


def add_features(daily):
    out = daily.copy()

    # net pressure: inflow to HHS system minus outflow (discharges)
    out["net_pressure"] = out["transferred_to_hhs"] - out["discharged"]

    target_cols = ["hhs_care", "discharged"]
    for col in target_cols:
        for lag in [1, 7, 14]:
            out[f"{col}_lag{lag}"] = out[col].shift(lag)
        for window in [7, 14]:
            out[f"{col}_rollmean{window}"] = (
                out[col].shift(1).rolling(window).mean()
            )
            out[f"{col}_rollstd{window}"] = (
                out[col].shift(1).rolling(window).std()
            )

    out["net_pressure_rollmean7"] = out["net_pressure"].shift(1).rolling(7).mean()

    # calendar effects
    out["dow"] = out.index.dayofweek
    out["month"] = out.index.month
    out["is_weekend"] = out["dow"].isin([5, 6]).astype(int)
    out["day_of_year"] = out.index.dayofyear

    return out


def get_clean_dataset(path=RAW_PATH):
    raw = load_raw(path)
    daily = build_daily_series(raw)
    featured = add_features(daily)
    return raw, daily, featured


if __name__ == "__main__":
    raw, daily, featured = get_clean_dataset()
    print("Raw reported rows:", len(raw))
    print("Daily calendar rows:", len(daily))
    print("Date range:", daily.index.min().date(), "to", daily.index.max().date())
    print("Reported fraction:", daily["is_reported"].mean().round(3))
    print(featured.tail())
