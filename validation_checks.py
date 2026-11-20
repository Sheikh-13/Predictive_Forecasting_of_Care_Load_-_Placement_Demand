"""
Supporting validation checks (run once, outside the dashboard):

1. Regime-aware walk-forward — evaluates every model at origins BEFORE, DURING
   and AFTER the early-2025 structural break, instead of only after it.
2. Interpolation sensitivity — compares accuracy scored on all days against
   accuracy scored only on days that were actually reported (not interpolated).

Every forecast uses train-only data (information available at the origin).
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from data_prep import get_clean_dataset
from forecasting import MODEL_REGISTRY, compute_metrics

HORIZONS = [1, 7, 14]
MAX_H = max(HORIZONS)

_, daily, featured = get_clean_dataset()
n = len(featured)
idx = featured.index
is_rep = featured["is_reported"].values


def day_index(date):
    return int((pd.Timestamp(date) - idx[0]).days)


BRK_START, BRK_END = day_index("2025-01-04"), day_index("2025-02-14")  # care load >=6000 -> <=2700

REGIMES = {
    "pre-break":  list(range(520, BRK_START - MAX_H + 1, 20))[:10],
    "transition": list(range(BRK_START - 11, BRK_END + 2, 6))[:10],
    "post-break": sorted(list(range(n - MAX_H, 500, -20))[:10]),
}


def collect(target):
    rows = []
    for regime, cutoffs in REGIMES.items():
        for cutoff in cutoffs:
            train = featured.iloc[:cutoff]
            actual = featured[target].iloc[cutoff:cutoff + MAX_H].values
            rep = is_rep[cutoff:cutoff + MAX_H]
            if len(actual) < MAX_H:
                continue
            for name, fn in MODEL_REGISTRY.items():
                try:
                    pred = np.asarray(fn(train, train, target, MAX_H))
                except Exception:
                    continue
                for h in HORIZONS:
                    m_all = compute_metrics(actual[:h], pred[:h])
                    keep = rep[:h]
                    m_rep = (compute_metrics(actual[:h][keep], pred[:h][keep])
                             if keep.any() else {"MAE": np.nan, "MAPE": np.nan})
                    rows.append({
                        "regime": regime, "model": name, "horizon": h,
                        "cutoff": idx[cutoff - 1], "MAE_all": m_all["MAE"],
                        "MAPE_all": m_all["MAPE"], "MAE_reported": m_rep["MAE"],
                        "MAPE_reported": m_rep["MAPE"],
                        "share_interpolated": float((~keep).mean()),
                    })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    for target in ["hhs_care", "discharged"]:
        res = collect(target)
        res.to_csv(f"outputs/regime_validation_{target}.csv", index=False)
        print(f"saved {target}: {len(res)} rows")
