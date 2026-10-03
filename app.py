"""
UAC Program — Predictive Forecasting of Care Load & Placement Demand
Streamlit dashboard.
"""
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data_prep import get_clean_dataset
from forecasting import (
    walk_forward_evaluate, summarize_results, forecast_future,
    MODEL_REGISTRY, HORIZONS,
)
from kpis import (
    forecast_accuracy_pct, capacity_breach_probability,
    surge_lead_time, model_robustness_table,
)

st.set_page_config(
    page_title="UAC Care Load & Placement Demand Forecasting",
    page_icon="📈",
    layout="wide",
)

TARGET_LABELS = {"hhs_care": "Children in HHS Care", "discharged": "Children Discharged / Day"}
MODEL_NAMES = list(MODEL_REGISTRY.keys())


# ----------------------data --------------------
@st.cache_data(show_spinner="Loading and preparing UAC data...")
def load_data():
    raw, daily, featured = get_clean_dataset()
    return raw, daily, featured


@st.cache_data(show_spinner="Running walk-forward validation across models... (first load only)")
def run_walkforward(target, n_splits=8, step_days=25, min_train_days=450):
    _, _, featured = load_data()
    res = walk_forward_evaluate(
        featured, target, n_splits=n_splits, step_days=step_days,
        min_train_days=min_train_days,
    )
    return res


@st.cache_data(show_spinner="Generating forecast...")
def run_future_forecast(target, model_name, horizon):
    _, _, featured = load_data()
    return forecast_future(featured, target, model_name, horizon)


raw, daily, featured = load_data()
last_date = daily.index.max()

# --------------------- sidebar --------------
st.sidebar.title("Forecast Controls")
horizon = st.sidebar.slider("Forecast horizon (days)", min_value=3, max_value=14, value=7)
model_choice = st.sidebar.selectbox("Model", MODEL_NAMES, index=MODEL_NAMES.index("SARIMA"))
compare_model = st.sidebar.selectbox(
    "Compare against", ["(none)"] + MODEL_NAMES, index=0,
    help="Pick a second model to overlay for scenario comparison.",
)
capacity_threshold = st.sidebar.number_input(
    "Care-load capacity threshold (for breach probability)",
    min_value=500, max_value=15000, value=int(daily["hhs_care"].iloc[-90:].max() * 1.1), step=100,
)
surge_multiple = st.sidebar.slider(
    "Surge definition (x current 30-day baseline)", 1.05, 2.0, 1.15, 0.05,
)
st.sidebar.caption(f"Data through **{last_date.date()}** · {raw.shape[0]} reported observations, "
                    f"reindexed to {daily.shape[0]} daily points.")

st.title("Predictive Forecasting of Care Load & Placement Demand")
st.caption("UAC Program — HHS/CBP daily operations data · forecasting dashboard for capacity planning")

tab_overview, tab_care, tab_discharge, tab_models, tab_scenarios = st.tabs(
    ["Overview & EDA", "Care Load Forecast", "Discharge Demand", "Model Comparison", "Scenario Comparison"]
)

# ======================= OVERVIEW TAB =====================
with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Children in HHS Care (latest)", f"{daily['hhs_care'].iloc[-1]:,.0f}",
              f"{daily['hhs_care'].iloc[-1] - daily['hhs_care'].iloc[-8]:+,.0f} vs 7d ago")
    c2.metric("Daily Discharges (latest)", f"{daily['discharged'].iloc[-1]:,.0f}",
              f"{daily['discharged'].iloc[-1] - daily['discharged'].iloc[-8]:+,.0f} vs 7d ago")
    c3.metric("Transfers into HHS (latest)", f"{daily['transferred_to_hhs'].iloc[-1]:,.0f}")
    net_pressure_now = (daily["transferred_to_hhs"] - daily["discharged"]).iloc[-7:].mean()
    c4.metric("Net Pressure (7d avg)", f"{net_pressure_now:+,.1f}",
              help="Transfers into HHS care minus discharges. Positive = system load growing.")

    st.subheader("Care load & discharge trend")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=daily.index, y=daily["hhs_care"], name="Children in HHS Care",
                              line=dict(color="#1f4e79")), secondary_y=False)
    fig.add_trace(go.Scatter(x=daily.index, y=daily["discharged"].rolling(7).mean(),
                              name="Discharges (7d avg)", line=dict(color="#c0392b")), secondary_y=True)
    fig.update_yaxes(title_text="Children in HHS Care", secondary_y=False)
    fig.update_yaxes(title_text="Daily Discharges (7d avg)", secondary_y=True)
    fig.update_layout(height=420, legend=dict(orientation="h", y=1.08), margin=dict(t=30))
    st.plotly_chart(fig, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Net system pressure")
        net_pressure = (daily["transferred_to_hhs"] - daily["discharged"]).rolling(7).mean()
        colors = np.where(net_pressure >= 0, "#c0392b", "#2e7d32")
        fig2 = go.Figure(go.Bar(x=daily.index, y=net_pressure, marker_color=colors))
        fig2.update_layout(height=340, margin=dict(t=10),
                            yaxis_title="Transfers-in minus Discharges (7d avg)")
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("Red = system load growing (more entering than leaving). "
                   "Green = system load shrinking.")
    with col_b:
        st.subheader("Correlation across program stages")
        corr_cols = ["apprehended", "cbp_custody", "transferred_to_hhs", "hhs_care", "discharged"]
        corr = daily[corr_cols].corr()
        fig3 = go.Figure(data=go.Heatmap(
            z=corr.values, x=corr_cols, y=corr_cols, colorscale="RdBu", zmid=0,
            text=corr.round(2).values, texttemplate="%{text}"))
        fig3.update_layout(height=340, margin=dict(t=10))
        st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Year-over-year averages")
    yearly = daily.groupby(daily.index.year)[["hhs_care", "transferred_to_hhs", "discharged"]].mean().round(1)
    yearly.index.name = "Year"
    st.dataframe(yearly, use_container_width=True)
    st.caption(
        """
      The system underwent a sharp structural shift in early 2025 — average children in HHS 
      care fell roughly 65% from 2024 to 2025, alongside a comparable drop in daily transfers 
      and discharges. Models trained on the full 2023–2025 history should be interpreted with 
      this regime change in mind (see Model Comparison tab)
      """
    )
# ====================== CARE LOAD TAB ===================

def render_forecast_tab(target, container):
    with container:
        fc = run_future_forecast(target, model_choice, horizon)
        history_tail = daily[target].iloc[-60:]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=history_tail.index, y=history_tail.values,
                                  name="Observed", line=dict(color="#333")))
        fig.add_trace(go.Scatter(x=fc.index, y=fc["forecast"], name=f"{model_choice} forecast",
                                  line=dict(color="#1f4e79", dash="dash")))
        fig.add_trace(go.Scatter(
            x=list(fc.index) + list(fc.index[::-1]),
            y=list(fc["upper"]) + list(fc["lower"][::-1]),
            fill="toself", fillcolor="rgba(31,78,121,0.15)", line=dict(width=0),
            mode="dots",
            hoverinfo="skip",showlegend=True,
            name="90% confidence interval",
        ))

        if compare_model != "(none)":
            fc2 = run_future_forecast(target, compare_model, horizon)
            fig.add_trace(go.Scatter(x=fc2.index, y=fc2["forecast"], name=f"{compare_model} forecast",
                                      line=dict(color="#c0392b", dash="dot")))

        if target == "hhs_care":
            fig.add_hline(y=capacity_threshold, line_color="red", line_dash="dot",
                           annotation_text="Capacity threshold")

        fig.update_layout(height=440, margin=dict(t=20),
                           yaxis_title=TARGET_LABELS[target],
                           legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)

        k1, k2, k3 = st.columns(3)
        res = run_walkforward(target)
        summary = summarize_results(res)
        
        # Walk-forward metrics are only computed at HORIZONS (1/7/14 days), but the
        # sidebar slider allows any value in between — snap to the closest evaluated
        # horizon instead of an exact match, or the accuracy metric goes blank for
        # most slider positions (e.g. horizon=5).

        nearest_h = min(HORIZONS, key=lambda h: abs(h - horizon))
        row = summary[(summary.model == model_choice) & (summary.horizon == nearest_h)]
        acc = forecast_accuracy_pct(row["MAPE"].values[0]) if len(row) else np.nan
        k1.metric("Forecast Accuracy (walk-forward)", f"{acc:.1f}%" if not np.isnan(acc) else "n/a",
                   help=f"100 - MAPE for {model_choice}, evaluated at the {nearest_h}-day horizon "
                        f"(closest to the selected {horizon}-day horizon).")

        if target == "hhs_care":
            breach = capacity_breach_probability(fc, capacity_threshold)
            k2.metric("Capacity Breach Probability", f"{breach['any_day_breach_prob']*100:.1f}%",
                       help=f"Probability care load exceeds {capacity_threshold:,} on any day in the forecast window.")
            surge = surge_lead_time(daily["hhs_care"], fc, surge_multiple=surge_multiple)
            if surge:
                k3.metric("Surge Lead Time", f"{surge['lead_time_days']} days",
                           help=f"Advance warning before load is projected to exceed {surge['surge_level']:,.0f} "
                                f"({surge_multiple}x the {int(30)}-day baseline of {surge['baseline']:,.0f}).")
            else:
                k3.metric("Surge Lead Time", "No surge in horizon",
                           help=f"Forecast does not cross {surge_multiple}x baseline within the selected horizon.")
        else:
            k2.metric("Forecast (avg over horizon)", f"{fc['forecast'].mean():.1f} / day")
            k3.metric("Cumulative expected discharges", f"{fc['forecast'].sum():.0f}",
                       help=f"Sum of forecasted daily discharges over the next {horizon} days.")

        with st.expander("Forecast table"):
            st.dataframe(fc.round(1), use_container_width=True)


with tab_care:
    st.subheader(f"Future Care Load Forecast — {model_choice}")
    render_forecast_tab("hhs_care", st.container())

with tab_discharge:
    st.subheader(f"Discharge Demand Forecast — {model_choice}")
    render_forecast_tab("discharged", st.container())

# ======================== MODEL COMPARISON ==============================
with tab_models:
    st.subheader("Model selection & walk-forward comparison")
    target_for_compare = st.radio("Target series", list(TARGET_LABELS.keys()),
                                   format_func=lambda t: TARGET_LABELS[t], horizontal=True)
    res = run_walkforward(target_for_compare)
    summary = summarize_results(res)

    fig = go.Figure()
    for m in MODEL_NAMES:
        sub = summary[summary.model == m]
        fig.add_trace(go.Bar(x=sub["horizon"], y=sub["MAPE"], name=m))
    fig.update_layout(barmode="group", height=420, margin=dict(t=20),
                       xaxis_title="Forecast horizon (days)", yaxis_title="MAPE (%)",
                       legend=dict(orientation="h", y=1.15))
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        summary.pivot(index="model", columns="horizon", values="MAE").round(2)
        .rename(columns=lambda h: f"{h}d MAE"),
        use_container_width=True,
    )

    st.subheader("Forecast Stability Index (model robustness)")
    rob = model_robustness_table(res, HORIZONS)
    rob_pivot = rob.pivot(index="model", columns="horizon", values="stability_score").round(1)
    rob_pivot.columns = [f"{h}d stability score" for h in rob_pivot.columns]
    st.dataframe(rob_pivot, use_container_width=True)
    st.caption("Stability score (0-100): how consistent each model's error is across different walk-forward origins. Higher = more robust / less prone to occasional large misses.")

# ======================== SCENARIOS =============================
with tab_scenarios:
    st.subheader("Scenario comparison")
    st.caption("Compare how forecasts and capacity risk shift under different assumptions.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Scenario A**")
        model_a = st.selectbox("Model A", MODEL_NAMES, index=MODEL_NAMES.index("SARIMA"), key="model_a")
        cap_a = st.number_input("Capacity threshold A", 500, 15000,
                                 int(daily["hhs_care"].iloc[-90:].max() * 1.1), 100, key="cap_a")
    with col2:
        st.markdown("**Scenario B**")
        model_b = st.selectbox("Model B", MODEL_NAMES, index=MODEL_NAMES.index("Gradient Boosting"), key="model_b")
        cap_b = st.number_input("Capacity threshold B", 500, 15000,
                                 int(daily["hhs_care"].iloc[-90:].max() * 1.3), 100, key="cap_b")

    fc_a = run_future_forecast("hhs_care", model_a, horizon)
    fc_b = run_future_forecast("hhs_care", model_b, horizon)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily.index[-45:], y=daily["hhs_care"].iloc[-45:],
                              name="Observed", line=dict(color="#333")))
    fig.add_trace(go.Scatter(x=fc_a.index, y=fc_a["forecast"], name=f"Scenario A: {model_a}",
                              line=dict(color="#1f4e79", dash="dash")))
    fig.add_trace(go.Scatter(x=fc_b.index, y=fc_b["forecast"], name=f"Scenario B: {model_b}",
                              line=dict(color="#e67e22", dash="dash")))
    fig.add_hline(y=cap_a, line_color="#1f4e79", line_dash="dot", annotation_text="Capacity A")
    fig.add_hline(y=cap_b, line_color="#e67e22", line_dash="dot", annotation_text="Capacity B")
    fig.update_layout(height=440, margin=dict(t=20), yaxis_title="Children in HHS Care",
                       legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)

    breach_a = capacity_breach_probability(fc_a, cap_a)
    breach_b = capacity_breach_probability(fc_b, cap_b)
    cA, cB = st.columns(2)
    cA.metric(f"Scenario A breach probability ({model_a}, cap={cap_a:,})",
              f"{breach_a['any_day_breach_prob']*100:.1f}%")
    cB.metric(f"Scenario B breach probability ({model_b}, cap={cap_b:,})",
              f"{breach_b['any_day_breach_prob']*100:.1f}%")

st.divider()
