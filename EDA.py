import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data_prep import get_clean_dataset

st.set_page_config(
    page_title="UAC Care Load & Placement Demand Forecasting(EDA)",
    page_icon="📈",
    layout="wide",
)

# ----------------------data --------------------
@st.cache_data(show_spinner="Loading and preparing UAC data...")
def load_data():
    raw, daily, featured = get_clean_dataset()
    return raw, daily, featured

raw, daily, featured = load_data()
last_date = daily.index.max()

# =================== EDA ======================
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
       "The system underwent a sharp structural shift in early 2025 — average children in HHS "
       "care fell roughly 65% from 2024 to 2025, alongside a comparable drop in daily transfers "
       "and discharges. Models trained on the full 2023–2025 history should be interpreted with "
       "this regime change in mind (see Model Comparison tab)."
     )