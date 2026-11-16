"""
app.py  –  Spay/Neuter Hotspot Predictor
Run:  streamlit run app.py
"""

from __future__ import annotations

import sys
import os

# Allow imports from the project root when launched from any working directory
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data.generate_data import generate_if_missing
from model import FEATURE_COLS, predict, train_model

# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Spay/Neuter Hotspot Predictor",
    page_icon="🐾",
    layout="wide",
)

# ──────────────────────────────────────────────────────────────────────────────
# Custom CSS – minimal, clean
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        .metric-card {
            background: #f7f8fa;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 1rem 1.2rem;
            text-align: center;
        }
        .metric-card .label  { font-size: 0.78rem; color: #57606a; margin-bottom: 4px; }
        .metric-card .value  { font-size: 1.6rem; font-weight: 700; color: #1f2328; }
        .risk-high   { color: #dc2626; font-weight: 600; }
        .risk-medium { color: #d97706; font-weight: 600; }
        .risk-low    { color: #16a34a; font-weight: 600; }
        h1 { font-size: 1.9rem !important; }
        .stDataFrame { font-size: 0.84rem; }
        .action-banner {
            border-radius: 8px;
            padding: 0.85rem 1.2rem;
            margin-top: 0.6rem;
            font-size: 0.92rem;
            line-height: 1.55;
        }
        .action-banner strong { font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.04em; }
        .action-high   { background: #fef2f2; border-left: 4px solid #dc2626; color: #7f1d1d; }
        .action-medium { background: #fffbeb; border-left: 4px solid #d97706; color: #78350f; }
        .action-low    { background: #f0fdf4; border-left: 4px solid #16a34a; color: #14532d; }
        .action-all    { background: #f0f9ff; border-left: 4px solid #3b82d4; color: #1e3a5f; }
        .roi-card {
            background: #f7f8fa;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 1rem 1.2rem;
            text-align: center;
        }
        .roi-card .label { font-size: 0.78rem; color: #57606a; margin-bottom: 4px; }
        .roi-card .value { font-size: 1.5rem; font-weight: 700; color: #1f2328; }
        .roi-card .sub   { font-size: 0.72rem; color: #57606a; margin-top: 3px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("# 🐾 Spay / Neuter Hotspot Predictor")
st.markdown(
    "Machine-learning powered overpopulation risk scoring by zip code — "
    "helping shelters prioritise targeted spay/neuter outreach."
)
st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# Data loading & model training (cached)
# ──────────────────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "shelter_data.csv")


@st.cache_data(show_spinner="Loading shelter data…")
def load_data() -> pd.DataFrame:
    return generate_if_missing(DATA_PATH)


@st.cache_resource(show_spinner="Training model…")
def get_model(data_hash: int):  # data_hash busts cache if data changes
    df = load_data()
    pipeline, metrics = train_model(df)
    return pipeline, metrics


raw_df = load_data()
pipeline, metrics = get_model(hash(raw_df.to_json()))
scored_df = predict(pipeline, raw_df)

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar – filters & controls
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🔧 Filters & Controls")

    tier_options = ["All", "🔴 High", "🟡 Medium", "🟢 Low"]
    selected_tier = st.selectbox("Risk Tier", tier_options, index=0)

    risk_range = st.slider(
        "Predicted Risk Score",
        min_value=0,
        max_value=100,
        value=(0, 100),
        step=1,
    )

    min_intake = st.number_input(
        "Min Total Intake", min_value=0, max_value=int(scored_df["total_intake"].max()), value=0
    )

    st.divider()
    st.subheader("📊 Model Performance")
    st.metric("R² (test)", metrics["r2_test"])
    st.metric("MAE (test)", f"{metrics['mae_test']} pts")
    st.metric("CV R² (5-fold)", f"{metrics['cv_r2_mean']} ± {metrics['cv_r2_std']}")

# ──────────────────────────────────────────────────────────────────────────────
# Apply filters
# ──────────────────────────────────────────────────────────────────────────────
filtered = scored_df.copy()
if selected_tier != "All":
    filtered = filtered[filtered["risk_tier"] == selected_tier]
filtered = filtered[
    filtered["predicted_risk"].between(risk_range[0], risk_range[1])
    & (filtered["total_intake"] >= min_intake)
]

# ──────────────────────────────────────────────────────────────────────────────
# KPI row  (driven by filtered data)
# ──────────────────────────────────────────────────────────────────────────────
high_risk_count = (filtered["risk_tier"] == "🔴 High").sum()
med_risk_count  = (filtered["risk_tier"] == "🟡 Medium").sum()
low_risk_count  = (filtered["risk_tier"] == "🟢 Low").sum()
avg_risk        = filtered["predicted_risk"].mean() if len(filtered) else 0.0

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
for col, label, value, css_class in [
    (kpi1, "Total Zip Codes",  len(filtered),           ""),
    (kpi2, "🔴 High Risk",     high_risk_count,          "risk-high"),
    (kpi3, "🟡 Medium Risk",   med_risk_count,           "risk-medium"),
    (kpi4, "🟢 Low Risk",      low_risk_count,           "risk-low"),
    (kpi5, "Avg Risk Score",   f"{avg_risk:.1f} / 100",  ""),
]:
    col.markdown(
        f'<div class="metric-card">'
        f'<div class="label">{label}</div>'
        f'<div class="value {css_class}">{value}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────────────────────────────────────
# Actionable Resource Allocation Banner
# ──────────────────────────────────────────────────────────────────────────────
_BANNER_CFG = {
    "🔴 High": (
        "action-high",
        "🚨 Deploy Mobile Surgical Unit for a 3-day targeted clinic and "
        "dispatch TNR trapper teams immediately.",
    ),
    "🟡 Medium": (
        "action-medium",
        "📢 Distribute targeted community spay/neuter vouchers and conduct "
        "door-to-door educational outreach.",
    ),
    "🟢 Low": (
        "action-low",
        "✅ Maintain standard intake monitoring and routine public education.",
    ),
    "All": (
        "action-all",
        "📋 Viewing all risk tiers — select a specific tier in the sidebar for a "
        "targeted operational recommendation.",
    ),
}

_banner_css, _banner_msg = _BANNER_CFG[selected_tier]
st.markdown(
    f'<div class="action-banner {_banner_css}">'
    f"<strong>Recommended Operational Action</strong><br>{_banner_msg}"
    f"</div>",
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# Financial ROI & Animal Welfare Metrics
# ──────────────────────────────────────────────────────────────────────────────
_INTAKE_PREVENTED_PER_ZIP = 45
_COST_PER_ANIMAL          = 150

# Use high-risk count from the *full* scored_df so ROI always reflects the
# absolute programme opportunity, independent of tier filter.
_full_high_count      = int((scored_df["risk_tier"] == "🔴 High").sum())
_intakes_prevented    = _full_high_count * _INTAKE_PREVENTED_PER_ZIP
_budget_saved         = _intakes_prevented * _COST_PER_ANIMAL

roi1, roi2 = st.columns(2)
roi1.markdown(
    f'<div class="roi-card">'
    f'<div class="label">🐾 Estimated Intakes Prevented</div>'
    f'<div class="value">{_intakes_prevented:,}</div>'
    f'<div class="sub">({_full_high_count} high-risk zips × {_INTAKE_PREVENTED_PER_ZIP} animals/zip)</div>'
    f"</div>",
    unsafe_allow_html=True,
)
roi2.markdown(
    f'<div class="roi-card">'
    f'<div class="label">💰 Shelter Budget Saved</div>'
    f'<div class="value">${_budget_saved:,.0f}</div>'
    f'<div class="sub">({_intakes_prevented:,} prevented × ${_COST_PER_ANIMAL}/animal care cost)</div>'
    f"</div>",
    unsafe_allow_html=True,
)

st.markdown("<br>", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Row 1 – Bar chart of top hotspots  |  Pie chart of tier distribution
# ──────────────────────────────────────────────────────────────────────────────
col_bar, col_pie = st.columns([2, 1], gap="large")

with col_bar:
    st.subheader("Top 20 Highest-Risk Zip Codes")
    top20 = filtered.nlargest(20, "predicted_risk").sort_values("predicted_risk")
    color_map = {"🔴 High": "#dc2626", "🟡 Medium": "#d97706", "🟢 Low": "#16a34a"}
    fig_bar = px.bar(
        top20,
        x="predicted_risk",
        y="zip_code",
        color="risk_tier",
        color_discrete_map=color_map,
        orientation="h",
        labels={"predicted_risk": "Predicted Risk Score", "zip_code": "Zip Code", "risk_tier": "Tier"},
        hover_data=["total_intake", "prior_spay_neuter_rate", "median_household_income"],
        height=460,
    )
    fig_bar.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend_title_text="Risk Tier",
        xaxis=dict(range=[0, 100], gridcolor="#e5e7eb"),
        yaxis=dict(tickfont=dict(size=11)),
        margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with col_pie:
    st.subheader("Risk Tier Distribution")
    tier_counts = filtered["risk_tier"].value_counts().reset_index()
    tier_counts.columns = ["Tier", "Count"]
    fig_pie = px.pie(
        tier_counts,
        names="Tier",
        values="Count",
        color="Tier",
        color_discrete_map=color_map,
        hole=0.45,
        height=460,
    )
    fig_pie.update_traces(textposition="outside", textinfo="percent+label")
    fig_pie.update_layout(
        showlegend=False,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="white",
    )
    st.plotly_chart(fig_pie, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# Row 2 – Scatter: intake vs risk  |  Feature importance
# ──────────────────────────────────────────────────────────────────────────────
col_scatter, col_feat = st.columns(2, gap="large")

with col_scatter:
    st.subheader("Total Intake vs Predicted Risk")
    fig_sc = px.scatter(
        filtered,
        x="total_intake",
        y="predicted_risk",
        color="risk_tier",
        color_discrete_map=color_map,
        size="repeat_intakes",
        hover_name="zip_code",
        hover_data={
            "prior_spay_neuter_rate": ":.2f",
            "median_household_income": ":,",
            "total_intake": True,
            "predicted_risk": ":.1f",
        },
        labels={
            "total_intake": "Total Intake (animals/yr)",
            "predicted_risk": "Predicted Risk Score",
            "risk_tier": "Tier",
        },
        height=380,
    )
    fig_sc.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(gridcolor="#e5e7eb"),
        yaxis=dict(gridcolor="#e5e7eb", range=[0, 100]),
        margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(fig_sc, use_container_width=True)

with col_feat:
    st.subheader("Feature Importances")
    fi = (
        pd.Series(metrics["feature_importances"], name="importance")
        .sort_values(ascending=True)
        .reset_index()
    )
    fi.columns = ["Feature", "Importance"]
    fig_fi = px.bar(
        fi,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale=["#bfdbfe", "#1d4ed8"],
        labels={"Importance": "Relative Importance"},
        height=380,
    )
    fig_fi.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        coloraxis_showscale=False,
        xaxis=dict(gridcolor="#e5e7eb"),
        margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(fig_fi, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# Row 3 – S/N rate vs risk heatmap-style scatter
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("Spay/Neuter Rate vs Predicted Risk (by Income Band)")

filtered = filtered.copy()
filtered["income_band"] = pd.cut(
    filtered["median_household_income"],
    bins=[0, 40_000, 65_000, 90_000, 200_000],
    labels=["< $40k", "$40k–$65k", "$65k–$90k", "> $90k"],
)

fig_sn = px.scatter(
    filtered,
    x="prior_spay_neuter_rate",
    y="predicted_risk",
    color="income_band",
    symbol="risk_tier",
    size="total_intake",
    hover_name="zip_code",
    hover_data={"median_household_income": ":,", "total_intake": True},
    labels={
        "prior_spay_neuter_rate": "Prior Spay/Neuter Rate",
        "predicted_risk": "Predicted Risk Score",
        "income_band": "Income Band",
    },
    color_discrete_sequence=px.colors.qualitative.Bold,
    height=420,
)
fig_sn.update_layout(
    plot_bgcolor="white",
    paper_bgcolor="white",
    xaxis=dict(tickformat=".0%", gridcolor="#e5e7eb"),
    yaxis=dict(gridcolor="#e5e7eb", range=[0, 100]),
    margin=dict(l=10, r=10, t=30, b=10),
)
st.plotly_chart(fig_sn, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# Row 4 – Filtered data table
# ──────────────────────────────────────────────────────────────────────────────
st.subheader(f"📋 Filtered Zip Code Data ({len(filtered)} rows)")

display_cols = [
    "zip_code",
    "risk_tier",
    "predicted_risk",
    "risk_score",
    "total_intake",
    "prior_spay_neuter_rate",
    "adoption_rate",
    "median_household_income",
    "population_density",
    "vet_clinics_per_10k",
]

styled = filtered[display_cols].sort_values("predicted_risk", ascending=False).reset_index(drop=True)

st.dataframe(
    styled,
    use_container_width=True,
    column_config={
        "zip_code":               st.column_config.TextColumn("Zip Code"),
        "risk_tier":              st.column_config.TextColumn("Tier"),
        "predicted_risk":         st.column_config.ProgressColumn("Predicted Risk", min_value=0, max_value=100, format="%.1f"),
        "risk_score":             st.column_config.NumberColumn("Actual Risk", format="%.1f"),
        "total_intake":           st.column_config.NumberColumn("Total Intake"),
        "prior_spay_neuter_rate": st.column_config.NumberColumn("S/N Rate", format="%.2f"),
        "adoption_rate":          st.column_config.NumberColumn("Adoption Rate", format="%.2f"),
        "median_household_income":st.column_config.NumberColumn("Median Income", format="$%d"),
        "population_density":     st.column_config.NumberColumn("Pop. Density (sq-mi)"),
        "vet_clinics_per_10k":    st.column_config.NumberColumn("Vet Clinics/10k", format="%.2f"),
    },
    height=420,
)

# ──────────────────────────────────────────────────────────────────────────────
# Priority Action Plan – Top 5 High-Risk Zip Codes
# ──────────────────────────────────────────────────────────────────────────────
st.divider()
st.subheader("🗂️ Priority Action Plan — Top 5 High-Risk Zip Codes")

_ACTION_LOOKUP = {
    "🔴 High":   "Deploy Mobile Surgical Unit + TNR Trapper Teams",
    "🟡 Medium": "Distribute Vouchers + Door-to-Door Outreach",
    "🟢 Low":    "Standard Monitoring + Public Education",
}

_top5_src = scored_df[scored_df["risk_tier"] == "🔴 High"].nlargest(5, "predicted_risk")

if _top5_src.empty:
    st.info("No High-Risk zip codes found in the current dataset.")
else:
    _top5 = _top5_src[["zip_code", "predicted_risk", "risk_tier"]].copy()
    _top5["recommended_action"]   = _top5["risk_tier"].map(_ACTION_LOOKUP)
    _top5["est_cost_savings"]     = _INTAKE_PREVENTED_PER_ZIP * _COST_PER_ANIMAL  # per zip
    _top5 = _top5[["zip_code", "predicted_risk", "recommended_action", "est_cost_savings"]].reset_index(drop=True)

    st.dataframe(
        _top5,
        use_container_width=True,
        column_config={
            "zip_code":           st.column_config.TextColumn("Zip Code"),
            "predicted_risk":     st.column_config.ProgressColumn(
                                      "Predicted Risk Score", min_value=0, max_value=100, format="%.1f"
                                  ),
            "recommended_action": st.column_config.TextColumn("Recommended Action"),
            "est_cost_savings":   st.column_config.NumberColumn(
                                      "Est. Cost Savings / Zip", format="$%d"
                                  ),
        },
        hide_index=True,
        height=245,
    )

# ──────────────────────────────────────────────────────────────────────────────
# Row 5 – Predict on custom input
# ──────────────────────────────────────────────────────────────────────────────
st.divider()
st.subheader("🔮 Predict Risk for a Custom Zip Code")

with st.form("custom_predict"):
    c1, c2, c3, c4 = st.columns(4)
    stray_i   = c1.number_input("Stray Intake",        min_value=0,   max_value=2000, value=200)
    owner_s   = c1.number_input("Owner Surrender",     min_value=0,   max_value=2000, value=100)
    adopt_r   = c2.slider("Adoption Rate",             0.0,  1.0,  0.55,  0.01)
    rto_r     = c2.slider("Return-to-Owner Rate",      0.0,  1.0,  0.10,  0.01)
    income    = c3.number_input("Median HH Income ($)", min_value=10_000, max_value=200_000, value=55_000, step=1_000)
    pop_den   = c3.number_input("Pop. Density (sq-mi)", min_value=100, max_value=50_000, value=3_000, step=100)
    sn_rate   = c4.slider("Prior S/N Rate",            0.0,  1.0,  0.40,  0.01)
    vet_c     = c4.slider("Vet Clinics / 10k",         0.5,  8.0,  2.5,   0.1)
    litter    = c4.slider("Avg Litter Size",           2.0,  10.0, 5.0,   0.1)
    repeat_i  = c1.number_input("Repeat Intakes",      min_value=0,   max_value=500, value=30)

    submitted = st.form_submit_button("Predict ➜", type="primary")

if submitted:
    custom_row = pd.DataFrame(
        [{
            "stray_intake":            stray_i,
            "owner_surrender":         owner_s,
            "total_intake":            stray_i + owner_s,
            "adoption_rate":           adopt_r,
            "return_to_owner_rate":    rto_r,
            "median_household_income": income,
            "population_density":      pop_den,
            "prior_spay_neuter_rate":  sn_rate,
            "vet_clinics_per_10k":     vet_c,
            "avg_litter_size":         litter,
            "repeat_intakes":          repeat_i,
        }]
    )
    custom_result = predict(pipeline, custom_row)
    score = custom_result["predicted_risk"].iloc[0]
    tier  = custom_result["risk_tier"].iloc[0]

    tier_color = {"🔴 High": "#dc2626", "🟡 Medium": "#d97706", "🟢 Low": "#16a34a"}.get(tier, "#1f2328")

    col_res1, col_res2, col_res3 = st.columns(3)
    col_res1.metric("Predicted Risk Score", f"{score:.1f} / 100")
    col_res2.metric("Risk Tier", tier)
    col_res3.metric("Total Intake", stray_i + owner_s)

    # Gauge chart
    fig_gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": "Overpopulation Risk", "font": {"size": 16}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": tier_color},
                "steps": [
                    {"range": [0, 45],   "color": "#dcfce7"},
                    {"range": [45, 70],  "color": "#fef9c3"},
                    {"range": [70, 100], "color": "#fee2e2"},
                ],
                "threshold": {
                    "line": {"color": "#1f2328", "width": 3},
                    "thickness": 0.75,
                    "value": score,
                },
            },
            number={"suffix": " / 100", "font": {"size": 24}},
        )
    )
    fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=10), paper_bgcolor="white")
    st.plotly_chart(fig_gauge, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "⚠️ This dashboard uses **synthetic data** for demonstration purposes only. "
    "Model: Gradient Boosting Regressor (scikit-learn). "
    "Replace `data/shelter_data.csv` with real shelter intake records for production use."
)
