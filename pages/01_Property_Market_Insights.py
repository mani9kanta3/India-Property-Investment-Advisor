import pandas as pd
import plotly.express as px
import streamlit as st

# ------------------------------------
# Page config (controls browser tab title)
# Sidebar label is controlled by the file name in /pages/
# ------------------------------------
st.set_page_config(page_title="Property Market Insights", layout="wide")

st.title("🏡 Property Market Insights Dashboard")

# ------------------------------------
# Helpers
# ------------------------------------
@st.cache_data
def load_data():
    # Full dataset from repo (make sure this file exists in Streamlit Cloud)
    return pd.read_csv("data/processed/india_housing_with_targets.csv")


def format_indian_number(n: int) -> str:
    """Indian grouping: 12,34,56,789"""
    n = int(round(n))
    s = str(abs(n))
    if len(s) <= 3:
        out = s
    else:
        last3 = s[-3:]
        rest = s[:-3]
        parts = []
        while len(rest) > 2:
            parts.insert(0, rest[-2:])
            rest = rest[:-2]
        if rest:
            parts.insert(0, rest)
        out = ",".join(parts) + "," + last3
    return "-" + out if n < 0 else out


def format_inr(n: float, decimals: int = 0) -> str:
    """₹ with Indian grouping. Decimals supported."""
    neg = n < 0
    n = abs(float(n))
    if decimals == 0:
        s = format_indian_number(int(round(n)))
    else:
        whole = int(n)
        frac = round(n - whole, decimals)
        frac_str = f"{frac:.{decimals}f}".split(".")[1]
        s = f"{format_indian_number(whole)}.{frac_str}"
    return f"-₹ {s}" if neg else f"₹ {s}"


def safe_mean(series: pd.Series, default: float = 0.0) -> float:
    return float(series.mean()) if len(series) else default


# ------------------------------------
# Load data
# ------------------------------------
df = load_data()

# Basic cleanup for safety
df["City"] = df["City"].astype(str)
df["Locality"] = df["Locality"].astype(str)
df["Property_Type"] = df["Property_Type"].astype(str)
df["BHK"] = df["BHK"].astype(str)

# ------------------------------------
# Sidebar Filters (dropdown with "All")
# ------------------------------------
st.sidebar.header("🔎 Filters")

cities = ["All"] + sorted(df["City"].unique().tolist())
types = ["All"] + sorted(df["Property_Type"].unique().tolist())
bhks = ["All"] + sorted(df["BHK"].unique().tolist())

sel_city = st.sidebar.selectbox("City", cities, index=0)
sel_type = st.sidebar.selectbox("Property Type", types, index=0)
sel_bhk = st.sidebar.selectbox("BHK", bhks, index=0)

price_min, price_max = float(df["Price_in_Lakhs"].min()), float(df["Price_in_Lakhs"].max())
size_min, size_max = int(df["Size_in_SqFt"].min()), int(df["Size_in_SqFt"].max())
age_min, age_max = int(df["Age_of_Property"].min()), int(df["Age_of_Property"].max())

price_range = st.sidebar.slider(
    "Asking Price (₹ Lakhs)",
    min_value=float(round(price_min, 2)),
    max_value=float(round(price_max, 2)),
    value=(float(round(price_min, 2)), float(round(price_max, 2))),
)

size_range = st.sidebar.slider(
    "Size (SqFt)",
    min_value=int(size_min),
    max_value=int(size_max),
    value=(int(size_min), int(size_max)),
)

age_range = st.sidebar.slider(
    "Age of Property (Years)",
    min_value=int(age_min),
    max_value=int(age_max),
    value=(int(age_min), int(age_max)),
)

good_only = st.sidebar.checkbox("Show only Good Investments", value=False)

# ------------------------------------
# Apply filters (defaults show FULL DATA)
# ------------------------------------
df_f = df.copy()

if sel_city != "All":
    df_f = df_f[df_f["City"] == sel_city]

if sel_type != "All":
    df_f = df_f[df_f["Property_Type"] == sel_type]

if sel_bhk != "All":
    df_f = df_f[df_f["BHK"] == sel_bhk]

df_f = df_f[df_f["Price_in_Lakhs"].between(*price_range)]
df_f = df_f[df_f["Size_in_SqFt"].between(*size_range)]
df_f = df_f[df_f["Age_of_Property"].between(*age_range)]

if good_only:
    df_f = df_f[df_f["Good_Investment"] == 1]

# ------------------------------------
# Make KPI cards bigger + cleaner (no "Key Metrics" heading)
# ------------------------------------
st.markdown(
    """
    <style>
    /* Make metric values bigger */
    div[data-testid="stMetricValue"] { font-size: 2.1rem; font-weight: 800; }
    div[data-testid="stMetricLabel"] { font-size: 1.05rem; font-weight: 650; opacity: 0.9; }
    </style>
    """,
    unsafe_allow_html=True,
)

avg_price_lakhs = safe_mean(df_f["Price_in_Lakhs"])
avg_rs_sqft = safe_mean(df_f["calc_price_per_sqft"])
good_rate = safe_mean(df_f["Good_Investment"]) * 100.0
listings = int(len(df_f))

c1, c2, c3, c4 = st.columns(4)
c1.metric("Average Price", f"₹ {avg_price_lakhs:.1f} Lakhs")
c2.metric("Avg Price / SqFt", f"{format_inr(avg_rs_sqft, 0)}")
c3.metric("Good Investment Rate", f"{good_rate:.1f}%")
c4.metric("Listings", format_indian_number(listings))

st.markdown("---")

# ------------------------------------
# Charts (use FILTERED df_f everywhere)
# ------------------------------------

# 1) Price Distribution
st.subheader("💰 Price Distribution (₹ Lakhs)")
fig = px.histogram(
    df_f,
    x="Price_in_Lakhs",
    nbins=50,
)
st.plotly_chart(fig, use_container_width=True)

# 2) Avg Rs/sqft by City (Top 15 to keep readable)
st.subheader("📍 Average Price per SqFt by City (Top 15)")
city_price = (
    df_f.groupby("City")["calc_price_per_sqft"]
    .mean()
    .reset_index()
    .sort_values("calc_price_per_sqft", ascending=False)
    .head(15)
)
fig = px.bar(
    city_price,
    x="City",
    y="calc_price_per_sqft",
)
st.plotly_chart(fig, use_container_width=True)

# 3) Property Type Distribution
st.subheader("🏡 Property Type Share")
type_counts = df_f["Property_Type"].value_counts().reset_index()
type_counts.columns = ["Property_Type", "Count"]
fig = px.pie(
    type_counts,
    names="Property_Type",
    values="Count",
)
st.plotly_chart(fig, use_container_width=True)

# 4) Age of Property Distribution
st.subheader("⏳ Age of Property Distribution (Years)")
fig = px.histogram(
    df_f,
    x="Age_of_Property",
    nbins=30,
)
st.plotly_chart(fig, use_container_width=True)

# 5) Good Investment Rate by Property Type
st.subheader("⭐ Good Investment Rate by Property Type")
gi_type = df_f.groupby("Property_Type")["Good_Investment"].mean().reset_index()
gi_type["Good_Investment"] = gi_type["Good_Investment"] * 100
fig = px.bar(
    gi_type,
    x="Property_Type",
    y="Good_Investment",
)
st.plotly_chart(fig, use_container_width=True)

# 6) Good Investment Rate by City (Top 15)
st.subheader("🏙️ Good Investment Rate by City (Top 15)")
gi_city = df_f.groupby("City")["Good_Investment"].mean().reset_index()
gi_city["Good_Investment"] = gi_city["Good_Investment"] * 100
gi_city = gi_city.sort_values("Good_Investment", ascending=False).head(15)
fig = px.bar(
    gi_city,
    x="City",
    y="Good_Investment",
)
st.plotly_chart(fig, use_container_width=True)
