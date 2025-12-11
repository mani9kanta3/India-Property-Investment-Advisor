import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Property Insights Dashboard", layout="wide")

st.title("🏡 Property Market Insights Dashboard")


# ---------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------
@st.cache_data
def load_data():
    # Go from /pages to project root
    root = Path(__file__).resolve().parent.parent
    data_path = root / "data" / "processed" / "india_housing_with_targets.csv"

    df = pd.read_csv(data_path)
    return df


try:
    df = load_data()
except FileNotFoundError:
    st.error(
        "Processed dataset not found at 'data/processed/india_housing_with_targets.csv'. "
        "Make sure this file exists locally."
    )
    st.stop()

# Optional small info
st.markdown(f"**Rows:** {len(df):,} | **Columns:** {len(df.columns)}")
st.dataframe(df.head(), use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------
# 2. Price Distribution
# ---------------------------------------------------------
st.subheader("💰 Price Distribution (₹ Lakhs)")

fig = px.histogram(
    df,
    x="Price_in_Lakhs",
    nbins=50,
    title="Distribution of Property Prices (Lakhs)",
)
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# 3. Avg Rs/sqft by City
# ---------------------------------------------------------
st.subheader("📍 Average Rs/sq.ft by City")

city_price = (
    df.groupby("City")["calc_price_per_sqft"]
    .mean()
    .reset_index()
    .sort_values("calc_price_per_sqft", ascending=False)
)

fig = px.bar(
    city_price,
    x="City",
    y="calc_price_per_sqft",
    title="Average Price per Sq.ft by City",
)
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# 4. Property Type Distribution
# ---------------------------------------------------------
st.subheader("🏡 Property Type Distribution")

fig = px.pie(
    df,
    names="Property_Type",
    title="Share of Property Types",
)
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# 5. Age of Property Distribution
# ---------------------------------------------------------
st.subheader("⏳ Age of Property Distribution")

fig = px.histogram(
    df,
    x="Age_of_Property",
    nbins=30,
    title="Distribution of Property Age (Years)",
)
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# 6. Good Investment % by Property Type
# ---------------------------------------------------------
st.subheader("⭐ Good Investment Rate by Property Type")

gi_type = df.groupby("Property_Type")["Good_Investment"].mean().reset_index()
gi_type["Good_Investment"] = gi_type["Good_Investment"] * 100

fig = px.bar(
    gi_type,
    x="Property_Type",
    y="Good_Investment",
    title="Good Investment Rate (%) by Property Type",
)
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# 7. Good Investment % by City
# ---------------------------------------------------------
st.subheader("🏙️ Good Investment Rate by City")

gi_city = df.groupby("City")["Good_Investment"].mean().reset_index()
gi_city["Good_Investment"] = gi_city["Good_Investment"] * 100

fig = px.bar(
    gi_city,
    x="City",
    y="Good_Investment",
    title="Good Investment Rate (%) by City",
)
st.plotly_chart(fig, use_container_width=True)