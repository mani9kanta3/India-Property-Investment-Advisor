import math

import streamlit as st
import joblib
from pathlib import Path

# -------------------------------------------------------
# Load models safely for Streamlit Cloud
# -------------------------------------------------------
@st.cache_resource
def load_models():
    root = Path(__file__).parent
    clf = joblib.load(root / "models" / "classifier_pipeline.pkl")
    reg = joblib.load(root / "models" / "regression_pipeline.pkl")
    return clf, reg

clf_model, reg_model = load_models()


# -------------------------------------------------------
# Prediction helper function
# -------------------------------------------------------
def predict_property_investment(model_clf, model_reg, features):
    """features = dict of input values"""
    
    import pandas as pd
    
    df = pd.DataFrame([features])

    proba = model_clf.predict_proba(df)[0, 1]
    label = int(proba > 0.5)

    price_pred = model_reg.predict(df)[0]

    return {
        "good_investment_label": label,
        "good_investment_prob": float(proba),
        "predicted_price_lakhs": float(price_pred)
    }


# --------------------------------------------------
# Helper: pretty label from model output
# --------------------------------------------------
def format_investment_label(label: int) -> str:
    return "✅ GOOD investment" if label == 1 else "⚠️ RISKY / NOT attractive"


# --------------------------------------------------
# Streamlit app
# --------------------------------------------------
def main():
    st.set_page_config(
        page_title="India Property Investment Advisor",
        layout="wide",
    )

    st.title("🏠 India Property Investment Advisor")

    st.markdown("---")

    # ------------- INPUT FORM -------------
    with st.form("property_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            city = st.text_input("City", value="Hyderabad")
            locality = st.text_input("Locality", value="Madhapur")
            property_type = st.selectbox(
                "Property Type",
                ["Apartment", "Independent House", "Villa"],
            )

        with col2:
            bhk = st.selectbox("BHK", [1, 2, 3, 4, 5], index=2)
            size_sqft = st.number_input("Size (SqFt)", min_value=300, max_value=10000,
                                        value=1500, step=50)
            age = st.number_input("Age of Property (years)", min_value=0, max_value=50,
                                  value=10, step=1)

        with col3:
            nearby_schools = st.number_input("Nearby schools (within 5 km)",
                                             min_value=0, max_value=20, value=5, step=1)
            nearby_hospitals = st.number_input("Nearby hospitals (within 5 km)",
                                               min_value=0, max_value=20, value=3, step=1)
            price_lakhs = st.number_input("Asking price (₹ Lakhs)",
                                          min_value=10.0, max_value=1000.0,
                                          value=250.0, step=1.0)

        st.markdown("### Growth assumptions")

        colg1, colg2 = st.columns(2)
        with colg1:
            growth_pct = st.slider(
                "Expected annual growth rate (%)",
                min_value=5.0,
                max_value=12.0,
                value=8.5,
                step=0.1,
            )
        with colg2:
            horizon_years = st.slider(
                "Investment horizon (years)",
                min_value=3,
                max_value=10,
                value=5,
                step=1,
            )

        submitted = st.form_submit_button("Evaluate Investment 🚀")

    # ------------- PREDICTION -------------
    if submitted:
        if size_sqft <= 0:
            st.error("Size in SqFt must be > 0.")
            return

        # Derived features to match the training schema
        calc_price_per_sqft = (price_lakhs * 100000) / size_sqft

        annual_growth_rate = growth_pct / 100.0
        future_price_5y = price_lakhs * math.pow(1 + annual_growth_rate, horizon_years)

        # Build feature dict expected by predict_property_investment()
        features = {
            "City": city,
            "Locality": locality,
            "Property_Type": property_type,
            "BHK": str(bhk),
            "Size_in_SqFt": size_sqft,
            "Age_of_Property": age,
            "Nearby_Schools": nearby_schools,
            "Nearby_Hospitals": nearby_hospitals,
            "calc_price_per_sqft": calc_price_per_sqft,
            "Annual_Growth_Rate": annual_growth_rate,
            "Future_Price_5Y": future_price_5y,
        }

        try:
            result = predict_property_investment(clf_model, reg_model, features)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            return

        good_label = result["good_investment_label"]
        good_prob = result["good_investment_prob"]
        predicted_price = result["predicted_price_lakhs"]

        # ------------- OUTPUT LAYOUT -------------
        st.markdown("---")
        st.subheader("📊 Investment Assessment")

        col_left, col_right = st.columns([2, 1])

        with col_left:
            st.markdown(f"**Overall verdict:** {format_investment_label(good_label)}")
            st.metric(
                label="Probability of being a good investment",
                value=f"{good_prob * 100:.1f} %",
            )

        with col_right:
            st.metric(
                label="Model fair price (₹ Lakhs)",
                value=f"{predicted_price:.1f}",
            )

        # Under / over-valuation
        delta = predicted_price - price_lakhs
        st.markdown("### 💰 Valuation vs Asking Price")
        if delta > 0:
            st.success(
                f"Model thinks this property is **UNDER-valued** by "
                f"~₹ {abs(delta):.1f} Lakhs vs your asking price."
            )
        elif delta < 0:
            st.warning(
                f"Model thinks this property is **OVER-valued** by "
                f"~₹ {abs(delta):.1f} Lakhs vs your asking price."
            )
        else:
            st.info("Model fair price is roughly equal to the asking price.")

        # Show debug details
        with st.expander("Show model input features (debug)"):
            st.write(features)
            st.write(result)


if __name__ == "__main__":
    main()
