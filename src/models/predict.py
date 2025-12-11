import os
import sys
from typing import Dict, Any

import joblib
import pandas as pd

# -------------------------------------------------------------------
# 1) Ensure project root is on sys.path
# -------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.features.build_features import build_features  # noqa: E402

# -------------------------------------------------------------------
# 2) Constants – must match training code
# -------------------------------------------------------------------
NUM_FEATURES = [
    "Size_in_SqFt",
    "Age_of_Property",
    "Nearby_Schools",
    "Nearby_Hospitals",
    "calc_price_per_sqft",
    "Annual_Growth_Rate",
    "Future_Price_5Y",
]

CAT_FEATURES = ["City", "Locality", "Property_Type", "BHK"]

ALL_FEATURES = NUM_FEATURES + CAT_FEATURES

CLASSIFIER_PATH = os.path.join(PROJECT_ROOT, "models", "classifier_pipeline.pkl")
REGRESSOR_PATH = os.path.join(PROJECT_ROOT, "models", "regression_pipeline.pkl")

# -------------------------------------------------------------------
# 3) Lazy loaders – load once, reuse
# -------------------------------------------------------------------
_classifier_model = None
_regression_model = None


def _load_classifier():
    global _classifier_model
    if _classifier_model is None:
        if not os.path.exists(CLASSIFIER_PATH):
            raise FileNotFoundError(
                f"Classifier model file not found at {CLASSIFIER_PATH}. "
                f"Run train_classification.py first."
            )
        _classifier_model = joblib.load(CLASSIFIER_PATH)
    return _classifier_model


def _load_regressor():
    global _regression_model
    if _regression_model is None:
        if not os.path.exists(REGRESSOR_PATH):
            raise FileNotFoundError(
                f"Regression model file not found at {REGRESSOR_PATH}. "
                f"Run train_regression.py first."
            )
        _regression_model = joblib.load(REGRESSOR_PATH)
    return _regression_model


# -------------------------------------------------------------------
# 4) Core prediction function for a SINGLE property
# -------------------------------------------------------------------
def predict_property_investment(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run both classification + regression models for a single property.

    Parameters
    ----------
    features : dict
        Keys must include at least:
        - "City", "Locality", "Property_Type", "BHK"
        - "Size_in_SqFt", "Age_of_Property",
          "Nearby_Schools", "Nearby_Hospitals",
          "calc_price_per_sqft", "Annual_Growth_Rate", "Future_Price_5Y"

        Extra keys are ignored.

    Returns
    -------
    dict with:
        - good_investment_label (int 0/1)
        - good_investment_prob (float 0–1)
        - predicted_price_lakhs (float)
    """

    # 1) Build a single-row DataFrame
    row = {col: features.get(col) for col in ALL_FEATURES}
    df = pd.DataFrame([row])

    # 2) Apply same feature engineering as training
    df = build_features(df)

    # 3) Slice to the exact columns used by the pipelines
    X = df[ALL_FEATURES]

    # 4) Load models
    clf = _load_classifier()
    reg = _load_regressor()

    # 5) Classification prediction
    good_prob = float(clf.predict_proba(X)[0, 1])
    good_label = int(clf.predict(X)[0])

    # 6) Regression prediction (price in Lakhs)
    predicted_price = float(reg.predict(X)[0])

    return {
        "good_investment_label": good_label,
        "good_investment_prob": good_prob,
        "predicted_price_lakhs": predicted_price,
    }


# -------------------------------------------------------------------
# 5) Quick CLI test (optional)
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Dumb sanity check with fake values. Replace with a real row if you want.
    sample = {
        "City": "Hyderabad",
        "Locality": "Test Locality",
        "Property_Type": "Apartment",
        "BHK": "3",
        "Size_in_SqFt": 1500,
        "Age_of_Property": 10,
        "Nearby_Schools": 5,
        "Nearby_Hospitals": 3,
        "calc_price_per_sqft": 12000,
        "Annual_Growth_Rate": 0.09,
        "Future_Price_5Y": 400.0,
    }

    out = predict_property_investment(sample)
    print(out)
