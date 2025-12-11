import pandas as pd

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Core feature engineering applied consistently across
    training and prediction.
    """

    df = df.copy()

    # Ensure numeric types (safety for inference)
    numeric_cols = [
        "Size_in_SqFt", "Age_of_Property", "Nearby_Schools",
        "Nearby_Hospitals", "calc_price_per_sqft",
        "Annual_Growth_Rate", "Future_Price_5Y"
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Convert BHK to categorical string
    df["BHK"] = df["BHK"].astype(str)

    # Convert other categorical columns
    cat_cols = ["City", "Locality", "Property_Type"]
    for col in cat_cols:
        df[col] = df[col].astype(str)

    return df