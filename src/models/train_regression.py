import os
import sys
import joblib

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
import mlflow
import mlflow.sklearn

# Ensure project root on path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.features.build_features import build_features
from src.models.preprocessing import get_preprocessing_pipeline


# -----------------------------
# CONFIG
# -----------------------------
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "india_housing_with_targets.csv")

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

TARGET = "Price_in_Lakhs"


def main():
    # -----------------------------
    # 1. Load & feature engineering
    # -----------------------------
    df = pd.read_csv(DATA_PATH)
    df = build_features(df)

    X = df[NUM_FEATURES + CAT_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -----------------------------
    # 2. Build preprocessing + model pipeline
    # -----------------------------
    preprocessor = get_preprocessing_pipeline(NUM_FEATURES, CAT_FEATURES)

    reg = XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )

    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", reg),
        ]
    )

    # -----------------------------
    # 3. Train model
    # -----------------------------
    model_pipeline.fit(X_train, y_train)

   # -----------------------------
    # 4. Evaluate
    # -----------------------------
    y_pred = model_pipeline.predict(X_test)

    # Some sklearn versions don't support `squared` argument,
    # so we compute RMSE manually from MSE.
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Regression metrics:")
    print(f"  MSE  : {mse:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  R^2  : {r2:.4f}")


    # -----------------------------
    # 5. Log to MLflow
    # -----------------------------
    mlflow.set_experiment("india_property_investment_regression")

    with mlflow.start_run():
        mlflow.log_param("model_type", "XGBRegressor")
        mlflow.log_param("n_estimators", reg.n_estimators)
        mlflow.log_param("max_depth", reg.max_depth)
        mlflow.log_param("learning_rate", reg.learning_rate)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        mlflow.sklearn.log_model(model_pipeline, "model")

        print("Regression model and metrics logged to MLflow.")

    # -----------------------------------------------------------
    # SAVE MODEL LOCALLY (for Streamlit inference)
    # -----------------------------------------------------------
        models_dir = os.path.join(PROJECT_ROOT, "models")
        os.makedirs(models_dir, exist_ok=True)

        reg_path = os.path.join(models_dir, "regression_pipeline.pkl")
        joblib.dump(model_pipeline, reg_path)

        print(f"Saved regression pipeline to: {reg_path}")
    # -----------------------------------------------------------



if __name__ == "__main__":
    main()
