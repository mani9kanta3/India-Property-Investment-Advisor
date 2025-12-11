import os
import sys
import joblib


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import mlflow
import mlflow.sklearn

# Make sure project root is on sys.path when running as a script
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

TARGET = "Good_Investment"


def main():
    # -----------------------------
    # 1. Load & feature engineering
    # -----------------------------
    df = pd.read_csv(DATA_PATH)
    df = build_features(df)

    X = df[NUM_FEATURES + CAT_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # -----------------------------
    # 2. Build preprocessing + model pipeline
    # -----------------------------
    preprocessor = get_preprocessing_pipeline(NUM_FEATURES, CAT_FEATURES)

    clf = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
    )

    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", clf),
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
    y_proba = model_pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)

    print("Classification metrics:")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1-score : {f1:.4f}")
    print(f"  ROC-AUC  : {roc:.4f}")

    # -----------------------------
    # 5. Log to MLflow
    # -----------------------------
    mlflow.set_experiment("india_property_investment_classification")

    with mlflow.start_run():
        # Log parameters (basic)
        mlflow.log_param("model_type", "XGBClassifier")
        mlflow.log_param("n_estimators", clf.n_estimators)
        mlflow.log_param("max_depth", clf.max_depth)
        mlflow.log_param("learning_rate", clf.learning_rate)

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc)

        # Log model
        mlflow.sklearn.log_model(model_pipeline, "model")

        print("Model and metrics logged to MLflow.")


    # -----------------------------------------------------------
    # SAVE MODEL LOCALLY (for Streamlit inference)
    # -----------------------------------------------------------
        models_dir = os.path.join(PROJECT_ROOT, "models")
        os.makedirs(models_dir, exist_ok=True)

        clf_path = os.path.join(models_dir, "classifier_pipeline.pkl")
        joblib.dump(model_pipeline, clf_path)

        print(f"Saved classification pipeline to: {clf_path}")
    # -----------------------------------------------------------



if __name__ == "__main__":
    main()
