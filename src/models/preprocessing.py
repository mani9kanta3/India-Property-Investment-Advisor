from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline


def get_preprocessing_pipeline(numeric_features, categorical_features):
    """
    Build preprocessing pipeline for numeric + categorical features.

    Parameters
    ----------
    numeric_features : list of str
        Names of numeric columns to scale.
    categorical_features : list of str
        Names of categorical columns to one-hot encode.

    Returns
    -------
    preprocessor : ColumnTransformer
        A sklearn ColumnTransformer that applies scaling and one-hot encoding.
    """

    # Pipeline for numeric columns: StandardScaler
    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler())
        ]
    )

    # Pipeline for categorical columns: OneHotEncoder
    categorical_transformer = Pipeline(
        steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    # Combine into a single ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor