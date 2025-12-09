#src/preprocessing.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from config import DATA_PATH, TARGET_COL, RANDOM_STATE, TEST_SIZE


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load dataset from CSV file."""
    df = pd.read_csv(path)
    return df


def split_features_target(df: pd.DataFrame, target_col: str = TARGET_COL):
    """Split dataframe into features X and target y."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def encode_categorical_features(X: pd.DataFrame):
    """
    Encode all object-type columns using LabelEncoder.

    Returns:
        X_encoded: dataframe with encoded categorical features
        encoders: dict of fitted LabelEncoders per column
    """
    X_encoded = X.copy()
    encoders = {}

    cat_cols = X_encoded.select_dtypes(include=["object"]).columns

    for col in cat_cols:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col])
        encoders[col] = le

    return X_encoded, encoders


def encode_target(y: pd.Series):
    """
    Encode target column using LabelEncoder.

    Returns:
        y_encoded: encoded target
        le_target: the fitted LabelEncoder
    """
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    return y_encoded, le


def train_test_scale(X: pd.DataFrame, y: pd.Series):
    """
    Perform train-test split and feature scaling.

    Returns:
        X_train, X_test, y_train, y_test,
        X_train_scaled, X_test_scaled, scaler
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler


def preprocess_dataframe(df: pd.DataFrame):
    """
    Full preprocessing pipeline:
    - split X, y
    - encode categorical features
    - train-test split
    - scaling

    Returns:
        X_train, X_test, y_train, y_test,
        X_train_scaled, X_test_scaled, scaler, encoders
    """
    X, y = split_features_target(df)
    y_encoded, le_target = encode_target(y)

    X_encoded, encoders = encode_categorical_features(X)
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler = train_test_scale(X_encoded, y_encoded)

    return (
    X_train, X_test, y_train, y_test,
    X_train_scaled, X_test_scaled,
    scaler, encoders, le_target
    )



if __name__ == "__main__":
    # Quick self-check when running this file directly
    df = load_data()
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler, encoders, le_target = preprocess_dataframe(df)

    print("preprocessing.py OK – pipeline runs without errors.")
