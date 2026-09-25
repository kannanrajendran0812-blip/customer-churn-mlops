import os

import pandas as pd
from sklearn.model_selection import train_test_split


RAW_DATA_PATH = "data/raw/Telco-Customer-Churn.csv"
PROCESSED_DIR = "data/processed"


def load_data():
    """Load the raw Telco Customer Churn dataset."""
    return pd.read_csv(RAW_DATA_PATH)


def preprocess_data(df):
    """Clean and prepare the raw dataset."""

    # Remove customer ID because it is an identifier,
    # not a useful predictive feature.
    df = df.drop(columns=["customerID"])

    # Convert TotalCharges from string to numeric.
    # Invalid/blank values become NaN.
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # Remove rows where TotalCharges could not be converted.
    df = df.dropna(subset=["TotalCharges"])

    # Convert target to binary.
    df["Churn"] = df["Churn"].map({
        "No": 0,
        "Yes": 1
    })

    return df


def save_splits(df):
    """Split data into train, validation and test datasets."""

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # 70% train, 30% temporary
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    # Split temporary 50/50:
    # 15% validation and 15% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp
    )

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    train = X_train.copy()
    train["Churn"] = y_train

    validation = X_val.copy()
    validation["Churn"] = y_val

    test = X_test.copy()
    test["Churn"] = y_test

    train.to_csv(
        f"{PROCESSED_DIR}/train.csv",
        index=False
    )

    validation.to_csv(
        f"{PROCESSED_DIR}/validation.csv",
        index=False
    )

    test.to_csv(
        f"{PROCESSED_DIR}/test.csv",
        index=False
    )

    print("Data preprocessing completed.")
    print(f"Train shape: {train.shape}")
    print(f"Validation shape: {validation.shape}")
    print(f"Test shape: {test.shape}")


if __name__ == "__main__":
    data = load_data()
    data = preprocess_data(data)
    save_splits(data)