import os

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


TRAIN_PATH = "data/processed/train.csv"
VALIDATION_PATH = "data/processed/validation.csv"

MODEL_DIR = "models"


def load_data():
    train = pd.read_csv(TRAIN_PATH)
    validation = pd.read_csv(VALIDATION_PATH)

    return train, validation


def create_preprocessor(X):
    numerical_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                StandardScaler(),
                numerical_features,
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_features,
            ),
        ]
    )

    return preprocessor


def train_model():
    train, validation = load_data()

    X_train = train.drop(columns=["Churn"])
    y_train = train["Churn"]

    X_val = validation.drop(columns=["Churn"])
    y_val = validation["Churn"]

    preprocessor = create_preprocessor(X_train)

    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            ),
        ]
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_val)
    probabilities = pipeline.predict_proba(X_val)[:, 1]

    metrics = {
        "accuracy": accuracy_score(
            y_val,
            predictions
        ),
        "precision": precision_score(
            y_val,
            predictions
        ),
        "recall": recall_score(
            y_val,
            predictions
        ),
        "f1": f1_score(
            y_val,
            predictions
        ),
        "roc_auc": roc_auc_score(
            y_val,
            probabilities
        ),
    }

    print("\nModel Performance")
    print("=================")

    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    os.makedirs(MODEL_DIR, exist_ok=True)

    model_path = (
        f"{MODEL_DIR}/logistic_regression.joblib"
    )

    joblib.dump(
        pipeline,
        model_path
    )

    print(
        f"\nModel saved to: {model_path}"
    )


if __name__ == "__main__":
    train_model()