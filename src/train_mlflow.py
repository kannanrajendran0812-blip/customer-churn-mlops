import os

import mlflow
import mlflow.sklearn
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
    include=["str", "object"]
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


def main():

    train, validation = load_data()

    X_train = train.drop(columns=["Churn"])
    y_train = train["Churn"]

    X_val = validation.drop(columns=["Churn"])
    y_val = validation["Churn"]

    preprocessor = create_preprocessor(X_train)

    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )

    # Create MLflow experiment
    mlflow.set_experiment(
        "customer-churn-classification"
    )

    with mlflow.start_run(
        run_name="logistic-regression"
    ):

        # Train
        pipeline.fit(
            X_train,
            y_train,
        )

        # Predictions
        predictions = pipeline.predict(
            X_val
        )

        probabilities = pipeline.predict_proba(
            X_val
        )[:, 1]

        # Metrics
        accuracy = accuracy_score(
            y_val,
            predictions,
        )

        precision = precision_score(
            y_val,
            predictions,
        )

        recall = recall_score(
            y_val,
            predictions,
        )

        f1 = f1_score(
            y_val,
            predictions,
        )

        roc_auc = roc_auc_score(
            y_val,
            probabilities,
        )

        # Log parameters
        mlflow.log_param(
            "model",
            "LogisticRegression",
        )

        mlflow.log_param(
            "max_iter",
            1000,
        )

        # Log metrics
        mlflow.log_metric(
            "accuracy",
            accuracy,
        )

        mlflow.log_metric(
            "precision",
            precision,
        )

        mlflow.log_metric(
            "recall",
            recall,
        )

        mlflow.log_metric(
            "f1",
            f1,
        )

        mlflow.log_metric(
            "roc_auc",
            roc_auc,
        )

        # Log model
        mlflow.sklearn.log_model(
            pipeline,
            name="model",
        )

        print("\nMLflow Run")
        print("==========")
        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1       : {f1:.4f}")
        print(f"ROC-AUC  : {roc_auc:.4f}")

        print(
            f"\nRun ID: {mlflow.active_run().info.run_id}"
        )


if __name__ == "__main__":
    main()