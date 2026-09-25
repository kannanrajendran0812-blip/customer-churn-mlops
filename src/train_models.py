import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


TRAIN_PATH = "data/processed/train.csv"
VALIDATION_PATH = "data/processed/validation.csv"


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


def evaluate_model(
    model_name,
    model,
    X_train,
    y_train,
    X_val,
    y_val,
):

    preprocessor = create_preprocessor(X_train)

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

    with mlflow.start_run(
        run_name=model_name
    ):

        # Train
        pipeline.fit(
            X_train,
            y_train
        )

        # Predict
        predictions = pipeline.predict(
            X_val
        )

        probabilities = pipeline.predict_proba(
            X_val
        )[:, 1]

        # Metrics
        accuracy = accuracy_score(
            y_val,
            predictions
        )

        precision = precision_score(
            y_val,
            predictions
        )

        recall = recall_score(
            y_val,
            predictions
        )

        f1 = f1_score(
            y_val,
            predictions
        )

        roc_auc = roc_auc_score(
            y_val,
            probabilities
        )

        # Log parameters
        mlflow.log_param(
            "model",
            model_name
        )

        # Log metrics
        mlflow.log_metric(
            "accuracy",
            accuracy
        )

        mlflow.log_metric(
            "precision",
            precision
        )

        mlflow.log_metric(
            "recall",
            recall
        )

        mlflow.log_metric(
            "f1",
            f1
        )

        mlflow.log_metric(
            "roc_auc",
            roc_auc
        )

        # Log model
        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            skops_trusted_types=[
                "sklearn.tree._tree.Tree",
                "xgboost.core.Booster",
                "xgboost.sklearn.XGBClassifier",
            ]
        )
                # Save model locally
        os.makedirs("models", exist_ok=True)

        model_filename = (
            model_name.replace("-", "_")
            + ".joblib"
        )

        model_path = os.path.join(
            "models",
            model_filename
        )

        joblib.dump(
            pipeline,
            model_path
        )

        print(
            f"Model saved to: {model_path}"
        )

        print(
            f"\n{model_name}"
        )

        print(
            f"Accuracy : {accuracy:.4f}"
        )

        print(
            f"Precision: {precision:.4f}"
        )

        print(
            f"Recall   : {recall:.4f}"
        )

        print(
            f"F1       : {f1:.4f}"
        )

        print(
            f"ROC-AUC  : {roc_auc:.4f}"
        )

        return {
            "model": model_name,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
        }


def main():

    train, validation = load_data()

    X_train = train.drop(
        columns=["Churn"]
    )

    y_train = train["Churn"]

    X_val = validation.drop(
        columns=["Churn"]
    )

    y_val = validation["Churn"]

    models = {

        "logistic-regression":
            LogisticRegression(
                max_iter=1000,
                random_state=42
            ),

        "random-forest":
            RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),

        "xgboost":
            XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                random_state=42,
                eval_metric="logloss",
                n_jobs=-1
            ),

        "gradient-boosting":
            GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=3,
                random_state=42
            ),
    }

    mlflow.set_experiment(
        "customer-churn-classification"
    )

    results = []

    for model_name, model in models.items():

        result = evaluate_model(
            model_name,
            model,
            X_train,
            y_train,
            X_val,
            y_val,
        )

        results.append(result)

    # Create comparison table
    results_df = pd.DataFrame(results)

    print("\n")
    print("=" * 70)
    print("MODEL PERFORMANCE COMPARISON")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False
        )
    )

    # Sort by ROC-AUC
    results_df = results_df.sort_values(
        by="roc_auc",
        ascending=False
    )

    print("\n")
    print("=" * 70)
    print("BEST MODEL")
    print("=" * 70)

    print(
        results_df.iloc[0].to_string()
    )

    # Save comparison
    results_df.to_csv(
        "reports/model_comparison.csv",
        index=False
    )

    print(
        "\nComparison saved to:"
        " reports/model_comparison.csv"
    )


if __name__ == "__main__":
    main()