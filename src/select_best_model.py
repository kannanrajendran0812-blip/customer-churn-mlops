import os
import shutil
import pandas as pd


COMPARISON_FILE = "reports/model_comparison.csv"
MODEL_DIR = "models"

MODEL_FILES = {
    "logistic-regression": "models/logistic_regression.joblib",
    "random-forest": "models/random_forest.joblib",
    "xgboost": "models/xgboost.joblib",
    "gradient-boosting": "models/gradient_boosting.joblib",
}


def main():

    if not os.path.exists(COMPARISON_FILE):
        raise FileNotFoundError(
            f"Comparison file not found: {COMPARISON_FILE}"
        )

    results = pd.read_csv(COMPARISON_FILE)

    # Select model with highest ROC-AUC
    best_row = results.loc[
        results["roc_auc"].idxmax()
    ]

    best_model_name = best_row["model"]
    best_roc_auc = best_row["roc_auc"]

    print("\n" + "=" * 60)
    print("BEST MODEL SELECTION")
    print("=" * 60)

    print(f"Model   : {best_model_name}")
    print(f"ROC-AUC : {best_roc_auc:.4f}")

    source_model = MODEL_FILES.get(
        best_model_name
    )

    if source_model is None:
        raise ValueError(
            f"No model file configured for "
            f"{best_model_name}"
        )

    if not os.path.exists(source_model):
        raise FileNotFoundError(
            f"Model file not found: {source_model}"
        )

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    destination = os.path.join(
        MODEL_DIR,
        "best_model.joblib"
    )

    shutil.copy2(
        source_model,
        destination
    )

    print(
        f"Saved best model to: {destination}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()