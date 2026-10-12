
from pathlib import Path

import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "wdbc.data"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "breast_cancer_pipeline.joblib"

FEATURES = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
    "smoothness_mean", "compactness_mean", "concavity_mean",
    "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se",
    "smoothness_se", "compactness_se", "concavity_se",
    "concave_points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
    "smoothness_worst", "compactness_worst", "concavity_worst",
    "concave_points_worst", "symmetry_worst", "fractal_dimension_worst",
]

COLUMNS = ["id", "diagnosis"] + FEATURES


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Load the original, unscaled dataset.
    df = pd.read_csv(RAW_DATA_PATH, header=None, names=COLUMNS)

    # Validate the dataset before training.
    if df.isnull().values.any():
        raise ValueError("Dataset contains missing values.")

    X = df[FEATURES]
    y = df["diagnosis"].map({"B": 0, "M": 1})

    if y.isnull().any():
        raise ValueError("Dataset contains unexpected diagnosis labels.")

    # Keep the same deterministic, stratified split.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # Scaling is fitted on training data only.
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=2000,
            random_state=42,
        )),
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("Pipeline training completed.")
    print(f"Training records: {len(X_train)}")
    print(f"Test records: {len(X_test)}")
    print(f"\nTest accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    print("\nClassification report:")
    print(classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Benign", "Malignant"],
        zero_division=0,
    ))

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nPipeline saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
