from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    recall_score,
)
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "wdbc.data"
MODEL_PATH = PROJECT_ROOT / "models" / "breast_cancer_pipeline.joblib"
REPORTS_PATH = PROJECT_ROOT / "reports"

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
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Pipeline not found. Run python src/train.py first."
        )

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    REPORTS_PATH.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH, header=None, names=COLUMNS)
    X = df[FEATURES]
    y = df["diagnosis"].map({"B": 0, "M": 1})

    if y.isnull().any():
        raise ValueError("Unexpected diagnosis labels in dataset.")

    # Recreate exactly the same held-out test split used in train.py.
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    pipeline = joblib.load(MODEL_PATH)
    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    malignant_recall = recall_score(
        y_test, y_pred, pos_label=1, zero_division=0
    )

    print("=== PIPELINE EVALUATION ===")
    print(f"Test records: {len(X_test)}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy:.2%})")
    print(f"Malignant recall: {malignant_recall:.4f} ({malignant_recall:.2%})")

    print("\nClassification report:")
    print(classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Benign", "Malignant"],
        zero_division=0,
    ))

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Benign", "Malignant"],
        yticklabels=["Benign", "Malignant"],
    )
    plt.xlabel("Predicted diagnosis")
    plt.ylabel("Actual diagnosis")
    plt.title("Breast Cancer Classification - Confusion Matrix")
    plt.tight_layout()

    output_path = REPORTS_PATH / "confusion_matrix.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"\nConfusion matrix saved to: {output_path}")


if __name__ == "__main__":
    main()

