
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "wdbc.data"
MODEL_PATH = PROJECT_ROOT / "models" / "breast_cancer_pipeline.joblib"

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
            "Pipeline not found. Run 'python src/train.py' first."
        )

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    pipeline = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH, header=None, names=COLUMNS)

    # Select one row only for a demonstration.
    sample = df.iloc[[0]]
    X_sample = sample[FEATURES]

    prediction = int(pipeline.predict(X_sample)[0])
    probabilities = pipeline.predict_proba(X_sample)[0]

    labels = {0: "Benign", 1: "Malignant"}

    print("Breast Cancer Classification - Demo")
    print("-" * 40)
    print(f"Sample ID: {sample['id'].iloc[0]}")
    print(f"Predicted class: {labels[prediction]}")
    print(f"Benign probability: {probabilities[0]:.2%}")
    print(f"Malignant probability: {probabilities[1]:.2%}")
    print(f"Dataset label: {sample['diagnosis'].iloc[0]}")
    print("\nDemo only — not for medical diagnosis.")


if __name__ == "__main__":
    main()
