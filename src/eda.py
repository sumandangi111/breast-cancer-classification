
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = Path("data/wdbc.data")
REPORTS_PATH = Path("reports")


FEATURES = [
    "radius_mean", "texture_mean", "perimeter_mean",
    "area_mean", "smoothness_mean", "compactness_mean",
    "concavity_mean", "concave_points_mean", "symmetry_mean",
    "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se",
    "smoothness_se", "compactness_se", "concavity_se",
    "concave_points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst",
    "area_worst", "smoothness_worst", "compactness_worst",
    "concavity_worst", "concave_points_worst",
    "symmetry_worst", "fractal_dimension_worst"
]

COLUMNS = ["id", "diagnosis"] + FEATURES
df = pd.read_csv(DATA_PATH, header=None, names=COLUMNS)

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH.resolve()}"
        )

    REPORTS_PATH.mkdir(parents=True, exist_ok=True)

    

    # 1. Diagnosis distribution
    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x="diagnosis")
    plt.title("Benign vs Malignant Cases")
    plt.xlabel("Diagnosis (B = Benign, M = Malignant)")
    plt.ylabel("Number of Cases")
    plt.tight_layout()
    plt.savefig(REPORTS_PATH / "diagnosis_distribution.png")
    plt.close()

    # 2. Compare radius_mean across diagnosis groups
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="diagnosis", y="radius_mean")
    plt.title("Mean Radius by Diagnosis")
    plt.xlabel("Diagnosis")
    plt.ylabel("Radius Mean")
    plt.tight_layout()
    plt.savefig(REPORTS_PATH / "radius_by_diagnosis.png")
    plt.close()

    # 3. Correlation between numerical features
    plt.figure(figsize=(16, 12))
    correlation = df[FEATURES].corr()
    sns.heatmap(correlation, cmap="coolwarm", center=0)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(REPORTS_PATH / "correlation_heatmap.png")
    plt.close()

    print("EDA completed successfully.")
    print(f"Charts saved in: {REPORTS_PATH.resolve()}")


if __name__ == "__main__":
    main()
