from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

DATA_PATH = Path("data/wdbc.data")
PROCESSED_PATH = Path("data/processed")
MODELS_PATH = Path("models")

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

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH.resolve()}"
        )

    PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
    MODELS_PATH.mkdir(parents=True, exist_ok=True)

    
    df = pd.read_csv(DATA_PATH, header=None, names=COLUMNS)

    
    df["diagnosis"] = df["diagnosis"].map({"B": 0, "M": 1})

    if df["diagnosis"].isna().any():
        raise ValueError("Unexpected or missing diagnosis labels.")

    
    X = df[FEATURES]
    y = df["diagnosis"]

    
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    
    pd.DataFrame(X_train_scaled, columns=FEATURES).to_csv(
        PROCESSED_PATH / "X_train.csv", index=False
    )
    pd.DataFrame(X_test_scaled, columns=FEATURES).to_csv(
        PROCESSED_PATH / "X_test.csv", index=False
    )
    y_train.to_csv(PROCESSED_PATH / "y_train.csv", index=False)
    y_test.to_csv(PROCESSED_PATH / "y_test.csv", index=False)

    
    joblib.dump(scaler, MODELS_PATH / "scaler.pkl")

    print("Preprocessing completed successfully.")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Training features: {X_train.shape[1]}")
    print("\nTraining diagnosis distribution:")
    print(y_train.value_counts().sort_index())

    print("\nOriginal diagnosis mapping:")
    print(df["diagnosis"].value_counts(dropna=False))
    print(f"\nProcessed files saved in: {PROCESSED_PATH.resolve()}")
    print(f"Scaler saved in: {MODELS_PATH.resolve()}")


if __name__ == "__main__":
    main()

