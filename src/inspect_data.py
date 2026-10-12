from pathlib import Path
import pandas as pd
DATA_Path = Path("data/wdbc.data")
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
COLUMNS=["id","diagnosis"]+FEATURES

def main():
    if not DATA_Path.exists():
        raise FileNotFoundError(
            f"Dataset not found:{DATA_Path.resolve()}\n"
            "Place wdbc.data inside the data folder."
        )

    df = pd.read_csv(
        DATA_Path,
        header=None,
        names=COLUMNS
    )

    print("\n -----dataset shape-----")
    print(df.shape)

    print("\n-----first five records-----")
    print(df.head())

    print("\n-----Column data types-----")
    print(df.dtypes)

    print("\n-----missing values--------")
    print(df.isnull().sum().sum())

    print("\n -----duplicate records------")
    print(df.duplicated().sum())

    print("\n-----Diagnosis counts-------")
    print(df["diagnosis"].value_counts())

    print("\n=== DIAGNOSIS PERCENTAGES ===")
    print(
        (df["diagnosis"].value_counts(normalize=True) * 100).round(2)
    )

    print("\n=== NUMERICAL FEATURE SUMMARY ===")
    print(df[FEATURES].describe().round(2))
    print("\n=== VALIDATION CHECKS ===")
    print("Expected columns:", len(COLUMNS))
    print("Actual columns:", len(df.columns))
    print("Diagnosis labels:", sorted(df["diagnosis"].dropna().unique()))
    all_numeric = (
        df[FEATURES].select_dtypes(include="number").shape[1] == 30
    )
    print("All 30 features numeric:", all_numeric)


if __name__ == "__main__":
    main()
