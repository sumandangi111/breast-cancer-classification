
from pathlib import Path

import joblib
import pandas as pd

from .report_loader import load_report
from .validate_report import validate_report

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "breast_cancer_pipeline.joblib"

LABELS = {
    0: "Benign",
    1: "Malignant",
}


def main():
    print("\n" + "=" * 55)
    print("       BREAST CANCER REPORT ANALYSIS")
    print("=" * 55)
    print("Educational prototype — not for medical diagnosis.\n")

    if not MODEL_PATH.exists():
        print(f"Error: Model pipeline not found: {MODEL_PATH}")
        print("Run: python src/train.py")
        return

    file_path = input(
        "Enter the full path of a PDF or TXT report: "
    ).strip().strip('"')

    if not file_path:
        print("Error: No file path provided.")
        return

    try:
        report = load_report(file_path)

        print("\n" + "-" * 55)
        print("REPORT LOADED")
        print("-" * 55)
        print(f"File: {report['file_name']}")
        print(f"Type: {report['file_type']}")
        print(f"Characters extracted: {report['character_count']}")

        validation = validate_report(report["text"])

        print("\n" + "-" * 55)
        print("FEATURE VALIDATION")
        print("-" * 55)
        print(validation["message"])

        if not validation["valid"]:
            print("\nReport cannot be processed by this model.")
            print("No prediction was made.")
            return

        # Keep the feature names and order expected by the pipeline.
        feature_names = [
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

        features = pd.DataFrame(
            [validation["features"]],
            columns=feature_names,
        )

        pipeline = joblib.load(MODEL_PATH)
        prediction = int(pipeline.predict(features)[0])
        probabilities = pipeline.predict_proba(features)[0]

        print("\n" + "-" * 55)
        print("MODEL OUTPUT")
        print("-" * 55)
        print(f"Predicted class: {LABELS.get(prediction, 'Unknown')}")
        print(f"Benign class score: {probabilities[0]:.2%}")
        print(f"Malignant class score: {probabilities[1]:.2%}")

        print("\nImportant:")
        print("- This is an educational machine-learning demonstration.")
        print("- Scores are model outputs, not a clinical diagnosis.")
        print("- Do not use this result to make medical decisions.")
        print("- Consult a qualified healthcare professional.")

    except (FileNotFoundError, ValueError) as error:
        print(f"\nError: {error}")
    except OSError as error:
        print(f"\nFile access error: {error}")
    except Exception as error:
        print(f"\nUnexpected processing error: {error}")


if __name__ == "__main__":
    main()
