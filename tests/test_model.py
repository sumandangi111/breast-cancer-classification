
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


def test_saved_pipeline_exists():
    assert MODEL_PATH.exists()


def test_pipeline_predicts_valid_labels():
    pipeline = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH, header=None, names=COLUMNS)

    predictions = pipeline.predict(df[FEATURES].iloc[:5])

    assert len(predictions) == 5
    assert set(predictions).issubset({0, 1})


def test_pipeline_probabilities_are_valid():
    pipeline = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH, header=None, names=COLUMNS)

    probabilities = pipeline.predict_proba(df[FEATURES].iloc[:5])

    assert probabilities.shape == (5, 2)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    assert abs(probabilities.sum(axis=1) - 1).max() < 1e-9
