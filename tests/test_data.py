
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "wdbc.data"

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


def load_dataset():
    return pd.read_csv(DATA_PATH, header=None, names=COLUMNS)


def test_dataset_has_expected_shape():
    df = load_dataset()
    assert df.shape == (569, 32)


def test_dataset_has_expected_labels():
    df = load_dataset()
    assert set(df["diagnosis"].unique()) == {"B", "M"}


def test_features_are_numeric():
    df = load_dataset()
    assert df[FEATURES].select_dtypes(include="number").shape[1] == 30


def test_dataset_has_no_missing_values():
    df = load_dataset()
    assert not df.isnull().values.any()
