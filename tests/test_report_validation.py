import pytest

from src.validate_report import FEATURES, validate_report


def make_valid_sample():
    """Create synthetic feature data for software testing only."""
    return "\n".join(
        f"{name}: {index + 0.5}"
        for index, name in enumerate(FEATURES)
    )


def test_empty_report_is_rejected():
    result = validate_report("")

    assert result["valid"] is False
    assert result["features"] is None


def test_report_missing_features_is_rejected():
    result = validate_report("Patient ID: TEST-001")

    assert result["valid"] is False
    assert "Missing 30 of 30" in result["message"]


def test_all_named_numeric_fields_are_recognized():
    result = validate_report(make_valid_sample())

    assert result["valid"] is True
    assert len(result["features"]) == 30


def test_feature_order_matches_training_order():
    result = validate_report(make_valid_sample())

    expected = [index + 0.5 for index in range(30)]
    assert result["features"] == expected


def test_duplicate_feature_is_rejected():
    text = make_valid_sample() + "\nradius_mean: 2.0"
    result = validate_report(text)

    assert result["valid"] is False
    assert "Duplicate feature" in result["message"]


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_non_finite_feature_is_rejected(value):
    text = make_valid_sample().replace(
        "radius_mean: 0.5",
        f"radius_mean: {value}",
    )

    result = validate_report(text)

    assert result["valid"] is False
    assert "finite number" in result["message"]
