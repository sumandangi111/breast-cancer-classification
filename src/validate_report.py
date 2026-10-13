import math
import re


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


def validate_report(report_text: str) -> dict:
    """Validate explicitly named numeric fields in report text."""

    if not report_text or not report_text.strip():
        return {
            "valid": False,
            "message": "The report contains no readable text.",
            "features": None,
        }

    values = {}

    # Capture feature values, including invalid numeric strings such as nan
    # and inf, so they can be rejected explicitly.
    pattern = re.compile(
        r"^\s*([a-z_]+)\s*[:=]\s*(.*?)\s*$",
        re.IGNORECASE,
    )

    for line in report_text.splitlines():
        match = pattern.match(line)

        if not match:
            continue

        name = match.group(1).lower()

        # Ignore unrelated fields such as patient IDs and report titles.
        if name not in FEATURES:
            continue

        raw_value = match.group(2)

        try:
            value = float(raw_value)
        except ValueError:
            return {
                "valid": False,
                "message": f"{name} must be a numeric value.",
                "features": None,
            }

        if not math.isfinite(value):
            return {
                "valid": False,
                "message": f"{name} must be a finite number.",
                "features": None,
            }

        if name in values:
            return {
                "valid": False,
                "message": f"Duplicate feature: {name}.",
                "features": None,
            }

        values[name] = value

    missing = [name for name in FEATURES if name not in values]

    if missing:
        return {
            "valid": False,
            "message": (
                f"Missing {len(missing)} of {len(FEATURES)} required features. "
                "This report cannot be used by the current model. "
                "Missing: " + ", ".join(missing)
            ),
            "features": None,
        }

    ordered_values = [values[name] for name in FEATURES]

    return {
        "valid": True,
        "message": "All 30 named numeric fields were found.",
        "features": ordered_values,
    }
