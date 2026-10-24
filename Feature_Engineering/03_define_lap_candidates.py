from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

BASE_PATH = (
    ROOT
    / "data/processed/model_features_2023_2026.csv"
)

ENHANCED_PATH = (
    ROOT
    / "data/processed/model_features_with_laps_2023_2026.csv"
)

OUTPUT = (
    ROOT
    / "results/feature_analysis/lap_feature_candidates.csv"
)


# =========================================================
# Load datasets
# =========================================================

base = pd.read_csv(
    BASE_PATH,
    low_memory=False
)

enhanced = pd.read_csv(
    ENHANCED_PATH,
    low_memory=False
)


# =========================================================
# Identify lap-derived features
# =========================================================

base_columns = set(base.columns)

lap_features = [
    column
    for column in enhanced.columns
    if column not in base_columns
]


# =========================================================
# Validate
# =========================================================

assert len(base) == 1602
assert len(enhanced) == 1602

assert len(lap_features) == 31


# =========================================================
# Categorise only — do NOT transform or remove anything
# =========================================================

historical_features = [
    feature
    for feature in lap_features
    if (
        feature.startswith("previous_")
        or feature.startswith("last3_")
    )
]

current_features = [
    feature
    for feature in lap_features
    if feature not in historical_features
]


rows = []

for feature in current_features:

    rows.append({
        "feature": feature,
        "feature_group": "current_weekend_pre_race",
        "decision": "candidate"
    })

for feature in historical_features:

    rows.append({
        "feature": feature,
        "feature_group": "historical_lap_form",
        "decision": "candidate"
    })


manifest = pd.DataFrame(rows)

manifest.to_csv(
    OUTPUT,
    index=False
)


# =========================================================
# Report
# =========================================================

print("\nLAP FEATURE CANDIDATES")
print("=" * 70)

print(
    f"Race-level predictors:       "
    f"{len(base.columns) - 5}"
)

print(
    f"Current pre-race lap features: "
    f"{len(current_features)}"
)

print(
    f"Historical lap features:       "
    f"{len(historical_features)}"
)

print(
    f"Total lap candidates:          "
    f"{len(lap_features)}"
)

print(
    f"Total enhanced candidates:     "
    f"{23 + len(lap_features)}"
)


print("\nCURRENT WEEKEND FEATURES")
print("=" * 70)

for feature in current_features:
    print(feature)


print("\nHISTORICAL LAP FEATURES")
print("=" * 70)

for feature in historical_features:
    print(feature)


print("\nNO FEATURES WERE TRANSFORMED OR REMOVED.")
print(
    "Feature selection will determine which "
    "representations are useful or redundant."
)

print("\nSaved:")
print(OUTPUT)
