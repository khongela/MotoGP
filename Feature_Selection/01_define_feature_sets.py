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

OUT_DIR = ROOT / "results/feature_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)


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
# Columns that are not predictors
# =========================================================

IDENTIFIERS = [
    "event_id",
    "event_date",
    "rider_id",
    "rider",
]

TARGET = "podium_finish"


# =========================================================
# Dataset B: full 23 race-level predictors
# =========================================================

FULL_23 = [
    column
    for column in base.columns
    if column not in IDENTIFIERS + [TARGET]
]

assert len(FULL_23) == 23, (
    f"Expected 23 race-level predictors, found {len(FULL_23)}"
)


# =========================================================
# Dataset A: selected 15 race-level predictors
# =========================================================

SELECTED_15 = [
    "grid_position",
    "qualifying_gap_s",
    "team_name",
    "manufacturer",
    "physical_circuit",
    "season",
    "round",
    "sprint_position",
    "sprint_status",
    "points_before_race",
    "prior_season_starts",
    "last_race_finish_position",
    "last3_podium_rate",
    "last3_dnf_rate",
    "last3_race_points_mean",
]

assert len(SELECTED_15) == 15

for feature in SELECTED_15:
    assert feature in base.columns, (
        f"Selected feature missing from base dataset: {feature}"
    )


# =========================================================
# Identify 31 lap-derived predictors
# =========================================================

base_columns = set(base.columns)

LAP_31 = [
    column
    for column in enhanced.columns
    if column not in base_columns
]

assert len(LAP_31) == 31, (
    f"Expected 31 lap-derived predictors, found {len(LAP_31)}"
)


# =========================================================
# Dataset C: full race-level + lap-derived
# =========================================================

ENHANCED_54 = FULL_23 + LAP_31

assert len(ENHANCED_54) == 54, (
    f"Expected 54 enhanced predictors, found {len(ENHANCED_54)}"
)

assert len(set(ENHANCED_54)) == 54, (
    "Duplicate predictor found in enhanced feature set."
)


# =========================================================
# Check all features exist
# =========================================================

for feature in FULL_23:
    assert feature in base.columns

for feature in LAP_31:
    assert feature in enhanced.columns

for feature in ENHANCED_54:
    assert feature in enhanced.columns


# =========================================================
# Save feature-set manifests
# =========================================================

pd.DataFrame({
    "feature": SELECTED_15
}).to_csv(
    OUT_DIR / "feature_set_selected_15.csv",
    index=False
)

pd.DataFrame({
    "feature": FULL_23
}).to_csv(
    OUT_DIR / "feature_set_full_23.csv",
    index=False
)

pd.DataFrame({
    "feature": LAP_31
}).to_csv(
    OUT_DIR / "feature_set_lap_31.csv",
    index=False
)

pd.DataFrame({
    "feature": ENHANCED_54
}).to_csv(
    OUT_DIR / "feature_set_enhanced_54.csv",
    index=False
)


# =========================================================
# Report
# =========================================================

print("\nFEATURE SET DEFINITIONS")
print("=" * 70)

print(f"Dataset A - selected race-level: {len(SELECTED_15)}")
print(f"Dataset B - full race-level:     {len(FULL_23)}")
print(f"Lap-derived predictors:          {len(LAP_31)}")
print(f"Dataset C - lap-enhanced:        {len(ENHANCED_54)}")


print("\nDATASET A - SELECTED 15")
print("=" * 70)

for feature in SELECTED_15:
    print(feature)


print("\nDATASET B - FULL 23")
print("=" * 70)

for feature in FULL_23:
    print(feature)


print("\nLAP-DERIVED 31")
print("=" * 70)

for feature in LAP_31:
    print(feature)


print("\nFINAL COMPARISON")
print("=" * 70)

print("Dataset A: 15 selected race-level predictors")
print("Dataset B: 23 full race-level predictors")
print("Dataset C: 54 predictors")
print("           = 23 race-level + 31 lap-derived")

print("\nNo model training performed.")
print("No additional feature engineering performed.")

print("\nFEATURE SET DEFINITION COMPLETE")
