from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

BASE_PATH = ROOT / "data/processed/model_features_2023_2026.csv"
LAP_PATH = ROOT / "data/processed/model_features_with_laps_2023_2026.csv"

OUT_DIR = ROOT / "results/feature_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# Load datasets
# =========================================================

base = pd.read_csv(BASE_PATH, low_memory=False)
df = pd.read_csv(LAP_PATH, low_memory=False)

df["event_date"] = pd.to_datetime(df["event_date"])


# =========================================================
# 1. Structural validation
# =========================================================

print("\nSTRUCTURAL VALIDATION")
print("=" * 70)

print(f"Base rows:       {len(base):,}")
print(f"Enhanced rows:   {len(df):,}")
print(f"Base columns:    {len(base.columns)}")
print(f"Enhanced cols:   {len(df.columns)}")

assert len(base) == 1602
assert len(df) == 1602

duplicates = df.duplicated(
    ["event_id", "rider_id"]
).sum()

print(f"Duplicate rider-races: {duplicates}")

assert duplicates == 0


# =========================================================
# 2. Target validation
# =========================================================

target_match = (
    df["podium_finish"]
    .reset_index(drop=True)
    .equals(
        base["podium_finish"]
        .reset_index(drop=True)
    )
)

print(f"Podium target unchanged: {target_match}")

assert target_match


# =========================================================
# 3. Identify new lap-derived features
# =========================================================

original_columns = set(base.columns)

lap_features = [
    column
    for column in df.columns
    if column not in original_columns
]

print("\nNEW LAP FEATURES")
print("=" * 70)

for feature in lap_features:
    print(feature)

print(f"\nTotal new lap features: {len(lap_features)}")

assert len(lap_features) == 31, (
    f"Expected 31 lap-derived features, found {len(lap_features)}"
)


# =========================================================
# 4. Separate current and historical lap features
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

print("\nFEATURE GROUPS")
print("=" * 70)

print(
    f"Current-weekend pre-race features: "
    f"{len(current_features)}"
)

print(
    f"Historical lagged features: "
    f"{len(historical_features)}"
)


# =========================================================
# 5. Coverage
# =========================================================

coverage_rows = []

for feature in lap_features:

    available = df[feature].notna().sum()
    missing = df[feature].isna().sum()

    coverage_rows.append({
        "feature": feature,
        "available": available,
        "missing": missing,
        "coverage_pct":
            available / len(df) * 100,
    })

coverage = pd.DataFrame(
    coverage_rows
).sort_values(
    "coverage_pct",
    ascending=False
)

print("\nLAP FEATURE COVERAGE")
print("=" * 70)

print(
    coverage.to_string(
        index=False,
        formatters={
            "coverage_pct":
                lambda value: f"{value:.1f}%"
        }
    )
)


# =========================================================
# 6. Coverage by season
# =========================================================

season_rows = []

for season, season_df in df.groupby("season"):

    for feature in lap_features:

        available = season_df[feature].notna().sum()

        season_rows.append({
            "season": season,
            "feature": feature,
            "rows": len(season_df),
            "available": available,
            "coverage_pct":
                available / len(season_df) * 100,
        })

season_coverage = pd.DataFrame(season_rows)


important_features = [
    "best_practice_lap_s",
    "gap_to_best_practice_s",
    "best_quali_lap_s",
    "gap_to_best_quali_s",
    "longrun_median_s",
    "longrun_rank",
    "pre_race_top_speed_rank",
    "previous_race_pace_rank",
    "last3_race_pace_rank_mean",
]

important_features = [
    feature
    for feature in important_features
    if feature in df.columns
]

print("\nIMPORTANT FEATURE COVERAGE BY SEASON")
print("=" * 70)

if important_features:

    table = (
        season_coverage[
            season_coverage["feature"]
            .isin(important_features)
        ]
        .pivot(
            index="feature",
            columns="season",
            values="coverage_pct"
        )
        .round(1)
    )

    print(table.to_string())


# =========================================================
# 7. Validate historical shifting
# =========================================================

first_races = (
    df
    .sort_values("event_date")
    .groupby("rider_id")
    .head(1)
)

print("\nHISTORICAL LAG VALIDATION")
print("=" * 70)

lag_errors = []

for feature in historical_features:

    non_null = (
        first_races[feature]
        .notna()
        .sum()
    )

    print(
        f"{feature:40s} "
        f"non-null on rider first race: "
        f"{non_null}"
    )

    if non_null > 0:
        lag_errors.append(feature)

if lag_errors:

    print("\nWARNING:")
    print(
        "Historical values exist on some riders' "
        "first observed races."
    )

    print(
        "Features requiring review:",
        lag_errors
    )

else:

    print(
        "\nAll historical lap features correctly "
        "start after each rider's first observed race."
    )


# =========================================================
# 8. Target correlations
# =========================================================

numeric_lap_features = [
    feature
    for feature in lap_features
    if pd.api.types.is_numeric_dtype(
        df[feature]
    )
]

correlation_rows = []

for feature in numeric_lap_features:

    subset = df[
        [feature, "podium_finish"]
    ].dropna()

    if len(subset) < 10:
        continue

    correlation = (
        subset[feature]
        .corr(
            subset["podium_finish"]
        )
    )

    correlation_rows.append({
        "feature": feature,
        "target_correlation": correlation,
        "observations": len(subset),
    })

correlations = pd.DataFrame(
    correlation_rows
)

if len(correlations):

    correlations["abs_correlation"] = (
        correlations[
            "target_correlation"
        ].abs()
    )

    correlations = correlations.sort_values(
        "abs_correlation",
        ascending=False
    )

print("\nSTRONGEST LAP FEATURE / TARGET RELATIONSHIPS")
print("=" * 70)

if len(correlations):

    print(
        correlations[
            [
                "feature",
                "target_correlation",
                "observations",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )


# =========================================================
# 9. Suspicious correlations
# =========================================================

print("\nVERY STRONG CORRELATIONS (> |0.80|)")
print("=" * 70)

if len(correlations):

    suspicious = correlations[
        correlations[
            "abs_correlation"
        ] > 0.80
    ]

    if len(suspicious):

        print(
            suspicious[
                [
                    "feature",
                    "target_correlation",
                    "observations",
                ]
            ].to_string(index=False)
        )

    else:
        print("None detected.")

else:
    print("None detected.")


# =========================================================
# 10. Explicit leakage check
# =========================================================

FORBIDDEN_MODEL_FEATURES = [
    "finish_position",
    "status",
    "race_points",
    "round_points",
    "podium_finish",
]

print("\nLEAKAGE CHECK")
print("=" * 70)

print(
    "These columns remain in source tables "
    "but must never be passed into model X:"
)

for column in FORBIDDEN_MODEL_FEATURES:
    print(f" - {column}")


# =========================================================
# 11. Save audit outputs
# =========================================================

coverage.to_csv(
    OUT_DIR /
    "lap_feature_coverage.csv",
    index=False
)

season_coverage.to_csv(
    OUT_DIR /
    "lap_feature_season_coverage.csv",
    index=False
)

if len(correlations):

    correlations.to_csv(
        OUT_DIR /
        "lap_feature_target_correlations.csv",
        index=False
    )

pd.DataFrame({
    "feature": current_features,
    "group":
        "current_weekend_pre_race"
}).to_csv(
    OUT_DIR /
    "current_pre_race_feature_list.csv",
    index=False
)

pd.DataFrame({
    "feature": historical_features,
    "group":
        "historical_lagged"
}).to_csv(
    OUT_DIR /
    "historical_lap_feature_list.csv",
    index=False
)


print("\nVALIDATION COMPLETE")
print("=" * 70)

print(
    "Enhanced dataset row count, target integrity, "
    "feature coverage and historical shifting checked."
)
