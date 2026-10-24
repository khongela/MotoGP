
from pathlib import Path
import re
import unicodedata
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

LAPS_PATH = ROOT / "data/processed/laps.csv"
RACES_PATH = ROOT / "data/processed/model_features_2023_2026.csv"
EVENTS_PATH = ROOT / "results/feature_analysis/event_mapping.csv"
OUT = ROOT / "results/feature_analysis"

OUT.mkdir(parents=True, exist_ok=True)


def normalize_name(value):
    """Remove case, accent and spacing differences."""
    if pd.isna(value):
        return ""

    value = unicodedata.normalize("NFKD", str(value))
    value = "".join(
        c for c in value
        if not unicodedata.combining(c)
    )
    return re.sub(r"[^a-z0-9]", "", value.lower())


# ---------------------------------------------------------
# Load datasets
# ---------------------------------------------------------

laps = pd.read_csv(LAPS_PATH, low_memory=False)
races = pd.read_csv(RACES_PATH, low_memory=False)
events = pd.read_csv(EVENTS_PATH)

# Validate event mapping
assert events["event_id"].is_unique
assert not events["circuit_code"].isna().any()

assert not events.duplicated(
    ["season", "circuit_code"]
).any()

# ---------------------------------------------------------
# Associate lap records with events
# ---------------------------------------------------------

lap_riders = laps[
    ["year", "circuit", "rider", "rider_number"]
].drop_duplicates()

lap_riders = lap_riders.merge(
    events[
        ["event_id", "season", "circuit_code"]
    ],
    left_on=["year", "circuit"],
    right_on=["season", "circuit_code"],
    how="left",
    validate="many_to_one"
)

if lap_riders["event_id"].isna().any():
    raise ValueError(
        "Some lap records could not be assigned to an event."
    )

lap_riders["name_key"] = (
    lap_riders["rider"].map(normalize_name)
)

# ---------------------------------------------------------
# Prepare race rider identities
# ---------------------------------------------------------

race_riders = races[
    ["event_id", "rider_id", "rider"]
].drop_duplicates().copy()

race_riders["name_key"] = (
    race_riders["rider"].map(normalize_name)
)

# Match using event and normalised rider name
matches = lap_riders.merge(
    race_riders[
        ["event_id", "rider_id", "rider", "name_key"]
    ].rename(columns={"rider": "race_rider"}),
    on=["event_id", "name_key"],
    how="left",
    validate="many_to_one"
)

# ---------------------------------------------------------
# Identify unresolved riders
# ---------------------------------------------------------

unmatched = matches[
    matches["rider_id"].isna()
].copy()

matched = matches[
    matches["rider_id"].notna()
].copy()

print("\nRIDER MAPPING SUMMARY")
print("-" * 65)

print(f"Lap rider identities: {len(lap_riders):,}")
print(f"Matched identities: {len(matched):,}")
print(f"Unmatched identities: {len(unmatched):,}")

# Detect mappings to multiple rider IDs within an event
ambiguous = (
    matched.groupby(
        ["event_id", "rider_number"]
    )["rider_id"]
    .nunique()
)

ambiguous = ambiguous[ambiguous > 1]

print(f"Ambiguous rider numbers: {len(ambiguous)}")

# ---------------------------------------------------------
# Save audit
# ---------------------------------------------------------

matches.to_csv(
    OUT / "rider_mapping_audit.csv",
    index=False
)

unmatched.to_csv(
    OUT / "unmatched_lap_riders.csv",
    index=False
)

matched.to_csv(
    OUT / "rider_mapping.csv",
    index=False
)

print("\nFiles saved:")
print(OUT / "rider_mapping_audit.csv")
print(OUT / "unmatched_lap_riders.csv")
print(OUT / "rider_mapping.csv")

if not unmatched.empty:
    print("\nUNMATCHED RIDERS")
    print("-" * 65)
    print(
        unmatched[
            ["year", "circuit", "rider", "rider_number"]
        ].drop_duplicates().to_string(index=False)
    )

if len(ambiguous):
    print("\nAMBIGUOUS RIDER NUMBERS")
    print(ambiguous.to_string())

if unmatched.empty and ambiguous.empty:
    print("\nRIDER MAPPING VALIDATION PASSED")
else:
    print("\nRIDER MAPPING REQUIRES REVIEW")
    raise SystemExit(1)

