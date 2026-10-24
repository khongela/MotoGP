
from pathlib import Path
import pandas as pd
import unicodedata
import re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/feature_analysis"
OUT.mkdir(parents=True, exist_ok=True)

FILES = {
    "master": ROOT / "data/interim/master.csv",
    "cleaned": ROOT / "data/interim/stage2_cleaned_rider_races.csv",
    "model": ROOT / "data/processed/model_features_2023_2026.csv",
}

def normalize(value):
    if pd.isna(value):
        return ""
    value = unicodedata.normalize("NFKD", str(value))
    value = "".join(
        c for c in value
        if not unicodedata.combining(c)
    )
    return re.sub(r"[^a-z0-9]", "", value.lower())


# Riders to investigate
RIDERS = [
    "Fabio Di Giannantonio",
    "Francesco Bagnaia",
    "Joan Mir",
    "Alex Rins",
    "Enea Bastianini",
    "Daniel Pedrosa",
]

print("\nSOURCE DATASET COMPARISON")
print("=" * 65)

for name, path in FILES.items():

    df = pd.read_csv(path, low_memory=False)

    print(f"\n{name.upper()}")
    print("-" * 65)
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    rider_columns = [
        c for c in df.columns
        if "rider" in c.lower() or "name" in c.lower()
    ]

    print(f"Rider/name columns: {rider_columns}")

    for rider in RIDERS:
        key = normalize(rider)
        count = 0

        for col in rider_columns:
            values = df[col].map(normalize)
            count += int((values == key).sum())

        print(f"{rider}: {count} matching cells")


# ---------------------------------------------------------
# Inspect modelling dataset coverage
# ---------------------------------------------------------

df = pd.read_csv(FILES["model"])

print("\nRACE DATASET COVERAGE")
print("=" * 65)

season_summary = df.groupby("season").agg(
    observations=("rider_id", "size"),
    races=("event_id", "nunique"),
    riders=("rider_id", "nunique"),
    podiums=("podium_finish", "sum")
)

season_summary["entries_per_race"] = (
    season_summary["observations"] /
    season_summary["races"]
)

print(season_summary.to_string())

# ---------------------------------------------------------
# Check podium labels per race
# ---------------------------------------------------------

race_summary = df.groupby(
    ["season", "round", "event_id"]
).agg(
    entries=("rider_id", "size"),
    unique_riders=("rider_id", "nunique"),
    podiums=("podium_finish", "sum")
).reset_index()

print("\nRACE-LEVEL TARGET VALIDATION")
print("=" * 65)

print("Races:", len(race_summary))

print("\nPodium counts per race:")
print(
    race_summary["podiums"]
    .value_counts()
    .sort_index()
    .to_string()
)

problems = race_summary[
    (race_summary["podiums"] != 3) |
    (race_summary["entries"] != race_summary["unique_riders"])
]

print("\nRaces requiring review:")
if problems.empty:
    print("No duplicate riders or unusual podium counts.")
else:
    print(problems.to_string(index=False))

# ---------------------------------------------------------
# Inspect selected rider-race entries
# ---------------------------------------------------------

checks = [
    (2023, 11, "Francesco Bagnaia"),
    (2023, 2, "Joan Mir"),
    (2023, 16, "Alex Rins"),
]

print("\nSPECIFIC RACE-ENTRY CHECKS")
print("=" * 65)

df["name_key"] = df["rider"].map(normalize)

for season, round_number, rider in checks:
    present = df[
        (df["season"] == season) &
        (df["round"] == round_number) &
        (df["name_key"] == normalize(rider))
    ]

    print(
        f"{season} round {round_number}: "
        f"{rider} -> {len(present)} entries"
    )

# ---------------------------------------------------------
# Save audit
# ---------------------------------------------------------

season_summary.to_csv(
    OUT / "race_dataset_season_summary.csv"
)

race_summary.to_csv(
    OUT / "race_entry_validation.csv",
    index=False
)

print("\nRACE DATASET AUDIT COMPLETE")

