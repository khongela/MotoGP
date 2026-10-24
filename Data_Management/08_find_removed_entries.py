from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

MASTER_PATH = ROOT / "data/interim/master.csv"
CLEANED_PATH = ROOT / "data/interim/stage2_cleaned_rider_races.csv"

master = pd.read_csv(MASTER_PATH, low_memory=False)
cleaned = pd.read_csv(CLEANED_PATH, low_memory=False)

# ---------------------------------------------------------
# Identify rows present in master but absent from cleaned
# ---------------------------------------------------------

KEYS = ["event_id", "rider_id"]

master_keys = master[KEYS].drop_duplicates()
cleaned_keys = cleaned[KEYS].drop_duplicates()

removed_keys = master_keys.merge(
    cleaned_keys,
    on=KEYS,
    how="left",
    indicator=True
)

removed_keys = removed_keys[
    removed_keys["_merge"] == "left_only"
][KEYS]

removed = master.merge(
    removed_keys,
    on=KEYS,
    how="inner"
)

print("\nREMOVED RIDER-RACE ENTRIES")
print("=" * 80)

print(f"Master rows:  {len(master):,}")
print(f"Cleaned rows: {len(cleaned):,}")
print(f"Removed rows: {len(removed):,}")

# ---------------------------------------------------------
# Show useful columns if available
# ---------------------------------------------------------

preferred_columns = [
    "season",
    "round",
    "event_date",
    "event_name",
    "circuit_name",
    "event_id",
    "rider_id",
    "rider_number",
    "rider",
    "position",
    "race_position",
    "classified_position",
    "status",
    "race_status",
    "points",
    "grid_position"
]

columns = [
    col for col in preferred_columns
    if col in removed.columns
]

# Add any status / position / result-related fields
for col in removed.columns:
    lower = col.lower()

    if any(term in lower for term in [
        "status",
        "position",
        "result",
        "finish",
        "classified"
    ]):
        if col not in columns:
            columns.append(col)

print("\nRemoved entries:")
print(
    removed[columns]
    .to_string(index=False)
)

print("\nALL MASTER COLUMNS")
print("-" * 80)

for i, col in enumerate(master.columns, 1):
    print(f"{i:2}. {col}")

# ---------------------------------------------------------
# Save audit
# ---------------------------------------------------------

OUT = ROOT / "results/feature_analysis"
OUT.mkdir(parents=True, exist_ok=True)

removed.to_csv(
    OUT / "removed_rider_race_entries.csv",
    index=False
)

print(
    "\nSaved:",
    OUT / "removed_rider_race_entries.csv"
)
