
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

laps = pd.read_csv(
    ROOT / "data/processed/laps.csv",
    low_memory=False
)

races = pd.read_csv(
    ROOT / "data/processed/model_features_2023_2026.csv"
)

print("\nLAP DATASET CIRCUITS")
print("-" * 60)

lap_events = (
    laps[["year", "circuit"]]
    .drop_duplicates()
    .sort_values(["year", "circuit"])
)

print(lap_events.to_string(index=False))
print(f"\nUnique year-circuit combinations: {len(lap_events)}")

print("\nRACE DATASET CIRCUITS")
print("-" * 60)

race_events = (
    races[
        ["season", "round", "event_date", "physical_circuit"]
    ]
    .drop_duplicates()
    .sort_values(["season", "round"])
)

print(race_events.to_string(index=False))
print(f"\nUnique race events: {len(race_events)}")

print("\nRIDER NAME COMPARISON")
print("-" * 60)

lap_riders = set(laps["rider"].dropna().unique())
race_riders = set(races["rider"].dropna().unique())

print(f"Lap dataset riders: {len(lap_riders)}")
print(f"Race dataset riders: {len(race_riders)}")

print("\nNames only in lap dataset:")
print(sorted(lap_riders - race_riders))

print("\nNames only in race dataset:")
print(sorted(race_riders - lap_riders))

print("\nSESSION COVERAGE BY SEASON")
print("-" * 60)

print(
    pd.crosstab(laps["year"], laps["session"])
    .to_string()
)

