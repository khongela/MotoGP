
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LAPS_PATH = ROOT / "data/processed/laps.csv"
MODEL_PATH = ROOT / "data/processed/model_features_2023_2026.csv"

laps = pd.read_csv(LAPS_PATH, low_memory=False)
model = pd.read_csv(MODEL_PATH, low_memory=False)

print("\n" + "=" * 65)
print("1. LAP DATASET OVERVIEW")
print("=" * 65)

print(f"Rows: {len(laps):,}")
print(f"Columns: {len(laps.columns)}")

print("\nColumn names and types:")
print(laps.dtypes.to_string())

print("\nFirst 5 records:")
print(laps.head().to_string(index=False))

print("\n" + "=" * 65)
print("2. SESSION AND EVENT INFORMATION")
print("=" * 65)

for col in laps.columns:
    if any(term in col.lower() for term in [
        "session", "type", "category", "season", "year"
    ]):
        print(f"\n{col}:")
        print(
            laps[col].value_counts(dropna=False)
            .head(30).to_string()
        )

print("\n" + "=" * 65)
print("3. MISSING VALUES")
print("=" * 65)

missing = pd.DataFrame({
    "missing": laps.isna().sum(),
    "percent": (laps.isna().mean() * 100).round(2)
})

print(
    missing.sort_values("percent", ascending=False)
    .to_string()
)

print("\n" + "=" * 65)
print("4. LAP VALIDITY AND TIMING FIELDS")
print("=" * 65)

terms = [
    "lap", "sector", "speed", "time",
    "cancel", "pit", "tyre", "tire", "valid"
]

for col in laps.columns:
    if any(term in col.lower() for term in terms):
        print(f"{col}: {laps[col].dtype}")

print("\n" + "=" * 65)
print("5. DATASET LINKING")
print("=" * 65)

shared = sorted(set(laps.columns) & set(model.columns))

print("\nColumns shared by both datasets:")
print(shared)

print("\nLikely identifiers in lap dataset:")
for col in laps.columns:
    if any(term in col.lower() for term in [
        "event", "rider", "session", "date", "round"
    ]):
        print(
            f"{col}: {laps[col].nunique(dropna=True):,} "
            "unique values"
        )

print("\n" + "=" * 65)
print("6. DUPLICATE ROWS")
print("=" * 65)

print(f"Exact duplicates: {laps.duplicated().sum():,}")

print("\n" + "=" * 65)
print("7. EXISTING MODEL DATA")
print("=" * 65)

print(f"Rider-race observations: {len(model):,}")
print(f"Unique races: {model['event_id'].nunique():,}")

print("\nINSPECTION COMPLETE")

