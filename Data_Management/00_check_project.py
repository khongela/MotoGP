from pathlib import Path
import pandas as pd


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

EXPECTED_FILES = {
    "Master dataset":
        INTERIM_DIR / "master.csv",

    "Cleaned rider-race dataset":
        INTERIM_DIR / "stage2_cleaned_rider_races.csv",

    "Model feature dataset":
        PROCESSED_DIR / "model_features_2023_2026.csv",

    "Lap dataset":
        PROCESSED_DIR / "laps.csv",
}


# ---------------------------------------------------------
# Check files
# ---------------------------------------------------------

print("\nPROJECT ROOT")
print("-" * 60)
print(PROJECT_ROOT)

print("\nEXPECTED DATA FILES")
print("-" * 60)

for name, path in EXPECTED_FILES.items():

    status = "FOUND" if path.exists() else "MISSING"

    print(f"{status:8} | {name}")
    print(f"         {path}")


# ---------------------------------------------------------
# Inspect modelling dataset
# ---------------------------------------------------------

model_path = EXPECTED_FILES["Model feature dataset"]

if model_path.exists():

    print("\nMODEL DATASET")
    print("-" * 60)

    df = pd.read_csv(model_path)

    print(f"Rows:    {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    print("\nColumns:")
    for i, column in enumerate(df.columns, start=1):
        print(f"{i:2}. {column}")

    print("\nSeasons:")
    if "season" in df.columns:
        print(sorted(df["season"].dropna().unique()))

    print("\nDataset preview:")
    print(df.head())

else:

    print("\nERROR")
    print("-" * 60)
    print("model_features_2023_2026.csv could not be found.")
    print("Do not continue to model training until this file is located.")
