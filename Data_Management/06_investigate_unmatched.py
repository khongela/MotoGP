
from pathlib import Path
import unicodedata
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/feature_analysis"

races = pd.read_csv(
    ROOT / "data/processed/model_features_2023_2026.csv"
)

unmatched = pd.read_csv(
    OUT / "rider_resolution_audit.csv"
)

unmatched = unmatched[
    unmatched["match_status"] == "unresolved"
].copy()


def normalize(value):
    if pd.isna(value):
        return ""

    value = unicodedata.normalize("NFKD", str(value))
    value = "".join(
        c for c in value
        if not unicodedata.combining(c)
    )
    return re.sub(r"[^a-z0-9]", "", value.lower())


races["name_key"] = races["rider"].map(normalize)
unmatched["name_key"] = unmatched["rider"].map(normalize)

# Find rider names in other race entries
global_riders = (
    races[["rider_id", "rider", "name_key"]]
    .drop_duplicates()
)

investigation = unmatched.merge(
    global_riders.rename(columns={
        "rider_id": "possible_rider_id",
        "rider": "possible_race_name"
    }),
    on="name_key",
    how="left"
)

# Identify whether a candidate appears in that event
event_entries = set(
    zip(races["event_id"], races["rider_id"].astype(str))
)

investigation["candidate_in_event"] = investigation.apply(
    lambda row: (
        (row["event_id"], str(row["possible_rider_id"]))
        in event_entries
    ) if pd.notna(row["possible_rider_id"]) else False,
    axis=1
)

columns = [
    "year",
    "circuit",
    "rider",
    "rider_number",
    "possible_race_name",
    "possible_rider_id",
    "candidate_in_event"
]

print("\nUNMATCHED RIDER INVESTIGATION")
print("-" * 75)
print(investigation[columns].to_string(index=False))

investigation.to_csv(
    OUT / "unmatched_rider_investigation.csv",
    index=False
)

print("\nSaved unmatched_rider_investigation.csv")

