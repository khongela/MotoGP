
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "feature_analysis"

audit = pd.read_csv(OUT / "rider_mapping_audit.csv")

# Already matched identities
matched = audit[audit["rider_id"].notna()].copy()
unmatched = audit[audit["rider_id"].isna()].copy()

# Establish verified rider-number mappings from existing
# successful matches within the same event.
number_map = (
    matched.groupby(["event_id", "rider_number"])["rider_id"]
    .agg(["nunique", "first"])
    .reset_index()
)

number_map = number_map[
    number_map["nunique"] == 1
][["event_id", "rider_number", "first"]]

number_map = number_map.rename(
    columns={"first": "resolved_rider_id"}
)

# Resolve only numbers already established
# by successful matches within that event.
resolved = unmatched.merge(
    number_map,
    on=["event_id", "rider_number"],
    how="left",
    validate="many_to_one"
)

resolved["match_status"] = resolved[
    "resolved_rider_id"
].apply(
    lambda x: "resolved_by_number"
    if pd.notna(x)
    else "unresolved"
)

print("\nRIDER RESOLUTION SUMMARY")
print("-" * 65)

print(f"Previously unmatched: {len(unmatched)}")
print(
    "Resolved using verified rider numbers:",
    (resolved["match_status"] == "resolved_by_number").sum()
)
print(
    "Still unresolved:",
    (resolved["match_status"] == "unresolved").sum()
)

print("\nUNRESOLVED RIDERS")
print("-" * 65)

remaining = resolved[
    resolved["match_status"] == "unresolved"
]

print(
    remaining[
        ["year", "circuit", "rider", "rider_number"]
    ].to_string(index=False)
)

resolved.to_csv(
    OUT / "rider_resolution_audit.csv",
    index=False
)

print("\nSaved rider_resolution_audit.csv")

