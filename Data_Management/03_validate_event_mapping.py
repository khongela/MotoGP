
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

# ---------------------------------------------------------
# Circuit code mapping
# ---------------------------------------------------------

CIRCUIT_CODES = {
    "Autódromo Internacional do Algarve": "POR",
    "Termas de Río Hondo": "ARG",
    "Circuit Of The Americas": "AME",
    "Circuito de Jerez - Ángel Nieto": "SPA",
    "Le Mans": "FRA",
    "Autodromo Internazionale del Mugello": "ITA",
    "Sachsenring": "GER",
    "TT Circuit Assen": "NED",
    "Silverstone Circuit": "GBR",
    "Red Bull Ring - Spielberg": "AUT",
    "Circuit de Barcelona-Catalunya": "CAT",
    "Misano World Circuit Marco Simoncelli": "RSM",
    "Buddh International Circuit": "IND",
    "Mobility Resort Motegi": "JPN",
    "Pertamina Mandalika Circuit": "INA",
    "Phillip Island": "AUS",
    "Chang International Circuit": "THA",
    "Petronas Sepang International Circuit": "MAL",
    "Lusail International Circuit": "QAT",
    "Circuit Ricardo Tormo": "VAL",
    "MotorLand Aragón": "ARA",
    "CREDITAS Autodrom Brno": "CZE",
    "Balaton Park Circuit": "HUN",
    "Autódromo Internacional de Goiânia - Ayrton Senna": "BRA",
}

# ---------------------------------------------------------
# Extract race events
# ---------------------------------------------------------

events = races[
    [
        "event_id",
        "season",
        "round",
        "event_date",
        "physical_circuit"
    ]
].drop_duplicates().copy()

events["circuit_code"] = (
    events["physical_circuit"].map(CIRCUIT_CODES)
)

# ---------------------------------------------------------
# Resolve special Grand Prix identifiers
# ---------------------------------------------------------

# Some circuits hosted two Grands Prix in the same
# season, each with a different timing-sheet code.

SPECIAL_EVENTS = {
    (2024, 14): "EMI",  # Emilia-Romagna GP
    (2024, 20): "SLD",  # Solidarity GP
    (2026, 3): "USA",   # Americas GP
}

for (season, round_number), code in SPECIAL_EVENTS.items():

    mask = (
        (events["season"] == season) &
        (events["round"] == round_number)
    )

    assert mask.sum() == 1, (
        f"Expected exactly one event for "
        f"{season} round {round_number}"
    )

    events.loc[mask, "circuit_code"] = code

# ---------------------------------------------------------
# Count race events
# ---------------------------------------------------------

event_counts = (
    events.groupby(
        ["season", "circuit_code"],
        dropna=False
    )
    .agg(
        event_count=("event_id", "nunique"),
        rounds=("round", lambda x: list(x)),
        dates=("event_date", lambda x: list(x))
    )
    .reset_index()
)

# ---------------------------------------------------------
# Count lap records
# ---------------------------------------------------------

lap_counts = (
    laps.groupby(["year", "circuit"])
    .size()
    .reset_index(name="lap_count")
)

# ---------------------------------------------------------
# Compare event identifiers
# ---------------------------------------------------------

audit = event_counts.merge(
    lap_counts,
    left_on=["season", "circuit_code"],
    right_on=["year", "circuit"],
    how="outer",
    indicator=True
)

print("\nEVENT MAPPING AUDIT")
print("-" * 65)
print(audit.to_string(index=False))

# ---------------------------------------------------------
# Identify mapping problems
# ---------------------------------------------------------

print("\nAMBIGUOUS OR UNMATCHED EVENTS")
print("-" * 65)

problems = audit[
    (audit["event_count"] != 1) |
    (audit["_merge"] != "both")
]

if problems.empty:
    print("No mapping issues detected.")
else:
    print(problems.to_string(index=False))

# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

print("\nMAPPING SUMMARY")
print("-" * 65)

print(f"Race events: {events['event_id'].nunique()}")
print(f"Lap event groups: {len(lap_counts)}")
print(f"Matched event groups: {(audit['_merge'] == 'both').sum()}")
print(f"Mapping problems: {len(problems)}")

# ---------------------------------------------------------
# Save results
# ---------------------------------------------------------

out = ROOT / "results" / "feature_analysis"
out.mkdir(parents=True, exist_ok=True)

audit.to_csv(
    out / "event_mapping_audit.csv",
    index=False
)

events.to_csv(
    out / "event_mapping.csv",
    index=False
)

print("\nFiles saved:")
print(out / "event_mapping_audit.csv")
print(out / "event_mapping.csv")

if not problems.empty:
    raise SystemExit(
        "Event mapping validation failed. "
        "Resolve the reported issues before proceeding."
    )

print("\nEVENT MAPPING VALIDATION PASSED")

