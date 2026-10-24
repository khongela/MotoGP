from pathlib import Path
import numpy as np
import pandas as pd


# =========================================================
# Paths
# =========================================================

ROOT = Path(__file__).resolve().parents[1]

LAPS_PATH = ROOT / "data/processed/laps.csv"
MODEL_PATH = ROOT / "data/processed/model_features_2023_2026.csv"
MASTER_PATH = ROOT / "data/interim/master.csv"
EVENT_MAP_PATH = ROOT / "results/feature_analysis/event_mapping.csv"

OUT_DIR = ROOT / "data/processed"
AUDIT_DIR = ROOT / "results/feature_analysis"

OUT_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# Session definitions
# =========================================================

# Everything known before the MotoGP main race.
PRE_RACE_SESSIONS = [
    "FP",
    "FP1",
    "FP2",
    "P1",
    "P2",
    "PR",
    "Q1",
    "Q2",
    "SPR",
    "WUP",
]

# Sessions used to estimate practice / long-run pace.
PRACTICE_SESSIONS = [
    "FP",
    "FP1",
    "FP2",
    "P1",
    "P2",
    "PR",
    "WUP",
]

QUALIFYING_SESSIONS = [
    "Q1",
    "Q2",
]

SPRINT_SESSIONS = [
    "SPR",
]

# Never use these for the current target race.
# They may only contribute to future races through lagged features.
RACE_SESSIONS = [
    "RAC",
    "RAC2",
]


# =========================================================
# Load data
# =========================================================

laps = pd.read_csv(
    LAPS_PATH,
    low_memory=False
)

model = pd.read_csv(
    MODEL_PATH,
    low_memory=False
)

master = pd.read_csv(
    MASTER_PATH,
    low_memory=False
)

events = pd.read_csv(
    EVENT_MAP_PATH,
    low_memory=False
)

model["event_date"] = pd.to_datetime(model["event_date"])
master["event_date"] = pd.to_datetime(master["event_date"])


# =========================================================
# Race starters
# =========================================================

# The master table contains four NOTSTARTED rows.
# They are intentionally excluded from the predictive dataset.

starters = master[
    master["status"] != "NOTSTARTED"
][
    [
        "event_id",
        "event_date",
        "season",
        "round",
        "rider_id",
        "rider_number",
        "rider",
        "team_name",
    ]
].drop_duplicates(
    subset=["event_id", "rider_id"]
)

assert len(starters) == 1602, (
    f"Expected 1,602 race starters, found {len(starters)}"
)

assert not starters.duplicated(
    ["event_id", "rider_number"]
).any(), (
    "Duplicate rider numbers found within an event."
)


# =========================================================
# Map lap records to event IDs
# =========================================================

event_keys = events[
    [
        "event_id",
        "season",
        "circuit_code",
    ]
].copy()

laps = laps.merge(
    event_keys,
    left_on=["year", "circuit"],
    right_on=["season", "circuit_code"],
    how="left",
    validate="many_to_one",
)

if laps["event_id"].isna().any():
    raise ValueError(
        "Some lap records could not be mapped to an event."
    )


# =========================================================
# Map laps to rider IDs
# =========================================================

rider_keys = starters[
    [
        "event_id",
        "rider_number",
        "rider_id",
    ]
]

laps = laps.merge(
    rider_keys,
    on=["event_id", "rider_number"],
    how="left",
    validate="many_to_one",
)

total_laps = len(laps)
mapped_laps = laps["rider_id"].notna().sum()
unmapped_laps = laps["rider_id"].isna().sum()

print("\nLAP MAPPING")
print("=" * 70)
print(f"Total lap rows:    {total_laps:,}")
print(f"Mapped to starters:{mapped_laps:,}")
print(f"Unmapped:          {unmapped_laps:,}")

# Laps belonging to riders who did not start the GP are not
# useful for a rider-race target and are deliberately excluded.

laps = laps[
    laps["rider_id"].notna()
].copy()


# =========================================================
# Clean lap records
# =========================================================

valid = laps[
    (~laps["cancelled"]) &
    (~laps["in_pit"]) &
    (laps["lap_time_s"].notna())
].copy()

# Remove extremely slow laps that would distort pace summaries.
# This is done separately within each event/session.

session_median = (
    valid.groupby(
        ["event_id", "session"]
    )["lap_time_s"]
    .transform("median")
)

valid = valid[
    valid["lap_time_s"] < session_median * 1.10
].copy()

print("\nVALID LAP FILTER")
print("=" * 70)
print(f"Mapped laps:       {mapped_laps:,}")
print(f"Valid pace laps:   {len(valid):,}")


# =========================================================
# CURRENT-WEEKEND PRE-RACE FEATURES
# =========================================================

pre = valid[
    valid["session"].isin(PRE_RACE_SESSIONS)
].copy()

current_rows = []

for event_id, weekend in pre.groupby("event_id"):

    event_best = weekend["lap_time_s"].min()

    best_sector = {
        sector: weekend[sector].min()
        for sector in ["t1", "t2", "t3", "t4"]
    }

    practice_weekend = weekend[
        weekend["session"].isin(PRACTICE_SESSIONS)
    ]

    qualifying_weekend = weekend[
        weekend["session"].isin(QUALIFYING_SESSIONS)
    ]

    best_practice = (
        practice_weekend["lap_time_s"].min()
        if len(practice_weekend)
        else np.nan
    )

    best_quali = (
        qualifying_weekend["lap_time_s"].min()
        if len(qualifying_weekend)
        else np.nan
    )

    for rider_id, rider_laps in weekend.groupby("rider_id"):

        row = {
            "event_id": event_id,
            "rider_id": rider_id,
        }

        # -------------------------------------------------
        # Overall pre-race activity
        # -------------------------------------------------

        row["pre_race_laps"] = len(rider_laps)

        row["pre_race_sessions_run"] = (
            rider_laps["session"].nunique()
        )

        row["best_pre_race_lap_s"] = (
            rider_laps["lap_time_s"].min()
        )

        row["gap_to_best_pre_race_s"] = (
            row["best_pre_race_lap_s"] - event_best
        )

        # -------------------------------------------------
        # Practice pace
        # -------------------------------------------------

        practice = rider_laps[
            rider_laps["session"].isin(
                PRACTICE_SESSIONS
            )
        ]

        if len(practice):

            row["best_practice_lap_s"] = (
                practice["lap_time_s"].min()
            )

            row["gap_to_best_practice_s"] = (
                row["best_practice_lap_s"]
                - best_practice
            )

        else:

            row["best_practice_lap_s"] = np.nan
            row["gap_to_best_practice_s"] = np.nan

        # -------------------------------------------------
        # Qualifying
        # -------------------------------------------------

        qualifying = rider_laps[
            rider_laps["session"].isin(
                QUALIFYING_SESSIONS
            )
        ]

        if len(qualifying):

            row["best_quali_lap_s"] = (
                qualifying["lap_time_s"].min()
            )

            row["gap_to_best_quali_s"] = (
                row["best_quali_lap_s"]
                - best_quali
            )

        else:

            row["best_quali_lap_s"] = np.nan
            row["gap_to_best_quali_s"] = np.nan

        row["reached_q2"] = int(
            "Q2" in set(rider_laps["session"])
        )

        # -------------------------------------------------
        # Sector performance
        # -------------------------------------------------

        for sector in ["t1", "t2", "t3", "t4"]:

            rider_best = rider_laps[sector].min()
            event_sector_best = best_sector[sector]

            row[f"{sector}_gap_s"] = (
                rider_best - event_sector_best
                if pd.notna(rider_best)
                and pd.notna(event_sector_best)
                else np.nan
            )

        # -------------------------------------------------
        # Top speed
        # -------------------------------------------------

        row["pre_race_top_speed_max"] = (
            rider_laps["top_speed_kmh"].max()
        )

        # -------------------------------------------------
        # Long-run practice pace
        # -------------------------------------------------

        longest = pd.DataFrame()

        for _, stint in practice.groupby(
            ["session", "run"]
        ):

            if len(stint) > len(longest):
                longest = stint.copy()

        row["longrun_laps"] = len(longest)

        if len(longest) >= 5:

            median_time = longest[
                "lap_time_s"
            ].median()

            std_time = longest[
                "lap_time_s"
            ].std()

            row["longrun_median_s"] = median_time
            row["longrun_std_s"] = std_time

            slope = np.polyfit(
                longest["lap"],
                longest["lap_time_s"],
                1,
            )[0]

            row["longrun_deg_s_per_lap"] = slope

            row["longrun_front_tyre"] = (
                longest["front_tyre"].iloc[0]
            )

            row["longrun_rear_tyre"] = (
                longest["rear_tyre"].iloc[0]
            )

        else:

            row["longrun_median_s"] = np.nan
            row["longrun_std_s"] = np.nan
            row["longrun_deg_s_per_lap"] = np.nan
            row["longrun_front_tyre"] = None
            row["longrun_rear_tyre"] = None

        current_rows.append(row)


current = pd.DataFrame(current_rows)


# =========================================================
# Current-weekend relative rankings
# =========================================================

current["best_pre_race_rank"] = (
    current.groupby("event_id")[
        "best_pre_race_lap_s"
    ]
    .rank(method="min")
)

current["best_quali_rank"] = (
    current.groupby("event_id")[
        "best_quali_lap_s"
    ]
    .rank(method="min")
)

current["longrun_rank"] = (
    current.groupby("event_id")[
        "longrun_median_s"
    ]
    .rank(method="min")
)

current["pre_race_top_speed_rank"] = (
    current.groupby("event_id")[
        "pre_race_top_speed_max"
    ]
    .rank(
        method="min",
        ascending=False
    )
)


# =========================================================
# Teammate-relative long-run performance
# =========================================================

team_lookup = starters[
    [
        "event_id",
        "rider_id",
        "team_name",
    ]
]

current = current.merge(
    team_lookup,
    on=["event_id", "rider_id"],
    how="left",
    validate="one_to_one",
)

team_best = (
    current.groupby(
        ["event_id", "team_name"]
    )["longrun_median_s"]
    .transform("min")
)

current[
    "longrun_gap_to_teammate_s"
] = (
    current["longrun_median_s"]
    - team_best
)


# =========================================================
# HISTORICAL MAIN-RACE LAP PERFORMANCE
# =========================================================

# Main-race laps are allowed here because they are shifted
# and only used to predict FUTURE races.

race_laps = valid[
    valid["session"].isin(RACE_SESSIONS)
].copy()

race_rows = []

for (event_id, rider_id), group in race_laps.groupby(
    ["event_id", "rider_id"]
):

    median_time = group["lap_time_s"].median()
    std_time = group["lap_time_s"].std()

    race_rows.append({
        "event_id": event_id,
        "rider_id": rider_id,

        "race_laps_completed":
            len(group),

        "race_best_lap_s":
            group["lap_time_s"].min(),

        "race_median_lap_s":
            median_time,

        "race_lap_std_s":
            std_time,

        "race_top_speed_max":
            group["top_speed_kmh"].max(),
    })


race_perf = pd.DataFrame(race_rows)


# =========================================================
# Relative race-performance measures
# =========================================================

race_perf[
    "race_pace_rank"
] = (
    race_perf.groupby("event_id")[
        "race_median_lap_s"
    ]
    .rank(method="min")
)

race_perf[
    "race_top_speed_rank"
] = (
    race_perf.groupby("event_id")[
        "race_top_speed_max"
    ]
    .rank(
        method="min",
        ascending=False
    )
)


# =========================================================
# Put race performance onto every starter row
# =========================================================

history_base = starters[
    [
        "event_id",
        "rider_id",
        "event_date",
        "season",
        "round",
    ]
].copy()

history_base = history_base.merge(
    race_perf,
    on=["event_id", "rider_id"],
    how="left",
    validate="one_to_one",
)

history_base = history_base.sort_values(
    ["rider_id", "event_date"]
).reset_index(drop=True)


# =========================================================
# Lagged + rolling historical features
# =========================================================

HISTORICAL_COLUMNS = {

    "race_pace_rank":
        "race_pace_rank",

    "race_top_speed_rank":
        "race_top_speed_rank",

    "race_laps_completed":
        "race_laps_completed",
}


for source_col, output_name in HISTORICAL_COLUMNS.items():

    history_base[
        f"previous_{output_name}"
    ] = (
        history_base.groupby(
            "rider_id"
        )[source_col]
        .shift(1)
    )

    history_base[
        f"last3_{output_name}_mean"
    ] = (
        history_base.groupby(
            "rider_id"
        )[source_col]
        .transform(
            lambda x:
            x.shift(1)
             .rolling(
                 window=3,
                 min_periods=1
             )
             .mean()
        )
    )


historical_feature_columns = [
    column
    for column in history_base.columns
    if column.startswith("previous_")
    or column.startswith("last3_")
]

historical = history_base[
    [
        "event_id",
        "rider_id",
    ]
    + historical_feature_columns
]


# =========================================================
# Final integration
# =========================================================

final = model.merge(
    current.drop(
        columns=["team_name"],
        errors="ignore"
    ),
    on=["event_id", "rider_id"],
    how="left",
    validate="one_to_one",
)

final = final.merge(
    historical,
    on=["event_id", "rider_id"],
    how="left",
    validate="one_to_one",
)


# =========================================================
# Validation
# =========================================================

assert len(final) == len(model), (
    "Final row count changed during lap integration."
)

assert final[
    ["event_id", "rider_id"]
].duplicated().sum() == 0

assert final["podium_finish"].equals(
    model["podium_finish"]
)

print("\nFINAL DATASET")
print("=" * 70)

print(f"Rows:                {len(final):,}")
print(f"Original columns:    {len(model.columns)}")
print(f"Final columns:       {len(final.columns)}")
print(
    f"New lap features:    "
    f"{len(final.columns) - len(model.columns)}"
)

print("\nCURRENT PRE-RACE COVERAGE")
print("=" * 70)

coverage_columns = [
    "best_practice_lap_s",
    "best_quali_lap_s",
    "longrun_median_s",
    "pre_race_top_speed_max",
]

for column in coverage_columns:

    available = final[column].notna().sum()

    print(
        f"{column:30s} "
        f"{available:4d}/{len(final)} "
        f"({available / len(final) * 100:.1f}%)"
    )


print("\nHISTORICAL FEATURE COVERAGE")
print("=" * 70)

for column in historical_feature_columns:

    available = final[column].notna().sum()

    print(
        f"{column:35s} "
        f"{available:4d}/{len(final)} "
        f"({available / len(final) * 100:.1f}%)"
    )


# =========================================================
# Explicit leakage audit
# =========================================================

FORBIDDEN_CURRENT_RACE_COLUMNS = [
    "finish_position",
    "status",
    "race_points",
    "round_points",
]

print("\nLEAKAGE RULE")
print("=" * 70)

print(
    "Current RAC/RAC2 laps are used ONLY to construct "
    "shifted historical features."
)

print(
    "Current-weekend lap features use only:",
    PRE_RACE_SESSIONS
)

print(
    "Forbidden target-race outcomes remain outside "
    "the candidate predictor set:",
    FORBIDDEN_CURRENT_RACE_COLUMNS
)


# =========================================================
# Save outputs
# =========================================================

CURRENT_PATH = (
    AUDIT_DIR /
    "current_pre_race_lap_features.csv"
)

HISTORICAL_PATH = (
    AUDIT_DIR /
    "historical_lap_features.csv"
)

FINAL_PATH = (
    OUT_DIR /
    "model_features_with_laps_2023_2026.csv"
)

current.to_csv(
    CURRENT_PATH,
    index=False
)

historical.to_csv(
    HISTORICAL_PATH,
    index=False
)

final.to_csv(
    FINAL_PATH,
    index=False
)

print("\nFILES SAVED")
print("=" * 70)
print(CURRENT_PATH)
print(HISTORICAL_PATH)
print(FINAL_PATH)

print("\nLAP FEATURE ENGINEERING COMPLETE")
