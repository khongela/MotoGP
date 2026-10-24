from pathlib import Path
import pandas as pd


# =========================================================
# Paths
# =========================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT
    / "data/processed/model_features_with_laps_2023_2026.csv"
)

OUT_DIR = ROOT / "results/model_development"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PATH = OUT_DIR / "chronological_split_manifest.csv"


# =========================================================
# Load data
# =========================================================

df = pd.read_csv(
    DATA_PATH,
    low_memory=False
)

df["event_date"] = pd.to_datetime(
    df["event_date"]
)


# =========================================================
# Event-level table
# =========================================================

events = (
    df[
        [
            "event_id",
            "event_date",
            "season",
            "round",
        ]
    ]
    .drop_duplicates()
    .sort_values(
        [
            "event_date",
            "season",
            "round",
        ]
    )
    .reset_index(drop=True)
)


# =========================================================
# Dataset roles
# =========================================================

events["dataset_role"] = "unused"

events.loc[
    events["season"].isin([2023, 2024]),
    "dataset_role"
] = "development"

events.loc[
    events["season"] == 2025,
    "dataset_role"
] = "temporal_test"

events.loc[
    events["season"] == 2026,
    "dataset_role"
] = "supplementary"


development = (
    events[
        events["dataset_role"] == "development"
    ]
    .reset_index(drop=True)
)

test_2025 = (
    events[
        events["dataset_role"] == "temporal_test"
    ]
    .reset_index(drop=True)
)

supplementary_2026 = (
    events[
        events["dataset_role"] == "supplementary"
    ]
    .reset_index(drop=True)
)


# =========================================================
# Validate season structure
# =========================================================

assert len(development) == 40, (
    f"Expected 40 development races, "
    f"found {len(development)}"
)

assert len(test_2025) == 22, (
    f"Expected 22 races in 2025, "
    f"found {len(test_2025)}"
)

assert len(supplementary_2026) == 12, (
    f"Expected 12 races in 2026, "
    f"found {len(supplementary_2026)}"
)


# =========================================================
# Expanding chronological folds
#
# Fold 1:
#   Train = races 1-28
#   Validation = races 29-32
#
# Fold 2:
#   Train = races 1-32
#   Validation = races 33-36
#
# Fold 3:
#   Train = races 1-36
#   Validation = races 37-40
# =========================================================

FOLDS = [
    {
        "name": "fold_1",
        "train_end": 28,
        "validation_end": 32,
    },
    {
        "name": "fold_2",
        "train_end": 32,
        "validation_end": 36,
    },
    {
        "name": "fold_3",
        "train_end": 36,
        "validation_end": 40,
    },
]

development_ids = development[
    "event_id"
].tolist()


for fold in FOLDS:

    fold_name = fold["name"]
    train_end = fold["train_end"]
    validation_end = fold["validation_end"]

    events[fold_name] = "not_used"

    train_ids = development_ids[
        :train_end
    ]

    validation_ids = development_ids[
        train_end:validation_end
    ]

    future_ids = development_ids[
        validation_end:
    ]

    events.loc[
        events["event_id"].isin(train_ids),
        fold_name
    ] = "train"

    events.loc[
        events["event_id"].isin(validation_ids),
        fold_name
    ] = "validation"

    events.loc[
        events["event_id"].isin(future_ids),
        fold_name
    ] = "future"


# =========================================================
# Season summary
# =========================================================

season_summary = (
    df.groupby("season")
    .agg(
        races=("event_id", "nunique"),
        rider_entries=("event_id", "size"),
        podiums=("podium_finish", "sum"),
    )
    .reset_index()
)


print("\nSEASON SPLIT")
print("=" * 70)

print(
    season_summary.to_string(
        index=False
    )
)


print("\nDATASET ROLES")
print("=" * 70)

print(
    f"Development:   "
    f"{len(development)} races "
    f"(2023-2024)"
)

print(
    f"Temporal test: "
    f"{len(test_2025)} races "
    f"(2025)"
)

print(
    f"Supplementary: "
    f"{len(supplementary_2026)} races "
    f"(2026)"
)


# =========================================================
# Validate each fold
# =========================================================

for fold in FOLDS:

    fold_name = fold["name"]

    train_events = events[
        events[fold_name] == "train"
    ]

    validation_events = events[
        events[fold_name] == "validation"
    ]

    train_ids = set(
        train_events["event_id"]
    )

    validation_ids = set(
        validation_events["event_id"]
    )

    train_rows = df[
        df["event_id"].isin(train_ids)
    ]

    validation_rows = df[
        df["event_id"].isin(validation_ids)
    ]

    print(f"\n{fold_name.upper()}")
    print("=" * 70)

    print(
        f"Training races:      "
        f"{len(train_events)}"
    )

    print(
        f"Training rows:       "
        f"{len(train_rows):,}"
    )

    print(
        f"Validation races:    "
        f"{len(validation_events)}"
    )

    print(
        f"Validation rows:     "
        f"{len(validation_rows):,}"
    )

    print(
        f"Training period:     "
        f"{train_events['event_date'].min().date()} "
        f"to "
        f"{train_events['event_date'].max().date()}"
    )

    print(
        f"Validation period:   "
        f"{validation_events['event_date'].min().date()} "
        f"to "
        f"{validation_events['event_date'].max().date()}"
    )

    assert (
        train_events["event_date"].max()
        <
        validation_events["event_date"].min()
    ), (
        f"{fold_name} is not chronological."
    )

    assert train_ids.isdisjoint(
        validation_ids
    ), (
        f"{fold_name} contains overlapping races."
    )


# =========================================================
# 2025 temporal test
# =========================================================

test_ids = set(
    test_2025["event_id"]
)

test_rows = df[
    df["event_id"].isin(test_ids)
]

print("\n2025 TEMPORAL TEST")
print("=" * 70)

print(
    f"Races:          "
    f"{len(test_2025)}"
)

print(
    f"Rider entries:  "
    f"{len(test_rows):,}"
)

print(
    f"Podiums:        "
    f"{int(test_rows['podium_finish'].sum())}"
)

assert len(test_rows) == 475
assert test_rows["podium_finish"].sum() == 66


# =========================================================
# 2026 supplementary set
# =========================================================

supplementary_ids = set(
    supplementary_2026["event_id"]
)

supplementary_rows = df[
    df["event_id"].isin(supplementary_ids)
]

print("\n2026 SUPPLEMENTARY SET")
print("=" * 70)

print(
    f"Races:          "
    f"{len(supplementary_2026)}"
)

print(
    f"Rider entries:  "
    f"{len(supplementary_rows):,}"
)

print(
    f"Podiums:        "
    f"{int(supplementary_rows['podium_finish'].sum())}"
)


# =========================================================
# Race grouping validation
# =========================================================

assert df["event_id"].nunique() == 74

print("\nRACE GROUPING")
print("=" * 70)

print(
    "All rider entries from each race remain "
    "inside the same chronological partition."
)


# =========================================================
# Save
# =========================================================

events.to_csv(
    OUT_PATH,
    index=False
)

print("\nSAVED")
print("=" * 70)

print(OUT_PATH)

print("\nCHRONOLOGICAL SPLITS COMPLETE")