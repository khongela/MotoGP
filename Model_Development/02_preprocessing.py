from pathlib import Path
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# =========================================================
# Paths
# =========================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT
    / "data/processed/model_features_with_laps_2023_2026.csv"
)

SPLIT_PATH = (
    ROOT
    / "results/model_development/chronological_split_manifest.csv"
)

FEATURE_DIR = ROOT / "results/feature_analysis"

OUT_DIR = ROOT / "results/model_development"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_PATH = OUT_DIR / "preprocessing_summary.csv"


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

splits = pd.read_csv(
    SPLIT_PATH,
    low_memory=False
)


# =========================================================
# Load feature sets
# =========================================================

def load_feature_set(filename):

    path = FEATURE_DIR / filename

    features = (
        pd.read_csv(path)["feature"]
        .tolist()
    )

    return features


FEATURE_SETS = {

    "selected_15":
        load_feature_set(
            "feature_set_selected_15.csv"
        ),

    "full_23":
        load_feature_set(
            "feature_set_full_23.csv"
        ),

    "enhanced_54":
        load_feature_set(
            "feature_set_enhanced_54.csv"
        ),
}


assert len(FEATURE_SETS["selected_15"]) == 15
assert len(FEATURE_SETS["full_23"]) == 23
assert len(FEATURE_SETS["enhanced_54"]) == 54


# =========================================================
# Preprocessing
# =========================================================

def build_preprocessor(
    frame,
    features,
    scale_numeric=False,
):

    categorical_features = [
        feature
        for feature in features
        if (
            pd.api.types.is_object_dtype(
                frame[feature]
            )
            or
            pd.api.types.is_string_dtype(
                frame[feature]
            )
        )
    ]

    numeric_features = [
        feature
        for feature in features
        if feature not in categorical_features
    ]


    # -----------------------------------------------------
    # Numeric pipeline
    # -----------------------------------------------------

    numeric_steps = [
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        )
    ]

    if scale_numeric:

        numeric_steps.append(
            (
                "scaler",
                StandardScaler()
            )
        )

    numeric_pipeline = Pipeline(
        steps=numeric_steps
    )


    # -----------------------------------------------------
    # Categorical pipeline
    # -----------------------------------------------------

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                )
            ),
        ]
    )


    # -----------------------------------------------------
    # Combined preprocessor
    # -----------------------------------------------------

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    return (
        preprocessor,
        numeric_features,
        categorical_features,
    )


# =========================================================
# Fold definitions
# =========================================================

FOLDS = [
    "fold_1",
    "fold_2",
    "fold_3",
]


summary_rows = []


# =========================================================
# Validate each feature set
# =========================================================

for dataset_name, features in FEATURE_SETS.items():

    print("\n" + "=" * 70)
    print(dataset_name.upper())
    print("=" * 70)

    print(
        f"Raw predictors: {len(features)}"
    )


    for fold_name in FOLDS:

        train_event_ids = set(
            splits.loc[
                splits[fold_name] == "train",
                "event_id"
            ]
        )

        validation_event_ids = set(
            splits.loc[
                splits[fold_name] == "validation",
                "event_id"
            ]
        )


        train = df[
            df["event_id"].isin(
                train_event_ids
            )
        ].copy()

        validation = df[
            df["event_id"].isin(
                validation_event_ids
            )
        ].copy()


        # =================================================
        # Build preprocessing using training data only
        # =================================================

        (
            scaled_preprocessor,
            numeric_features,
            categorical_features,
        ) = build_preprocessor(
            train,
            features,
            scale_numeric=True,
        )


        # =================================================
        # Fit only on training rows
        # =================================================

        X_train = train[features]
        X_validation = validation[features]

        X_train_processed = (
            scaled_preprocessor
            .fit_transform(X_train)
        )

        X_validation_processed = (
            scaled_preprocessor
            .transform(X_validation)
        )


        # =================================================
        # Validation
        # =================================================

        assert (
            X_train_processed.shape[0]
            ==
            len(train)
        )

        assert (
            X_validation_processed.shape[0]
            ==
            len(validation)
        )

        assert (
            X_train_processed.shape[1]
            ==
            X_validation_processed.shape[1]
        )


        transformed_features = (
            scaled_preprocessor
            .get_feature_names_out()
        )


        print(
            f"\n{fold_name}"
        )

        print(
            f"Train rows:          "
            f"{len(train):,}"
        )

        print(
            f"Validation rows:     "
            f"{len(validation):,}"
        )

        print(
            f"Numeric predictors:  "
            f"{len(numeric_features)}"
        )

        print(
            f"Categorical:         "
            f"{len(categorical_features)}"
        )

        print(
            f"Encoded columns:     "
            f"{len(transformed_features)}"
        )


        summary_rows.append({

            "dataset":
                dataset_name,

            "fold":
                fold_name,

            "raw_predictors":
                len(features),

            "numeric_predictors":
                len(numeric_features),

            "categorical_predictors":
                len(categorical_features),

            "encoded_predictors":
                len(transformed_features),

            "train_rows":
                len(train),

            "validation_rows":
                len(validation),
        })


# =========================================================
# 2025 transformation check
# =========================================================

print("\n" + "=" * 70)
print("2025 TEMPORAL TRANSFORMATION CHECK")
print("=" * 70)


development_ids = set(
    splits.loc[
        splits["dataset_role"]
        == "development",
        "event_id"
    ]
)

test_ids = set(
    splits.loc[
        splits["dataset_role"]
        == "temporal_test",
        "event_id"
    ]
)


development = df[
    df["event_id"].isin(
        development_ids
    )
].copy()

test_2025 = df[
    df["event_id"].isin(
        test_ids
    )
].copy()


assert len(development) == 867
assert len(test_2025) == 475


for dataset_name, features in FEATURE_SETS.items():

    (
        preprocessor,
        numeric_features,
        categorical_features,
    ) = build_preprocessor(
        development,
        features,
        scale_numeric=True,
    )

    X_dev = (
        preprocessor.fit_transform(
            development[features]
        )
    )

    X_test = (
        preprocessor.transform(
            test_2025[features]
        )
    )

    assert (
        X_dev.shape[1]
        ==
        X_test.shape[1]
    )

    print(
        f"{dataset_name:15s} "
        f"development={X_dev.shape} "
        f"2025={X_test.shape}"
    )


# =========================================================
# Save summary
# =========================================================

summary = pd.DataFrame(
    summary_rows
)

summary.to_csv(
    SUMMARY_PATH,
    index=False
)


print("\nPREPROCESSING RULES")
print("=" * 70)

print(
    "Numeric missing values: "
    "training-fold median"
)

print(
    "Categorical missing values: "
    "training-fold most frequent value"
)

print(
    "Categorical encoding: "
    "one-hot encoding"
)

print(
    "Unknown future categories: "
    "ignored safely"
)

print(
    "Numeric scaling: "
    "fitted on training rows only"
)

print(
    "Validation and 2025 data never fit "
    "the preprocessing transformations."
)


print("\nSAVED")
print("=" * 70)

print(SUMMARY_PATH)


print("\nPREPROCESSING VALIDATION COMPLETE")