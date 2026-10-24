from pathlib import Path
from itertools import product

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)

from xgboost import XGBClassifier


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

OUT_DIR = (
    ROOT
    / "results/model_development/xgboost"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


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
# Feature sets
# =========================================================

def load_features(filename):

    return (
        pd.read_csv(
            FEATURE_DIR / filename
        )["feature"]
        .tolist()
    )


FEATURE_SETS = {

    "selected_15":
        load_features(
            "feature_set_selected_15.csv"
        ),

    "full_23":
        load_features(
            "feature_set_full_23.csv"
        ),

    "enhanced_54":
        load_features(
            "feature_set_enhanced_54.csv"
        ),
}


# =========================================================
# XGBoost search space
# =========================================================

N_ESTIMATORS = [
    200,
    350,
]

LEARNING_RATES = [
    0.02,
    0.05,
]

MAX_DEPTHS = [
    2,
    3,
]

REG_ALPHA = [
    0.0,
    0.2,
]

REG_LAMBDA = [
    8.0,
]

SUBSAMPLE = 0.85
COLSAMPLE_BYTREE = 1.0
MIN_CHILD_WEIGHT = 1


CONFIGURATIONS = list(
    product(
        N_ESTIMATORS,
        LEARNING_RATES,
        MAX_DEPTHS,
        REG_ALPHA,
        REG_LAMBDA,
    )
)


FOLDS = [
    "fold_1",
    "fold_2",
    "fold_3",
]


# =========================================================
# Preprocessing
# =========================================================

def build_preprocessor(
    train,
    features
):

    categorical_features = [
        feature
        for feature in features
        if (
            pd.api.types.is_object_dtype(
                train[feature]
            )
            or
            pd.api.types.is_string_dtype(
                train[feature]
            )
        )
    ]

    numeric_features = [
        feature
        for feature in features
        if feature not in categorical_features
    ]


    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
        ]
    )


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


    return ColumnTransformer(
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
        ]
    )


# =========================================================
# Race-level top-three evaluation
# =========================================================

def top_three_metrics(
    validation,
    probabilities
):

    result = validation[
        [
            "event_id",
            "podium_finish",
        ]
    ].copy()

    result["probability"] = probabilities
    result["predicted_podium"] = 0


    for event_id, group in result.groupby(
        "event_id"
    ):

        top_three = (
            group
            .nlargest(
                3,
                "probability"
            )
            .index
        )

        result.loc[
            top_three,
            "predicted_podium"
        ] = 1


    precision = precision_score(
        result["podium_finish"],
        result["predicted_podium"],
        zero_division=0,
    )

    recall = recall_score(
        result["podium_finish"],
        result["predicted_podium"],
        zero_division=0,
    )

    f1 = f1_score(
        result["podium_finish"],
        result["predicted_podium"],
        zero_division=0,
    )


    podium_hits = (
        result[
            result["predicted_podium"] == 1
        ]
        .groupby("event_id")[
            "podium_finish"
        ]
        .sum()
        .mean()
    )


    return (
        precision,
        recall,
        f1,
        podium_hits,
        result,
    )


# =========================================================
# Cross-validation
# =========================================================

score_rows = []
prediction_rows = []


for dataset_name, features in FEATURE_SETS.items():

    print("\n" + "=" * 70)
    print(dataset_name.upper())
    print("=" * 70)


    for (
        n_estimators,
        learning_rate,
        max_depth,
        reg_alpha,
        reg_lambda,
    ) in CONFIGURATIONS:

        print(
            "\n"
            f"trees={n_estimators}, "
            f"lr={learning_rate}, "
            f"depth={max_depth}, "
            f"alpha={reg_alpha}, "
            f"lambda={reg_lambda}"
        )


        for fold_name in FOLDS:

            # ---------------------------------------------
            # Event IDs
            # ---------------------------------------------

            train_ids = set(
                splits.loc[
                    splits[fold_name]
                    == "train",
                    "event_id"
                ]
            )

            validation_ids = set(
                splits.loc[
                    splits[fold_name]
                    == "validation",
                    "event_id"
                ]
            )


            # ---------------------------------------------
            # Rows
            # ---------------------------------------------

            train = df[
                df["event_id"].isin(
                    train_ids
                )
            ].copy()

            validation = df[
                df["event_id"].isin(
                    validation_ids
                )
            ].copy()


            X_train = train[
                features
            ]

            y_train = train[
                "podium_finish"
            ]

            X_validation = validation[
                features
            ]

            y_validation = validation[
                "podium_finish"
            ]


            # ---------------------------------------------
            # Class imbalance
            # ---------------------------------------------

            negative_count = (
                y_train == 0
            ).sum()

            positive_count = (
                y_train == 1
            ).sum()

            scale_pos_weight = (
                negative_count
                / positive_count
            )


            # ---------------------------------------------
            # Pipeline
            # ---------------------------------------------

            preprocessor = (
                build_preprocessor(
                    train,
                    features
                )
            )


            model = XGBClassifier(

                n_estimators=n_estimators,

                learning_rate=learning_rate,

                max_depth=max_depth,

                min_child_weight=
                    MIN_CHILD_WEIGHT,

                subsample=
                    SUBSAMPLE,

                colsample_bytree=
                    COLSAMPLE_BYTREE,

                reg_alpha=
                    reg_alpha,

                reg_lambda=
                    reg_lambda,

                scale_pos_weight=
                    scale_pos_weight,

                objective=
                    "binary:logistic",

                eval_metric=
                    "logloss",

                random_state=42,

                n_jobs=-1,
            )


            pipeline = Pipeline(
                steps=[
                    (
                        "preprocessing",
                        preprocessor
                    ),
                    (
                        "model",
                        model
                    ),
                ]
            )


            # ---------------------------------------------
            # Train
            # ---------------------------------------------

            pipeline.fit(
                X_train,
                y_train
            )


            # ---------------------------------------------
            # Predict
            # ---------------------------------------------

            probabilities = (
                pipeline.predict_proba(
                    X_validation
                )[:, 1]
            )


            # ---------------------------------------------
            # Metrics
            # ---------------------------------------------

            average_precision = (
                average_precision_score(
                    y_validation,
                    probabilities
                )
            )

            roc_auc = roc_auc_score(
                y_validation,
                probabilities
            )


            (
                precision,
                recall,
                f1,
                podium_hits,
                race_predictions,
            ) = top_three_metrics(
                validation,
                probabilities
            )


            print(
                f"{fold_name:8s} "
                f"AP={average_precision:.3f} "
                f"ROC-AUC={roc_auc:.3f} "
                f"hits={podium_hits:.2f}"
            )


            # ---------------------------------------------
            # Save score
            # ---------------------------------------------

            score_rows.append({

                "dataset":
                    dataset_name,

                "n_estimators":
                    n_estimators,

                "learning_rate":
                    learning_rate,

                "max_depth":
                    max_depth,

                "reg_alpha":
                    reg_alpha,

                "reg_lambda":
                    reg_lambda,

                "scale_pos_weight":
                    scale_pos_weight,

                "fold":
                    fold_name,

                "average_precision":
                    average_precision,

                "roc_auc":
                    roc_auc,

                "precision":
                    precision,

                "recall":
                    recall,

                "f1":
                    f1,

                "podium_hits_per_race":
                    podium_hits,
            })


            # ---------------------------------------------
            # Save predictions
            # ---------------------------------------------

            race_predictions[
                "dataset"
            ] = dataset_name

            race_predictions[
                "n_estimators"
            ] = n_estimators

            race_predictions[
                "learning_rate"
            ] = learning_rate

            race_predictions[
                "max_depth"
            ] = max_depth

            race_predictions[
                "reg_alpha"
            ] = reg_alpha

            race_predictions[
                "reg_lambda"
            ] = reg_lambda

            race_predictions[
                "fold"
            ] = fold_name

            prediction_rows.append(
                race_predictions
            )


# =========================================================
# Results
# =========================================================

scores = pd.DataFrame(
    score_rows
)


summary = (
    scores
    .groupby(
        [
            "dataset",
            "n_estimators",
            "learning_rate",
            "max_depth",
            "reg_alpha",
            "reg_lambda",
        ],
        as_index=False,
    )
    .agg(

        mean_average_precision=(
            "average_precision",
            "mean"
        ),

        mean_roc_auc=(
            "roc_auc",
            "mean"
        ),

        mean_precision=(
            "precision",
            "mean"
        ),

        mean_recall=(
            "recall",
            "mean"
        ),

        mean_f1=(
            "f1",
            "mean"
        ),

        mean_podium_hits=(
            "podium_hits_per_race",
            "mean"
        ),
    )
)


# =========================================================
# Select best configuration
# =========================================================

best_rows = (
    summary
    .sort_values(
        [
            "dataset",
            "mean_average_precision",
        ],
        ascending=[
            True,
            False,
        ],
    )
    .groupby(
        "dataset",
        as_index=False
    )
    .head(1)
    .reset_index(drop=True)
)


# =========================================================
# Save outputs
# =========================================================

scores.to_csv(
    OUT_DIR
    / "fold_scores.csv",
    index=False
)

summary.to_csv(
    OUT_DIR
    / "hyperparameter_summary.csv",
    index=False
)

best_rows.to_csv(
    OUT_DIR
    / "selected_configurations.csv",
    index=False
)


predictions = pd.concat(
    prediction_rows,
    ignore_index=True
)

predictions.to_csv(
    OUT_DIR
    / "validation_predictions.csv",
    index=False
)


# =========================================================
# Report
# =========================================================

print("\n" + "=" * 70)
print("XGBOOST SELECTED CONFIGURATIONS")
print("=" * 70)

print(
    best_rows[
        [
            "dataset",
            "n_estimators",
            "learning_rate",
            "max_depth",
            "reg_alpha",
            "reg_lambda",
            "mean_average_precision",
            "mean_roc_auc",
            "mean_f1",
            "mean_podium_hits",
        ]
    ]
    .to_string(
        index=False
    )
)


print("\nNo 2025 outcome was used.")

print(
    "Configurations were selected only from "
    "the 2023-2024 chronological folds."
)

print(
    "\nXGBOOST DEVELOPMENT COMPLETE"
)