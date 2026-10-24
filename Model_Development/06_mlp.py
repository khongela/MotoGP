from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)


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
    / "results/model_development/mlp"
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
# MLP search space
# =========================================================

HIDDEN_LAYERS = [
    (16,),
    (32,),
]

ACTIVATIONS = [
    "tanh",
    "relu",
]

ALPHAS = [
    0.001,
    0.01,
]

LEARNING_RATES = [
    0.0003,
    0.001,
]

BATCH_SIZE = 16


CONFIGURATIONS = list(
    product(
        HIDDEN_LAYERS,
        ACTIVATIONS,
        ALPHAS,
        LEARNING_RATES,
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
            (
                "scaler",
                StandardScaler()
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
# Oversample podium observations
#
# Only training data is oversampled.
# Validation data is never modified.
# =========================================================

def oversample_training(
    X,
    y,
    random_state=42,
):

    y = np.asarray(y)

    positive_indices = np.where(
        y == 1
    )[0]

    negative_indices = np.where(
        y == 0
    )[0]


    if len(positive_indices) >= len(negative_indices):
        return X, y


    rng = np.random.RandomState(
        random_state
    )

    additional_positive_indices = rng.choice(
        positive_indices,
        size=(
            len(negative_indices)
            - len(positive_indices)
        ),
        replace=True,
    )


    balanced_indices = np.concatenate(
        [
            negative_indices,
            positive_indices,
            additional_positive_indices,
        ]
    )


    rng.shuffle(
        balanced_indices
    )


    return (
        X[balanced_indices],
        y[balanced_indices],
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
        hidden_layer,
        activation,
        alpha,
        learning_rate,
    ) in CONFIGURATIONS:

        print(
            "\n"
            f"hidden={hidden_layer}, "
            f"activation={activation}, "
            f"alpha={alpha}, "
            f"lr={learning_rate}"
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


            X_train_raw = train[
                features
            ]

            y_train = train[
                "podium_finish"
            ].to_numpy()


            X_validation_raw = validation[
                features
            ]

            y_validation = validation[
                "podium_finish"
            ].to_numpy()


            # ---------------------------------------------
            # Preprocessing
            #
            # Fit on training data only
            # ---------------------------------------------

            preprocessor = (
                build_preprocessor(
                    train,
                    features
                )
            )


            X_train = (
                preprocessor.fit_transform(
                    X_train_raw
                )
            )

            X_validation = (
                preprocessor.transform(
                    X_validation_raw
                )
            )


            # ---------------------------------------------
            # Oversample training observations only
            # ---------------------------------------------

            (
                X_train_balanced,
                y_train_balanced,
            ) = oversample_training(
                X_train,
                y_train,
                random_state=42,
            )


            # ---------------------------------------------
            # MLP
            # ---------------------------------------------

            model = MLPClassifier(

                hidden_layer_sizes=
                    hidden_layer,

                activation=
                    activation,

                solver=
                    "adam",

                alpha=
                    alpha,

                batch_size=
                    BATCH_SIZE,

                learning_rate_init=
                    learning_rate,

                max_iter=
                    1500,

                tol=
                    1e-4,

                n_iter_no_change=
                    40,

                early_stopping=
                    False,

                random_state=
                    42,
            )


            # ---------------------------------------------
            # Train
            # ---------------------------------------------

            model.fit(
                X_train_balanced,
                y_train_balanced
            )


            # ---------------------------------------------
            # Predict
            # ---------------------------------------------

            probabilities = (
                model.predict_proba(
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
            # Save scores
            # ---------------------------------------------

            score_rows.append({

                "dataset":
                    dataset_name,

                "hidden_layer":
                    hidden_layer[0],

                "activation":
                    activation,

                "alpha":
                    alpha,

                "learning_rate":
                    learning_rate,

                "batch_size":
                    BATCH_SIZE,

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

                "iterations":
                    model.n_iter_,
            })


            # ---------------------------------------------
            # Save predictions
            # ---------------------------------------------

            race_predictions[
                "dataset"
            ] = dataset_name

            race_predictions[
                "hidden_layer"
            ] = hidden_layer[0]

            race_predictions[
                "activation"
            ] = activation

            race_predictions[
                "alpha"
            ] = alpha

            race_predictions[
                "learning_rate"
            ] = learning_rate

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
            "hidden_layer",
            "activation",
            "alpha",
            "learning_rate",
            "batch_size",
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

        mean_iterations=(
            "iterations",
            "mean"
        ),
    )
)


# =========================================================
# Select configuration using mean AP
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
print("MLP SELECTED CONFIGURATIONS")
print("=" * 70)

print(
    best_rows[
        [
            "dataset",
            "hidden_layer",
            "activation",
            "alpha",
            "learning_rate",
            "batch_size",
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
    "Podium oversampling was applied only "
    "to training observations."
)

print(
    "\nMLP DEVELOPMENT COMPLETE"
)