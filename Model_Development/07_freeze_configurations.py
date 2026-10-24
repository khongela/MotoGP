from pathlib import Path
import json


ROOT = Path(__file__).resolve().parents[1]

OUT_PATH = (
    ROOT
    / "results/model_development/frozen_configurations.json"
)


# =========================================================
# These configurations were selected using ONLY
# the 2023-2024 chronological validation folds.
#
# They must not be changed after viewing 2025 results.
# =========================================================

FROZEN_CONFIGURATIONS = {

    "logistic_regression": {

        "selected_15": {
            "C": 1.0,
            "class_weight": "balanced",
            "max_iter": 5000,
            "random_state": 42,
        },

        "full_23": {
            "C": 1.0,
            "class_weight": "balanced",
            "max_iter": 5000,
            "random_state": 42,
        },

        "enhanced_54": {
            "C": 0.01,
            "class_weight": "balanced",
            "max_iter": 5000,
            "random_state": 42,
        },
    },


    "random_forest": {

        "selected_15": {
            "n_estimators": 300,
            "max_depth": 8,
            "min_samples_leaf": 4,
            "min_samples_split": 10,
            "max_features": "sqrt",
            "class_weight": "balanced",
            "random_state": 42,
        },

        "full_23": {
            "n_estimators": 300,
            "max_depth": None,
            "min_samples_leaf": 4,
            "min_samples_split": 10,
            "max_features": "sqrt",
            "class_weight": "balanced",
            "random_state": 42,
        },

        "enhanced_54": {
            "n_estimators": 300,
            "max_depth": 8,
            "min_samples_leaf": 4,
            "min_samples_split": 2,
            "max_features": "sqrt",
            "class_weight": "balanced",
            "random_state": 42,
        },
    },


    "xgboost": {

        "selected_15": {
            "n_estimators": 350,
            "learning_rate": 0.02,
            "max_depth": 2,
            "reg_alpha": 0.2,
            "reg_lambda": 8.0,
            "min_child_weight": 1,
            "subsample": 0.85,
            "colsample_bytree": 1.0,
            "random_state": 42,
        },

        "full_23": {
            "n_estimators": 200,
            "learning_rate": 0.02,
            "max_depth": 2,
            "reg_alpha": 0.0,
            "reg_lambda": 8.0,
            "min_child_weight": 1,
            "subsample": 0.85,
            "colsample_bytree": 1.0,
            "random_state": 42,
        },

        "enhanced_54": {
            "n_estimators": 350,
            "learning_rate": 0.02,
            "max_depth": 2,
            "reg_alpha": 0.0,
            "reg_lambda": 8.0,
            "min_child_weight": 1,
            "subsample": 0.85,
            "colsample_bytree": 1.0,
            "random_state": 42,
        },
    },


    "mlp": {

        "selected_15": {
            "hidden_layer_sizes": [16],
            "activation": "tanh",
            "alpha": 0.001,
            "learning_rate_init": 0.001,
            "batch_size": 16,
            "max_iter": 1500,
            "tol": 0.0001,
            "n_iter_no_change": 40,
            "random_state": 42,
        },

        "full_23": {
            "hidden_layer_sizes": [32],
            "activation": "tanh",
            "alpha": 0.01,
            "learning_rate_init": 0.0003,
            "batch_size": 16,
            "max_iter": 1500,
            "tol": 0.0001,
            "n_iter_no_change": 40,
            "random_state": 42,
        },

        "enhanced_54": {
            "hidden_layer_sizes": [32],
            "activation": "relu",
            "alpha": 0.001,
            "learning_rate_init": 0.0003,
            "batch_size": 16,
            "max_iter": 1500,
            "tol": 0.0001,
            "n_iter_no_change": 40,
            "random_state": 42,
        },
    },
}


# =========================================================
# Metadata
# =========================================================

OUTPUT = {

    "status": "FROZEN",

    "development_seasons": [
        2023,
        2024,
    ],

    "test_season": 2025,

    "selection_metric":
        "mean_average_precision",

    "validation_method":
        "three expanding chronological folds",

    "folds": {
        "fold_1": "28 train races -> 4 validation races",
        "fold_2": "32 train races -> 4 validation races",
        "fold_3": "36 train races -> 4 validation races",
    },

    "configurations":
        FROZEN_CONFIGURATIONS,
}


with open(
    OUT_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        OUTPUT,
        f,
        indent=4,
    )


print("=" * 70)
print("MODEL CONFIGURATIONS FROZEN")
print("=" * 70)

print(
    "Development seasons: 2023-2024"
)

print(
    "Selection metric: mean average precision"
)

print(
    "Primary test season: 2025"
)

print(
    "\n12 configurations frozen:"
)

print(
    "4 models x 3 feature sets"
)

print(
    f"\nSaved to:\n{OUT_PATH}"
)

print(
    "\nDo not change these configurations "
    "after inspecting 2025 results."
)