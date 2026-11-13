# MotoGP Podium Prediction Using Machine Learning

A machine learning research project investigating whether historical MotoGP race data can be used to predict which riders will finish on the podium.

The task is formulated as a **binary classification problem**:

- `1` — the rider finishes 1st, 2nd, or 3rd
- `0` — the rider does not finish on the podium

The project compares **Logistic Regression, Random Forest, XGBoost, and a Multilayer Perceptron (MLP)** using chronologically structured validation and multiple feature sets.

## Research Question

**How effectively can machine learning models predict whether a MotoGP rider will achieve a podium finish using historical race data?**

The project focuses not only on predictive performance, but also on how feature engineering, historical rider form, qualifying performance, circuit history, sprint results, and lap-level information contribute to podium prediction.

## Dataset

The modelling dataset covers the **2023–2026 MotoGP seasons** and contains **1,602 rider-race observations across 74 races**.

| Season | Rider-race entries | Races | Riders | Podiums |
|---|---:|---:|---:|---:|
| 2023 | 423 | 20 | 31 | 60 |
| 2024 | 444 | 20 | 29 | 60 |
| 2025 | 475 | 22 | 29 | 66 |
| 2026 | 260 | 12 | 29 | 36 |

The project contains both race-level data and lap-level data. Lap data is transformed into pre-race performance features before being combined with the main modelling dataset.

## Feature Sets

Three feature sets are evaluated.

### Selected 15

A smaller set of 15 race-level predictors containing variables such as:

- Grid position
- Qualifying gap
- Team and manufacturer
- Circuit
- Sprint result
- Championship points before the race
- Recent finishing position
- Recent podium rate
- Recent DNF rate
- Recent race points

### Full 23

The complete set of 23 available race-level predictors, including additional historical form and circuit-specific information.

### Enhanced 54

The 23 race-level predictors combined with **31 engineered lap-derived features**.

The additional features represent information available before the main race, including:

- Practice and qualifying pace
- Gap to the fastest rider
- Qualifying progression
- Sector performance
- Top speed
- Long-run pace and consistency
- Tyre information
- Teammate comparisons
- Historical pace and top-speed form

Current-race information that would only be known after the race is excluded to prevent target leakage.

## Experimental Design

MotoGP is temporal by nature, so random train/test splitting is avoided.

The seasons are assigned the following roles:

| Data | Role |
|---|---|
| 2023–2024 | Model development and validation |
| 2025 | Held-out temporal test set |
| 2026 | Supplementary data |

Model development uses **three expanding chronological validation folds**:

```text
Fold 1: 28 training races -> 4 validation races
Fold 2: 32 training races -> 4 validation races
Fold 3: 36 training races -> 4 validation races
```

This ensures that every validation race occurs chronologically after the races used to train the model.

Model configurations are selected using only the 2023–2024 development data and are frozen before evaluation on the held-out 2025 season.

## Models

Four supervised classification models are evaluated:

### Logistic Regression

Used as an interpretable linear benchmark and to determine whether the podium prediction problem can already be modelled effectively without complex non-linear methods.

### Random Forest

An ensemble of decision trees capable of modelling non-linear relationships and interactions between race variables.

### XGBoost

A gradient-boosted tree model designed to sequentially correct prediction errors and capture complex relationships in structured data.

### Multilayer Perceptron

A feed-forward neural network used to evaluate whether a neural approach can learn useful non-linear representations from the available data.

## Preprocessing

Preprocessing is fitted separately inside each training fold to prevent information from the validation set leaking into model training.

The pipeline includes:

- Median imputation for missing numerical values
- Most-frequent imputation for categorical values
- One-hot encoding of categorical variables
- Feature scaling where required by the model

All randomised models use a fixed random seed for reproducibility.

## Evaluation

Because podium finishers form a minority of all rider-race entries, accuracy alone is not used as the primary evaluation measure.

The main model-selection metric is **Average Precision (AP)**.

Additional metrics include:

- ROC-AUC
- Precision
- Recall
- F1-score
- Mean podium hits per race

For race-level evaluation, each model assigns a podium probability to every rider in a race. The **three riders with the highest probabilities are selected as the predicted podium**.

This makes the evaluation reflect the actual MotoGP podium structure rather than applying a single probability threshold independently to every rider.

## Validation Results

Best configuration for each model during the 2023–2024 chronological validation stage:

| Model | Feature Set | Average Precision | ROC-AUC | F1 | Podium Hits / Race |
|---|---|---:|---:|---:|---:|
| Logistic Regression | Full 23 | **0.767** | 0.947 | 0.722 | 2.17 |
| Random Forest | Full 23 | 0.759 | **0.956** | **0.750** | **2.25** |
| XGBoost | Full 23 | 0.755 | 0.942 | 0.667 | 2.00 |
| MLP | Enhanced 54 | 0.741 | 0.948 | 0.667 | 2.00 |

Logistic Regression currently produces the highest mean validation Average Precision, while Random Forest achieves the strongest race-level F1-score and average number of correctly predicted podium riders.

These results also show that increased model complexity does not automatically improve predictive performance.

**Note:** These are cross-validation results on the development seasons, not results from the held-out 2025 test season.

## Repository Structure

```text
MotoGP/
│
├── Data_Management/
│   ├── 00_check_project.py
│   ├── 01_inspect_laps.py
│   ├── ...
│   └── 08_find_removed_entries.py
│
├── Feature_Engineering/
│   ├── 01_build_lap_features.py
│   ├── 02_validate_lap_features.py
│   └── 03_define_lap_candidates.py
│
├── Feature_Selection/
│   └── 01_define_feature_sets.py
│
├── Model_Development/
│   ├── 01_define_splits.py
│   ├── 02_preprocessing.py
│   ├── 03_logistic_regression.py
│   ├── 04_random_forest.py
│   ├── 05_xgboost.py
│   ├── 06_mlp.py
│   └── 07_freeze_configurations.py
│
├── data/
│   ├── interim/
│   │   ├── master.csv
│   │   └── stage2_cleaned_rider_races.csv
│   │
│   └── processed/
│       ├── laps.csv
│       ├── model_features_2023_2026.csv
│       └── model_features_with_laps_2023_2026.csv
│
└── results/
    ├── feature_analysis/
    └── model_development/
```

## Running the Project

Clone the repository:

```bash
git clone https://github.com/khongela/MotoGP.git
cd MotoGP
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Install the main dependencies:

```bash
pip install pandas numpy scikit-learn xgboost
```

Run the main experimental pipeline in order:

```bash
python Feature_Engineering/01_build_lap_features.py
python Feature_Engineering/02_validate_lap_features.py
python Feature_Engineering/03_define_lap_candidates.py

python Feature_Selection/01_define_feature_sets.py

python Model_Development/01_define_splits.py
python Model_Development/02_preprocessing.py
python Model_Development/03_logistic_regression.py
python Model_Development/04_random_forest.py
python Model_Development/05_xgboost.py
python Model_Development/06_mlp.py
python Model_Development/07_freeze_configurations.py
```

The generated feature-analysis, validation, hyperparameter, prediction, and configuration files are stored under `results/`.

## Reproducibility

The experiment is designed to minimise data leakage and preserve the temporal structure of MotoGP racing.

In particular:

- Future races are never used to predict earlier races.
- Preprocessing is fitted using training data only.
- Current-race post-outcome information is excluded from predictors.
- Model selection is performed using chronological validation.
- The 2025 season is separated as an unseen temporal test set.
- Selected model configurations are frozen before test evaluation.

## Author

**Msesenyane Makhongela**  
School of Computer Science and Applied Mathematics  
University of the Witwatersrand

This repository forms part of the research project **“Evaluating Machine Learning Algorithms for MotoGP.”**
