# Explainable Fraud Rule Miner

A Python and Streamlit prototype for finding human-readable fraud rules in IEEE-CIS transaction data and expressing them as SQL conditions.

## Workflow

1. Load raw identity and transaction tables, or a merged dataset.
2. Preprocess the data and create features.
3. Select features and mine decision-tree rules.
4. Review candidate rules using validation precision, lift, support, and votes.
5. Inspect SQL conditions and export the filtered rule catalog as CSV.

## Repository layout

- `app.py`: Streamlit interface and pipeline orchestration
- `scripts/preprocess_data.py`: data preparation and splitting
- `scripts/feature_engineering.py`: feature creation
- `scripts/feature_select.py`: feature selection
- `scripts/decisionTree.py`: bootstrapped decision-tree rule mining
- `scripts/helperFuncs.py`: rule formatting and supporting utilities
- `requirements.txt` and `environment.yml`: recorded dependency information

## Setup notes

The app expects IEEE-CIS data supplied by the user; datasets are not bundled.

The code uses Python, pandas, NumPy, scikit-learn, XGBoost, statsmodels, and Streamlit. The current `requirements.txt` does not include every imported dependency, including Streamlit and XGBoost. The Conda environment file records an earlier local environment and may need platform-specific adjustments.

A clean installation and end-to-end run have not been revalidated as part of this documentation update.

## Scope and limitations

This is an exploratory prototype. Rules and evaluation results depend on the data, split, and configuration. SQL output is intended for review and testing; production use would require independent validation, monitoring, and integration work.

