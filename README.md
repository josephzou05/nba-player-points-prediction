# NBA Player Points Prediction

This project predicts NBA player points using historical performance and game context features.  
I compare a linear regression baseline against a tree-based model and evaluate calibration using Stephen Curry as a case study.

## Models
- Linear Regression (baseline)
- Gradient Boosting Regressor (tree-based)

## Features
- Rolling scoring averages
- Rolling minutes
- Opponent-specific performance
- Home vs away context

## Evaluation
Models are evaluated using RMSE and MAE on a held-out test set, with additional visual analysis for Stephen Curry.

## Structure
- `src/01_dataIngestion.py` – fetches NBA game logs
- `src/02_featureEngineering.py` – builds modeling dataset
- `src/03_baselineLinearModel.py` – linear regression baseline
- `src/04_treeBasedModel.py` – tree-based model
- `src/visualizations.py` – feature importance and calibration plots
