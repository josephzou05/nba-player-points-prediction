# NBA Player Points Prediction

This project builds a machine learning pipeline to **predict NBA player points** using historical performance and game context.  
I compare a **linear regression baseline** against a **tree-based model** and evaluate calibration using **:contentReference[oaicite:0]{index=0}** as a focused case study.

The goal of this project is not betting automation, but **interpretable player performance modeling** and disciplined model evaluation.

---

## Project Overview

Predicting NBA scoring output requires accounting for:
- Recent form
- Playing time trends
- Opponent-specific context
- Game location (home vs away)

This repository demonstrates a clean, reproducible workflow:
1. Programmatically fetch NBA data
2. Engineer rolling and contextual features
3. Train and evaluate multiple regression models
4. Visualize feature importance and prediction calibration

---

## Models

Two primary models are implemented and compared:

- **Linear Regression (Baseline)**
  - Provides interpretability through coefficients
  - Serves as a performance benchmark

- **Tree-Based Regression (Gradient Boosting)**
  - Captures nonlinear interactions
  - Evaluated using permutation feature importance

An optional experimental two-stage model (minutes × efficiency) is included for exploration, but the main analysis focuses on the two models above.

---

## Features

Key features include:
- Rolling averages of points and minutes
- Rolling scoring efficiency (points per minute)
- Opponent-specific recent performance
- Home vs away indicator

Early-season games without sufficient historical context are excluded to avoid information leakage.

---

## Evaluation

Models are evaluated on a held-out test set using:
- **RMSE (Root Mean Squared Error)**
- **MAE (Mean Absolute Error)**

In addition to aggregate metrics, I visualize:
- Feature importance for each model
- Actual vs. predicted points for Stephen Curry (test games only)

This ensures evaluation reflects **out-of-sample performance**.

---

## Repository Structure

