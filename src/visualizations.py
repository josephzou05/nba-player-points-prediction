import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.inspection import permutation_importance


def plot_linear_feature_importance(model, feature_names, top_n=15):
    coefs = np.abs(model.coef_)

    fi = (
        pd.DataFrame({
            "feature": feature_names,
            "importance": coefs
        })
        .sort_values("importance", ascending=False)
        .head(top_n)
    )

    plt.figure(figsize=(8, 6))
    plt.barh(fi["feature"], fi["importance"])
    plt.gca().invert_yaxis()
    plt.xlabel("Absolute Coefficient Value")
    plt.title("Linear Model Feature Importance")
    plt.tight_layout()
    plt.show()


def plot_tree_feature_importance(model, X, y, feature_names, top_n=15):
    result = permutation_importance(
        model,
        X,
        y,
        n_repeats=10,
        random_state=42,
        n_jobs=-1
    )

    fi = (
        pd.DataFrame({
            "feature": feature_names,
            "importance": result.importances_mean
        })
        .sort_values("importance", ascending=False)
        .head(top_n)
    )

    plt.figure(figsize=(8, 6))
    plt.barh(fi["feature"], fi["importance"])
    plt.gca().invert_yaxis()
    plt.xlabel("Permutation Importance")
    plt.title("Tree Model Feature Importance")
    plt.tight_layout()
    plt.show()


def plot_actual_vs_predicted(y_true, y_pred, title):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([0, max_val], [0, max_val], linestyle="--")
    plt.xlabel("Actual Points")
    plt.ylabel("Predicted Points")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def curry_actual_vs_predicted(df, y_pred, model_name):
    curry_df = df[df["PlayerName"] == "Stephen Curry"]

    common_idx = curry_df.index.intersection(y_pred.index)

    if len(common_idx) == 0:
        raise ValueError("No Stephen Curry games found in the test set.")

    plot_actual_vs_predicted(
        curry_df.loc[common_idx, "PTS"],
        y_pred.loc[common_idx],
        f"{model_name}: Actual vs Predicted (Stephen Curry)"
    )

