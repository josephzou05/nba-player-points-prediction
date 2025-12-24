import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

from visualizations import (
    plot_tree_feature_importance,
    curry_actual_vs_predicted
)

FEATURES = [
    "pointsLast10",
    "minutesLast10",
    "pointsPerMinuteLast10",
    "pointsVsOpponentLast3",
    "minutesVsOpponentLast3",
    "isHomeGame"
]

TARGET = "PTS"


def load_dataset():
    return pd.read_csv("data/processed/modelDataset.csv")


def main():
    df = load_dataset()
    df = df.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)


    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = pd.Series(
        model.predict(X_test),
        index=X_test.index
    )

    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print("\n=== Tree-Based Model Performance ===")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE : {mae:.2f}")


    plot_tree_feature_importance(
        model=model,
        X=X_train,
        y=y_train,
        feature_names=FEATURES
    )

    curry_actual_vs_predicted(
        df=df,
        y_pred=y_pred,
        model_name="Tree-Based Model"
    )


if __name__ == "__main__":
    main()
