import os
import math
import joblib
import pandas as pandas
import numpy as numpy

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor


# -----------------------------
# Shared Utilities
# -----------------------------

def computeExponentiallyWeightedAverage(series, span):
    return series.ewm(span=span, adjust=False).mean()


# -----------------------------
# Feature Engineering
# -----------------------------

def engineerEfficiencyFeatures(dataFrame):
    dataFrame = dataFrame.sort_values(["PlayerId", "GAME_DATE"])

    dataFrame["pointsPerMinute"] = dataFrame["PTS"] / dataFrame["MIN"]
    dataFrame["usagePerMinute"] = dataFrame["FGA"] / dataFrame["MIN"]
    dataFrame.replace([numpy.inf, -numpy.inf], numpy.nan, inplace=True)

    frames = []

    for _, playerFrame in dataFrame.groupby("PlayerId"):
        playerFrame = playerFrame.copy()

        playerFrame["ewmPointsPerMinute"] = (
            computeExponentiallyWeightedAverage(playerFrame["pointsPerMinute"], 10)
            .shift(1)
        )

        playerFrame["ewmUsagePerMinute"] = (
            computeExponentiallyWeightedAverage(playerFrame["usagePerMinute"], 10)
            .shift(1)
        )

        playerFrame["ewmMinutes"] = (
            computeExponentiallyWeightedAverage(playerFrame["MIN"], 10)
            .shift(1)
        )

        playerFrame["careerPointsPerMinute"] = (
            playerFrame["pointsPerMinute"]
            .expanding()
            .mean()
            .shift(1)
        )

        frames.append(playerFrame)

    dataFrame = pandas.concat(frames)
    dataFrame["isHomeGame"] = dataFrame["MATCHUP"].str.contains("vs").astype(int)

    efficiencyFeatures = [
        "ewmPointsPerMinute",
        "ewmUsagePerMinute",
        "ewmMinutes",
        "careerPointsPerMinute",
        "isHomeGame"
    ]

    modelFrame = dataFrame.dropna(subset=efficiencyFeatures + ["pointsPerMinute"])

    return modelFrame, efficiencyFeatures


def engineerMinutesFeatures(dataFrame):
    dataFrame = dataFrame.sort_values(["PlayerId", "GAME_DATE"])

    frames = []

    for _, playerFrame in dataFrame.groupby("PlayerId"):
        playerFrame = playerFrame.copy()

        playerFrame["ewmMinutes"] = (
            computeExponentiallyWeightedAverage(playerFrame["MIN"], 10)
            .shift(1)
        )

        playerFrame["minutesStdLast5"] = (
            playerFrame["MIN"]
            .rolling(5)
            .std()
            .shift(1)
        )

        playerFrame["careerMinutes"] = (
            playerFrame["MIN"]
            .expanding()
            .mean()
            .shift(1)
        )

        frames.append(playerFrame)

    dataFrame = pandas.concat(frames)
    dataFrame["isHomeGame"] = dataFrame["MATCHUP"].str.contains("vs").astype(int)

    minutesFeatures = [
        "ewmMinutes",
        "minutesStdLast5",
        "careerMinutes",
        "isHomeGame"
    ]

    modelFrame = dataFrame.dropna(subset=minutesFeatures + ["MIN"])

    return modelFrame, minutesFeatures


# -----------------------------
# Main Pipeline
# -----------------------------

def main():
    print("Loading data...")
    dataFrame = pandas.read_csv("data/raw/playerGameLogs.csv")
    dataFrame["GAME_DATE"] = pandas.to_datetime(dataFrame["GAME_DATE"])

    print("Engineering efficiency features...")
    efficiencyFrame, efficiencyFeatures = engineerEfficiencyFeatures(dataFrame)

    print("Engineering minutes features...")
    minutesFrame, minutesFeatures = engineerMinutesFeatures(dataFrame)

    # Align datasets
    commonIndex = efficiencyFrame.index.intersection(minutesFrame.index)
    efficiencyFrame = efficiencyFrame.loc[commonIndex]
    minutesFrame = minutesFrame.loc[commonIndex]

    XEff = efficiencyFrame[efficiencyFeatures]
    yEff = efficiencyFrame["pointsPerMinute"]

    XMin = minutesFrame[minutesFeatures]
    yMin = minutesFrame["MIN"]

    XEffTrain, XEffTest, yEffTrain, yEffTest = train_test_split(
        XEff, yEff, test_size=0.2, random_state=42
    )

    XMinTrain, XMinTest, yMinTrain, yMinTest = train_test_split(
        XMin, yMin, test_size=0.2, random_state=42
    )

    print("\nTraining efficiency model...")
    efficiencyModel = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
    efficiencyModel.fit(XEffTrain, yEffTrain)

    print("Training minutes model...")
    minutesModel = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
    minutesModel.fit(XMinTrain, yMinTrain)

    predictedPpm = efficiencyModel.predict(XEffTest)
    predictedMinutes = minutesModel.predict(XMinTest)

    predictedPoints = predictedPpm * predictedMinutes
    actualPoints = efficiencyFrame.loc[XEffTest.index, "PTS"]

    rmse = math.sqrt(mean_squared_error(actualPoints, predictedPoints))

    print("\nCombined Model Performance")
    print(f"Converted Points RMSE: {rmse:.2f}")

    print("\nError Decomposition")
    print(f"Efficiency RMSE (PPM): {math.sqrt(mean_squared_error(yEffTest, predictedPpm)):.3f}")
    print(f"Minutes RMSE: {math.sqrt(mean_squared_error(yMinTest, predictedMinutes)):.2f}")

    # -----------------------------
    # Save Models
    # -----------------------------

    os.makedirs("models", exist_ok=True)

    joblib.dump(efficiencyModel, "models/efficiencyModel.pkl")
    joblib.dump(minutesModel, "models/minutesModel.pkl")

    print("\nSaved models:")
    print(" - models/efficiencyModel.pkl")
    print(" - models/minutesModel.pkl")


if __name__ == "__main__":
    main()
