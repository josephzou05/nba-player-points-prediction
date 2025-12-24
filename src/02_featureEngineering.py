import pandas as pd


def loadRawGameLogs():
    return pd.read_csv("data/raw/playerGameLogs.csv")


def cleanAndSortGameLogs(gameLogs):
    gameLogs["GAME_DATE"] = pd.to_datetime(gameLogs["GAME_DATE"])
    gameLogs = gameLogs.sort_values(
        by=["PlayerId", "GAME_DATE"]
    ).reset_index(drop=True)
    return gameLogs


def addGameContextFeatures(gameLogs):
    gameLogs["isHomeGame"] = gameLogs["MATCHUP"].str.contains("vs").astype(int)

    def extractOpponent(matchup):
        if "vs." in matchup:
            return matchup.split("vs.")[-1].strip()
        return matchup.split("@")[-1].strip()

    gameLogs["opponentTeamAbbreviation"] = gameLogs["MATCHUP"].apply(extractOpponent)
    return gameLogs


def createRollingFeatures(gameLogs):
    for window in [5, 10]:
        gameLogs[f"pointsLast{window}"] = (
            gameLogs.groupby("PlayerId")["PTS"]
            .rolling(window)
            .mean()
            .reset_index(level=0, drop=True)
        )

        gameLogs[f"minutesLast{window}"] = (
            gameLogs.groupby("PlayerId")["MIN"]
            .rolling(window)
            .mean()
            .reset_index(level=0, drop=True)
        )

    return gameLogs


def createUsageFeatures(gameLogs):
    gameLogs["pointsPerMinute"] = gameLogs["PTS"] / gameLogs["MIN"]
    gameLogs["pointsPerMinute"] = gameLogs["pointsPerMinute"].replace(
        [float("inf")], 0
    )

    for window in [5, 10]:
        gameLogs[f"pointsPerMinuteLast{window}"] = (
            gameLogs.groupby("PlayerId")["pointsPerMinute"]
            .rolling(window)
            .mean()
            .reset_index(level=0, drop=True)
        )

    return gameLogs


def createOpponentRollingFeatures(gameLogs):
    rollingWindow = 3

    gameLogs["pointsVsOpponentLast3"] = (
        gameLogs
        .groupby(["PlayerId", "opponentTeamAbbreviation"])["PTS"]
        .rolling(rollingWindow)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )

    gameLogs["minutesVsOpponentLast3"] = (
        gameLogs
        .groupby(["PlayerId", "opponentTeamAbbreviation"])["MIN"]
        .rolling(rollingWindow)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )

    return gameLogs


def createPredictionTarget(gameLogs):
    gameLogs["pointsNextGame"] = (
        gameLogs.groupby("PlayerId")["PTS"].shift(-1)
    )
    return gameLogs


def saveProcessedDataset(gameLogs):
    gameLogs.to_csv(
        "data/processed/modelDataset.csv",
        index=False
    )
    print(f"Saved data/processed/modelDataset.csv with {len(gameLogs)} rows")


def main():
    gameLogs = loadRawGameLogs()
    gameLogs = cleanAndSortGameLogs(gameLogs)
    gameLogs = addGameContextFeatures(gameLogs)
    gameLogs = createRollingFeatures(gameLogs)
    gameLogs = createUsageFeatures(gameLogs)
    gameLogs = createOpponentRollingFeatures(gameLogs)
    gameLogs = createPredictionTarget(gameLogs)
    saveProcessedDataset(gameLogs)


if __name__ == "__main__":
    main()
