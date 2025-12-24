import time
import pandas as pd
from tqdm import tqdm
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players


def fetchPlayerGameLogs(playerId, seasonList):
    gameLogFrames = []

    for season in seasonList:
        try:
            gameLog = playergamelog.PlayerGameLog(
                player_id=playerId,
                season=season
            ).get_data_frames()[0]

            gameLogFrames.append(gameLog)
            time.sleep(0.6)

        except Exception as error:
            print(f"Error fetching player {playerId}: {error}")

    if len(gameLogFrames) == 0:
        return None

    return pd.concat(gameLogFrames, ignore_index=True)


def main():
    activePlayers = players.get_active_players()
    playerTable = pd.DataFrame(activePlayers)

    seasonList = ["2022-23", "2023-24", "2024-25"]
    allGameLogs = []

    for _, playerRow in tqdm(playerTable.iterrows(), total=len(playerTable)):
        playerId = playerRow["id"]
        playerName = playerRow["full_name"]

        playerLogs = fetchPlayerGameLogs(playerId, seasonList)

        if playerLogs is not None:
            playerLogs["PlayerId"] = playerId
            playerLogs["PlayerName"] = playerName
            allGameLogs.append(playerLogs)

    fullDataset = pd.concat(allGameLogs, ignore_index=True)
    fullDataset.to_csv("data/raw/playerGameLogs.csv", index=False)

    print("Saved data/raw/playerGameLogs.csv")


if __name__ == "__main__":
    main()
