import time
from typing import List

import numpy as np
import pandas as pd
from api import config
from api.MachineLearning import data
from api.MachineLearning.classifier import classifier
import logging

logger = logging.getLogger(__name__)


def predict(
    hiscores,
    names,
    binary_classifier: classifier,
    multi_classifier: classifier,
) -> List[dict]:
    """
    This function takes in a list of hiscores, a list of names, and two classifiers.
    It then predicts the probability of each hiscore being a bot or a real player.
    It then returns a list of dictionaries with the predictions.
    The predictions are based on the binary classifier, and the multi classifier.
    The binary classifier is used to predict if the player is a real player or a bot.
    The multi classifier is used to predict the type of bot.
    If the binary classifier predicts that the player is a real player, then the multi classifier is not used.
    If the binary classifier predicts that the player is a bot, then the multi classifier is used to predict the type of bot.
    The output is a list of dictionaries with the predictions.
    """
    logger.debug("Predicting hiscores for players")
    hiscores = data.hiscoreData(hiscores)
    low_level = hiscores.df_low.index
    hiscores = hiscores.features()

    logger.debug("Predicting binary for players")
    # binary prediction
    binary_pred = binary_classifier.predict_proba(hiscores)
    binary_pred = pd.DataFrame(
        binary_pred, index=hiscores.index, columns=["Real_Player", "Unknown_bot"]
    )

    # multi prediction
    logger.debug("Predicting multi for players")
    multi_pred = multi_classifier.predict_proba(hiscores)
    multi_pred = pd.DataFrame(
        multi_pred, index=hiscores.index, columns=np.unique(config.LABELS)
    )

    # remove real players from multi
    logger.debug("Removing real players from multi for players")
    real_players = binary_pred.query("Real_Player > 0.5").index
    mask = ~(multi_pred.index.isin(real_players))
    multi_pred = multi_pred[mask]

    # remove bots from real players
    logger.debug("Removing bots from binary for players")
    bots = multi_pred.index
    mask = ~(binary_pred.index.isin(bots))
    binary_pred = binary_pred[mask]

    # combine binary & player_pred
    logger.debug("Combining binary and multi for players")
    output = pd.DataFrame(names).set_index("id")
    output = output.merge(binary_pred, left_index=True, right_index=True, how="left")

    output = output.merge(
        multi_pred,
        left_index=True,
        right_index=True,
        suffixes=["", "_multi"],
        how="left",
    )

    # cleanup predictions
    logger.debug("Cleaning up predictions for players")
    mask = output["Real_Player"].isna()  # all multi class predictions

    # cleanup multi suffixes
    output.loc[mask, "Unknown_bot"] = output[mask]["Unknown_bot_multi"]
    output.loc[mask, "Real_Player"] = output[mask]["Real_Player_multi"]

    output.drop(columns=["Real_Player_multi", "Unknown_bot_multi"], inplace=True)
    output.fillna(0, inplace=True)

    # add Predictions, Predicted_confidence, created
    logger.debug("Adding Predictions, Predicted_confidence, created for players")
    columns = [c for c in output.columns if c != "name"]
    output["Predicted_confidence"] = round(output[columns].max(axis=1) * 100, 2)
    output["Prediction"] = output[columns].idxmax(axis=1)
    output["created"] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    output.reset_index(inplace=True)

    # low level player predictions are not accurate
    logger.debug("Removing low level players for players")

    mask = output["id"].isin(low_level)
    output.loc[mask, "Prediction"] = "Stats_Too_Low"

    len_too_low_players = len(output[output["Prediction"] == "Stats_Too_Low"])
    logger.debug(f"Len low level players {len_too_low_players}")

    # cut off name
    output["name"] = output["name"].astype(str).str[:12]

    # parsing values
    output[columns] = round(output[columns] * 100, 2)

    # convert output to dict
    output = output.to_dict(orient="records")
    return output
