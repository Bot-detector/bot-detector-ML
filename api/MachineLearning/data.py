import logging
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

skills = [
    "attack",
    "defence",
    "strength",
    "hitpoints",
    "ranged",
    "prayer",
    "magic",
    "cooking",
    "woodcutting",
    "fletching",
    "fishing",
    "firemaking",
    "crafting",
    "smithing",
    "mining",
    "herblore",
    "agility",
    "thieving",
    "slayer",
    "farming",
    "runecraft",
    "hunter",
    "construction",
]
minigames = [
    "league",
    "bounty_hunter_hunter",
    "bounty_hunter_rogue",
    "cs_all",
    "cs_beginner",
    "cs_easy",
    "cs_medium",
    "cs_hard",
    "cs_elite",
    "cs_master",
    "lms_rank",
    "soul_wars_zeal",
]


class hiscoreData:
    """
    This class is responsible for cleaning data & creating features.
    """

    def __init__(self, data: List[dict]) -> None:
        self.df = pd.DataFrame(data)
        self.df_clean = self.df.copy()

        self.skills = skills
        self.minigames = minigames

        self.__clean()
        self.__skill_ratio()
        self.__boss_ratio()

    def __clean(self) -> None:
        """
        Cleanup the dataframe.

        This method will:
            - drop unnecessary columns
            - set the index to the player id
            - replace -1 with 0
            - create a list of bosses (not skills or minigames)
            - create a total xp column
            - create a total boss kc column
            - reduces memory of dataframe
            - fill na with 0
            - create a dataframe with only low level players (total level < 1_000_000)
        """
        self.df_clean.drop(columns=["id", "timestamp", "ts_date"], inplace=True)
        # set index to player id
        self.df_clean.set_index(["Player_id"], inplace=True)

        # if not on the hiscores it shows -1, replace with 0
        self.df_clean = self.df_clean.replace(-1, 0)

        # bosses
        self.bosses = [
            c for c in self.df_clean.columns if c not in ["total"] + skills + minigames
        ]
        # total is not always on hiscores, create a total xp column
        self.df_clean["total"] = self.df_clean[self.skills].sum(axis=1)

        # create a total boss kc column
        self.df_clean["boss_total"] = (
            self.df_clean[self.bosses].sum(axis=1).astype(np.int32)
        )

        # fillna
        self.df_clean.fillna(0, inplace=True)

        # apply smaller data types to reduce memory usage
        non_total_features = [
            col for col in self.df_clean.columns if "total" not in col
        ]
        self.df_clean[non_total_features] = self.df_clean[non_total_features].astype(
            np.int32
        )

        # get low lvl players
        mask = self.df_clean["total"] < 1_000_000
        self.df_low = self.df_clean[mask].copy()

    def __skill_ratio(self):
        """
        Create a dataframe with the ratio of each skill to the total level.

        This method will:
            - create a dataframe with the index of the original dataframe
            - create a column for each skill with the ratio of the skill to the total level
            - fill na with 0
        """
        self.skill_ratio = pd.DataFrame(index=self.df_clean.index)

        total = self.df_clean["total"]

        for skill in self.skills:
            self.skill_ratio[f"{skill}/total"] = (self.df_clean[skill] / total).astype(
                np.float16
            )

        self.skill_ratio.fillna(0, inplace=True)

    def __boss_ratio(self):
        """
        Create a dataframe with the ratio of each boss to the total boss level.

        This method will:
            - create a dataframe with the index of the original dataframe
            - create a column for each boss with the ratio of the boss to the total boss level
            - fill na with 0
        """
        self.boss_ratio = pd.DataFrame(index=self.df_clean.index)

        total = self.df_clean["boss_total"]
        for boss in self.bosses:
            self.boss_ratio[f"{boss}/total"] = (self.df_clean[boss] / total).astype(
                np.float16
            )

        self.boss_ratio.fillna(0, inplace=True)

    def features(
        self, base: bool = True, skill_ratio: bool = True, boss_ratio: bool = True
    ):
        """
        Create a dataframe with the features.

        This method will:
            - create a dataframe with the index of the original dataframe
            - merge the original dataframe, the skill ratio dataframe and the boss ratio dataframe

        Parameters
        ----------
        base : bool, optional
            Whether to include the original dataframe, by default True
        skill_ratio : bool, optional
            Whether to include the skill ratio dataframe, by default True
        boss_ratio : bool, optional
            Whether to include the boss ratio dataframe, by default True

        Returns
        -------
        pd.DataFrame
            Dataframe containing the features
        """
        features = pd.DataFrame(index=self.df_clean.index)
        if base:
            features = features.merge(self.df_clean, left_index=True, right_index=True)
        if skill_ratio:
            features = features.merge(
                self.skill_ratio, left_index=True, right_index=True
            )
        if boss_ratio:
            features = features.merge(
                self.boss_ratio, left_index=True, right_index=True
            )
        return features


class playerData:
    """
    Class to handle the data from the json files.
    """

    def __init__(self, player_data: List[dict], label_data: List[dict]) -> None:
        """
        Initialize the class.

        Parameters
        ----------
        player_data : List[dict]
            List of dictionaries containing the player data
        label_data : List[dict]
            List of dictionaries containing the label data
        """
        self.df_players = pd.DataFrame(player_data)
        self.df_labels = pd.DataFrame(label_data)
        self.__clean()

    def __clean(self):
        """
        Clean the data.

        This method will:
            - set the index of the player dataframe to the player id
            - set the index of the label dataframe to the label id
            - merge the two dataframes
            - create a binary label column
        """
        # clean players
        self.df_players.set_index("id", inplace=True)

        # reduce memory of player dataframe
        small_size_columns = [
            "possible_ban",
            "confirmed_ban",
            "confirmed_player",
            "label_id",
            "label_jagex",
        ]
        self.df_players[small_size_columns] = self.df_players[
            small_size_columns
        ].astype(np.int8)

        # clean labels
        self.df_labels.set_index("id", inplace=True)

        # merge
        self.df_players = self.df_players.merge(
            self.df_labels, left_on="label_id", right_index=True
        )
        self.df_players.drop(columns=["label_id"], inplace=True)

        # binary label, 1 = bot, 0 = not bot
        self.df_players["binary_label"] = np.where(
            self.df_players["label_jagex"] == 2, 1, 0
        )

    def get(self, binary: bool = False):
        """
        Get the target data.

        This method will:
            - return the binary label or the label column

        Parameters
        ----------
        binary : bool, optional
            Whether to return the binary label or not, by default False

        Returns
        -------
        pd.DataFrame
            Dataframe containing the target data
        """
        if binary:
            out = self.df_players.loc[:, ["binary_label"]].astype(np.int8)
            out.rename(columns={"binary_label": "target"}, inplace=True)
        else:
            out = self.df_players.loc[:, ["label"]].astype("category")
            out.rename(columns={"label": "target"}, inplace=True)

        return out
